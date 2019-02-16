#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl, filter_frame_corners
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def calc_corners_flow(image_0, image_1, corners_0, config):
    corners_1, status, err = cv2.calcOpticalFlowPyrLK(
        np.round(image_0 * 255).astype(np.uint8),
        np.round(image_1 * 255).astype(np.uint8),
        corners_0,
        None,
        **config['lk_params']
    )
    return corners_1, status


def _get_new_points_global(image, st_global_params, cur_points):
    if cur_points is not None:
        mask = np.ones(image.shape, dtype=np.uint8)
        for point in cur_points:
            cv2.circle(mask,
                       tuple(point),
                       st_global_params['minDistance'],
                       color=0,
                       thickness=cv2.FILLED)
    else:
        mask = None
    points = cv2.goodFeaturesToTrack(image,
                                     **st_global_params,
                                     mask=mask,
                                     useHarrisDetector=False)
    sizes = np.ones(len(points), dtype=np.int) * st_global_params['blockSize']

    if points is not None:
        points, sizes = points.reshape(-1, 2), sizes.reshape(-1, 1)
    return points, sizes


def _get_new_points_blocks(image, n_rows, n_cols,
                           st_blocks_params, cur_points):
    mask_global = np.ones(image.shape, dtype=np.uint8)
    if cur_points is not None:
        for point in cur_points:
            cv2.circle(mask_global,
                       tuple(point),
                       st_blocks_params['minDistance'],
                       color=0,
                       thickness=cv2.FILLED)

    points_blocks_list = []
    xs = np.linspace(0, image.shape[0], num=n_cols + 1, dtype=np.int)
    ys = np.linspace(0, image.shape[1], num=n_rows + 1, dtype=np.int)
    for xa, xb in zip(xs[:-1], xs[1:]):
        for ya, yb in zip(ys[:-1], ys[1:]):
            mask_block = cv2.rectangle(np.zeros_like(mask_global),
                                       (xa, ya),
                                       (xb, yb),
                                       color=1,
                                       thickness=cv2.FILLED)
            points_block = cv2.goodFeaturesToTrack(
                image,
                **st_blocks_params,
                mask=mask_global & mask_block,
                useHarrisDetector=False
            )
            if points_block is not None:
                points_blocks_list.append(points_block)

    if not points_blocks_list:
        return None, None
    points = np.concatenate(points_blocks_list, axis=0)
    sizes = np.ones(len(points), dtype=np.int) * st_blocks_params['blockSize']
    points, sizes = points.reshape(-1, 2), sizes.reshape(-1, 1)
    return points, sizes


def get_new_points(image, config, cur_points=None):
    new_points_global, new_sizes_global = \
        _get_new_points_global(image,
                               config['st_global_params'],
                               cur_points)

    new_cur_points = np.concatenate([cur_points, new_points_global], axis=0) \
        if cur_points is not None and new_points_global is not None \
        else cur_points if new_points_global is None else new_points_global
    new_points_blocks, new_sizes_blocks = \
        _get_new_points_blocks(image,
                               config['blocks_n_rows'],
                               config['blocks_n_cols'],
                               config['st_blocks_params'],
                               new_cur_points)

    new_points_list, new_sizes_list = [], []
    if new_points_global is not None:
        new_points_list.append(new_points_global)
        new_sizes_list.append(new_sizes_global)
    if new_points_blocks is not None:
        new_points_list.append(new_points_blocks)
        new_sizes_list.append(new_sizes_blocks)
    return np.concatenate(new_points_list, axis=0), \
        np.concatenate(new_sizes_list, axis=0)


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder,
                config: {}) -> None:
    image_0 = frame_sequence[0]
    points, sizes = get_new_points(image_0, config)
    corners = FrameCorners(
        np.arange(len(points)),
        points,
        sizes
    )
    n_tracks = len(points)
    builder.set_corners_at_frame(0, corners)

    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        next_points, status = calc_corners_flow(image_0, image_1,
                                                corners.points, config)
        # print(len(corners_), len(status))
        corners = FrameCorners(corners.ids, next_points, corners.sizes)
        corners = filter_frame_corners(corners, np.squeeze(status == 1))

        new_points, new_sizes = get_new_points(image_1,
                                               config,
                                               corners.points)
        # print(maxCorners - len(corners.points), len(new_points))
        if new_points is not None:
            new_ids = \
                np.arange(n_tracks,
                          n_tracks + len(new_points))
            n_tracks += len(new_points)
            corners = FrameCorners(
                np.concatenate([corners.ids, new_ids.reshape(-1, 1)],
                               axis=0),
                np.concatenate([corners.points, new_points],
                               axis=0),
                np.concatenate([corners.sizes, new_sizes],
                               axis=0),
            )

        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          config: {},
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param config: detection parameters config
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder, config)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder, config)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
