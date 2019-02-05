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

    return points, sizes


def get_new_points(image, config, cur_points=None):
    return _get_new_points_global(image,
                                  config['st_global_params'],
                                  cur_points)


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

        new_points, new_sizes = get_new_points(image_1, config, corners.points)
        # print(maxCorners - len(corners.points), len(new_points))
        if new_points is not None:
            new_ids = \
                np.arange(n_tracks,
                          n_tracks + len(new_points))
            n_tracks += len(new_points)
            corners = FrameCorners(
                np.concatenate([corners.ids, new_ids.reshape(-1, 1)],
                               axis=0),
                np.concatenate([corners.points, new_points.reshape(-1, 2)],
                               axis=0),
                np.concatenate([corners.sizes, new_sizes.reshape(-1, 1)],
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
    if config['min_track_len']:
        return without_short_tracks(builder.build_corner_storage(),
                                    min_len=config['min_track_len'])
    else:
        return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
