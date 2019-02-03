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


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder,
                config: {}) -> None:
    image_0 = frame_sequence[0]
    st_params = config['st_params']
    block_size = st_params['blockSize']
    points = cv2.goodFeaturesToTrack(image_0,
                                     **st_params,
                                     useHarrisDetector=False)
    corners = FrameCorners(
        np.arange(len(points)),
        points,
        np.ones(len(points)) * block_size
    )
    # print(len(corners_))
    builder.set_corners_at_frame(0, corners)
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        new_points, status = calc_corners_flow(image_0, image_1,
                                               corners.points, config)
        # print(len(corners_), len(status))
        corners = FrameCorners(corners.ids, new_points, corners.sizes)
        corners = filter_frame_corners(corners, np.squeeze(status == 1))
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
