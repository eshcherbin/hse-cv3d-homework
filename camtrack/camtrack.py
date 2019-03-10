#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Tuple

import numpy as np

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import *

import cv2


# TODO: calibrate everything
_HOM_ESS_RATIO_THRESHOLD = 0.5
_TRIANGULATION_PARAMETERS = TriangulationParameters(5, 5, 0.5)
_N_TRIANGULATED_POINTS_THRESHOLD = 50

_DEBUG = True


def _track_camera(corner_storage: CornerStorage,
                  intrinsic_mat: np.ndarray) \
        -> Tuple[List[np.ndarray], PointCloudBuilder]:
    view_mats_res = [eye3x4()] + [None] * (len(corner_storage) - 1)
    builder_res = PointCloudBuilder()

    # initialization
    best_stats = (-1, -2)
    init_points, init_points_ids = None, None
    init_pose, init_frame = None, None
    for frame in range(1, len(corner_storage)):
        corrs = build_correspondences(corner_storage[0], corner_storage[frame])
        E, mask_E = cv2.findEssentialMat(corrs.points_1, corrs.points_2,
                                         intrinsic_mat)
        _, mask_H = cv2.findHomography(corrs.points_1, corrs.points_2,
                                       method=cv2.RANSAC)
        hom_ess_ratio = mask_H.mean() / mask_E.mean()
        R1, R2, t = cv2.decomposeEssentialMat(E)
        poses = [Pose(r_mat, t_vec) for r_mat, t_vec in ((R1, t), (R1, -t),
                                                         (R2, t), (R2, -t))]
        triang_res = [(triangulate_correspondences(corrs,
                                                   eye3x4(),
                                                   pose_to_view_mat3x4(pose),
                                                   intrinsic_mat,
                                                   _TRIANGULATION_PARAMETERS),
                       pose)
                      for pose in poses]
        (cur_points, cur_points_ids), cur_pose = \
            max(triang_res, key=lambda tp: tp[0][1].size)
        if _DEBUG and cur_points_ids.size:
            print(f'frame {frame}, E: {mask_E.mean():.3f}, H: {mask_H.mean():.3f}, H/E: {mask_H.mean() / mask_E.mean():.3f}, n_corrs: {len(corrs.ids)}')
            print(f'{cur_points_ids.size} 3d points triangulated')
        if cur_points_ids.size >= _N_TRIANGULATED_POINTS_THRESHOLD \
                and hom_ess_ratio <= _HOM_ESS_RATIO_THRESHOLD:
            init_points, init_points_ids = cur_points, cur_points_ids
            init_pose, init_frame = cur_pose, frame
            # break
        elif best_stats < (cur_points_ids.size, hom_ess_ratio):
            best_stats = (cur_points_ids.size, hom_ess_ratio)
            init_points, init_points_ids = cur_points, cur_points_ids
            init_pose, init_frame = cur_pose, frame
    else:
        print('Could not find a good initialization, '
              'using something kinda good instead')
    builder_res.add_points(init_points_ids, init_points)
    for i in range(1, len(view_mats_res)):
        view_mats_res[i] = eye3x4() if i < init_frame \
            else pose_to_view_mat3x4(init_pose)

    return view_mats_res, builder_res


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    view_mats, point_cloud_builder = _track_camera(
        corner_storage,
        intrinsic_mat
    )
    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    create_cli(track_and_calc_colors)()
