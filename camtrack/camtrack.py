#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Tuple

import numpy as np
import sortednp as snp

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import *

import cv2


# TODO: calibrate everything
_HOM_ESS_RATIO_THRESHOLD = 0.5
_TRIANGULATION_PARAMETERS = TriangulationParameters(5, 5, 0)
_N_TRIANGULATED_POINTS_THRESHOLD = 50
_STRICT_TRIANGULATION_PARAMETERS = TriangulationParameters(5, 5, 0)

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
    found_good = False
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
            if init_points_ids is None or \
                    init_points_ids.size < cur_points_ids.size:
                init_points, init_points_ids = cur_points, cur_points_ids
                init_pose, init_frame = cur_pose, frame
                print([tp[0][1].size for tp in triang_res], init_pose)
                print([tp[1] for tp in triang_res])
                found_good = True
        elif best_stats < (cur_points_ids.size, hom_ess_ratio):
            best_stats = (cur_points_ids.size, hom_ess_ratio)
            init_points, init_points_ids = cur_points, cur_points_ids
            init_pose, init_frame = cur_pose, frame
    if not found_good:
        print('Could not find a good initialization, '
              'using something kinda good instead')
    builder_res.add_points(init_points_ids, init_points)
    print(init_pose)

    prev_outliers = set()
    for frame in range(1, len(corner_storage)):
        if frame == init_frame:
            view_mats_res[frame] = pose_to_view_mat3x4(init_pose)
            continue
        # calculate pose for frame
        # cur_corners_ids = np.delete(corner_storage[frame].ids.flatten(),
        #                             list(prev_outliers),
        #                             axis=0)
        # print(cur_corners_ids.size,
        #       corner_storage[frame].ids.size,
        #       prev_outliers,
        #       cur_corners_ids
        #       )
        ids2d = corner_storage[frame].ids.flatten()
        _, (points2d_idx,
            points3d_idx) = snp.intersect(ids2d,
                                          builder_res.ids.flatten(),
                                          indices=True)
        to_remove_from_ids = np.searchsorted(ids2d, list(prev_outliers))
        to_remove_from_idx = np.searchsorted(points2d_idx, to_remove_from_ids)
        points2d_idx = np.delete(points2d_idx, to_remove_from_idx, 0)
        points3d_idx = np.delete(points3d_idx, to_remove_from_idx, 0)
        points2d = corner_storage[frame].points[points2d_idx]
        points3d = builder_res.points[points3d_idx]
        if points2d_idx.size >= 6:
            # print(points2d_idx, points3d_idx)
            # if frame == 2:
            #     break
            retval, r_vec, t_vec, inliers = cv2.solvePnPRansac(points3d, points2d, intrinsic_mat, None,
                                                               reprojectionError=_TRIANGULATION_PARAMETERS.max_reprojection_error)
            if not retval:
                print('BOGDAN POMOGI!!!!!!')
            view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
            view_mats_res[frame] = view_mat
            if _DEBUG:
                print(f'tracking frame {frame}, n_builder: {builder_res.ids.size}, n_cur_corrs: {points2d_idx.size}, inliers_ratio: {inliers.size / points2d_idx.size}')

            outliers = np.delete(np.arange(points2d_idx.size, dtype=np.int),
                                 inliers.astype(np.int))
            # print('AAA', prev_outliers)
            prev_outliers.update(outliers)
            # builder_res.remove_ids(outliers)
        else:
            print('PROBLEMO RANSACO BRAGO BRAGO KUKURUZO')
            view_mats_res[frame] = eye3x4()

        for second_frame in range(frame):
            corrs = build_correspondences(corner_storage[second_frame],
                                          corner_storage[frame],
                                          builder_res.ids)
            if corrs.ids.size:
                new_points, new_ids = triangulate_correspondences(
                    corrs,
                    view_mats_res[second_frame],
                    view_mats_res[frame],
                    intrinsic_mat,
                    _STRICT_TRIANGULATION_PARAMETERS
                )
                builder_res.add_points(new_ids, new_points)

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
