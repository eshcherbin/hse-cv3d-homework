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
_INIT_STRICT_PARAMS = (0.5, 700, 1)
_INIT_LOOSE_PARAMS = (0.7, 300, 3)
# _DIST_RETRIANG_THRESHOLD = 1

_DEBUG = False


def _try_init(init_params, corner_storage,
              intrinsic_mat, triangulation_parameters):
    if _DEBUG:
        print(f'trying init params: {init_params}')
    hom_ess_ratio_threshold, n_triang_pts_threshold, ransac_threshold = init_params
    init_points, init_points_ids = None, None
    init_pose, init_frame = None, None
    best_init_params = (-1, -1)  # -median_cos, init_points_ids.size
    for frame in range(1, len(corner_storage)):
        corrs = build_correspondences(corner_storage[0], corner_storage[frame])
        E, mask_E = cv2.findEssentialMat(corrs.points_1, corrs.points_2,
                                         intrinsic_mat, threshold=ransac_threshold,
                                         prob=0.99999999)
        outliers = corrs.ids[mask_E.flatten() == 0]
        # outliers = np.array([], dtype=np.int)
        corrs = build_correspondences(corner_storage[0], corner_storage[frame],
                                      outliers)

        _, mask_H = cv2.findHomography(corrs.points_1, corrs.points_2,
                                       method=cv2.RANSAC)
        hom_ess_ratio = mask_H.mean()

        R1, R2, t = cv2.decomposeEssentialMat(E)
        poses = [Pose(r_mat, t_vec) for r_mat, t_vec in ((R1.T, R1.T @ t), (R1.T, R1.T @ -t),
                                                         (R2.T, R2.T @ t), (R2.T, R2.T @ -t))]
        triang_res = [(triangulate_correspondences(corrs,
                                                   eye3x4(),
                                                   pose_to_view_mat3x4(pose),
                                                   intrinsic_mat,
                                                   triangulation_parameters),
                       pose)
                      for pose in poses]
        (cur_points, cur_points_ids, median_cos), cur_pose = \
            max(triang_res, key=lambda tp: tp[0][1].size)

        # retval, R, t, mask = cv2.recoverPose(E, corrs.points_1, corrs.points_2, intrinsic_mat)
        # cur_pose = Pose(R, t)
        #
        # outliers = snp.merge(outliers, corrs.ids[mask.flatten() == 0])
        # corrs = build_correspondences(corner_storage[0], corner_storage[frame],
        #                               outliers)
        #
        # if corrs.ids.size <= _N_TRIANGULATED_POINTS_THRESHOLD:
        #     continue
        # cur_points, cur_points_ids, median_cos = triangulate_correspondences(corrs,
        #                                                                      eye3x4(),
        #                                                                      pose_to_view_mat3x4(cur_pose),
        #                                                                      intrinsic_mat,
        #                                                                      triangulation_parameters)

        # if cur_pose.t_vec[0] < 0:
        #     print(frame, cur_pose)

        if _DEBUG and cur_points_ids.size:
            print(f'frame {frame}, E: {mask_E.mean():.3f}, H: {mask_H.mean():.3f}, '
                  f'H/E: {hom_ess_ratio:.3f}, n_corrs: {len(corrs.ids)}, '
                  f'median_cos: {median_cos:.3f}')
            print(f'{cur_points_ids.size} 3d points triangulated')

        if cur_points_ids.size >= n_triang_pts_threshold and hom_ess_ratio <= hom_ess_ratio_threshold:
            if init_frame is None or best_init_params < (-median_cos, cur_points_ids.size):
                best_init_params = (-median_cos, cur_points_ids.size)
                init_points, init_points_ids = cur_points, cur_points_ids
                init_pose, init_frame = cur_pose, frame
                if _DEBUG:
                    print('found a better pair!')

    return (init_points, init_points_ids, init_pose, init_frame) \
        if init_frame is not None else None


def _camera_tracking_initialization(corner_storage, intrinsic_mat,
                                    triangulation_parameters):
    for init_params in [_INIT_STRICT_PARAMS, _INIT_LOOSE_PARAMS]:
        result = _try_init(init_params, corner_storage,
                           intrinsic_mat, triangulation_parameters)
        if result is not None:
            return result
    return None


def _track_camera(corner_storage: CornerStorage,
                  intrinsic_mat: np.ndarray,
                  frame_shape) \
        -> Tuple[List[np.ndarray], PointCloudBuilder]:
    triangulation_parameters_enrich = \
        TriangulationParameters(max(5, min(frame_shape[:2]) / 100), 5, 0)
    triangulation_parameters_init = \
        TriangulationParameters(max(5, min(frame_shape[:2]) / 100), None, 0)
    view_mats_res = [eye3x4() for _ in range(len(corner_storage))]
    builder_res = PointCloudBuilder()

    # initialization
    init_result = _camera_tracking_initialization(corner_storage, intrinsic_mat,
                                                  triangulation_parameters_init)
    if init_result is None:
        print('Coudn\'t initialize :(')
        return view_mats_res, builder_res

    init_points, init_points_ids, init_pose, init_frame = init_result
    builder_res.add_points(init_points_ids, init_points)

    if _DEBUG:
        print(f'{builder_res.ids.size} points currently in builder_res')
        print(triangulation_parameters_enrich)

    prev_outliers = set()
    prev_r_vec, prev_t_vec = np.zeros((3, 1)), np.zeros((3, 1))
    for frame in range(1, len(corner_storage)):
        # if frame >= init_frame:
        #     view_mats_res[frame] = pose_to_view_mat3x4(init_pose)
        #     continue
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
            retval, r_vec, t_vec, inliers = \
                cv2.solvePnPRansac(points3d, points2d, intrinsic_mat, None,
                                   rvec=prev_r_vec, tvec=prev_t_vec,
                                   useExtrinsicGuess=True,
                                   reprojectionError=triangulation_parameters_enrich.max_reprojection_error)
            if _DEBUG and not retval:
                print('BOGDAN POMOGI!!!!!!')
            view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
            view_mats_res[frame] = view_mat
            outliers = np.delete(np.arange(points2d_idx.size, dtype=np.int),
                                 inliers.astype(np.int))

            print(f'tracking frame {frame}, n_builder: {builder_res.ids.size}, '
                  f'n_cur_corrs: {points2d_idx.size}, '
                  f'inliers_ratio: {inliers.size / points2d_idx.size}')

            prev_outliers.update(ids2d[points2d_idx[outliers]])
        else:
            print('tracking frame {frame}, PROBLEMO RANSACO BRAGO BRAGO KUKURUZO')
            view_mats_res[frame] = view_mats_res[frame - 1]
            continue
        #
        for second_frame in range(frame):
            corrs = build_correspondences(corner_storage[second_frame],
                                          corner_storage[frame],
                                          builder_res.ids.flatten())
            if corrs.ids.size:
                new_points, new_ids, cos_thres = triangulate_correspondences(
                    corrs,
                    view_mats_res[second_frame],
                    view_mats_res[frame],
                    intrinsic_mat,
                    triangulation_parameters_enrich
                )
                # _, (not_new_idx, old_idx) = snp.intersect(new_ids.flatten(),
                #                                           builder_res.ids.flatten(),
                #                                           indices=True)
                # outliers_idx = old_idx[np.sum((new_points[not_new_idx] -
                #                                builder_res.points[old_idx])**2,
                #                               axis=1) > _DIST_RETRIANG_THRESHOLD]
                # prev_outliers.update(builder_res.ids[outliers_idx].flatten())
                # new_points = np.delete(new_points, not_new_idx, axis=0)
                # new_ids = np.delete(new_ids, not_new_idx, axis=0)
                if _DEBUG and new_ids.size:
                    print(f'adding {new_ids.size} points between frames '
                          f'{second_frame} and {frame}, cos_thres: {cos_thres}')
                builder_res.add_points(new_ids, new_points)

    # if _DEBUG:
    #     builder_res = PointCloudBuilder(init_points_ids, init_points)
    #     print(f'{builder_res.ids.size} points in builder_res!!!')

    if _DEBUG:
        for frame in range(len(corner_storage)):
            _, (point_cloud_idx, corners_idx) = snp.intersect(
                builder_res.ids.flatten(),
                corner_storage[frame].ids.flatten(),
                indices=True
            )
            print(f'on frame {frame} there will be {point_cloud_idx.size} projected points')

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
        intrinsic_mat,
        rgb_sequence[0].shape
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
