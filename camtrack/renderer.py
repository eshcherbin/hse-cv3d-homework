#! /usr/bin/env python3

__all__ = [
    'CameraTrackRenderer'
]

from typing import List, Tuple

import numpy as np
from OpenGL import GL
from OpenGL import GLUT
from OpenGL.GL import shaders
from OpenGL.arrays import vbo

import data3d


_CL2GL = np.diag([1, -1, -1, 1]).astype(np.float32)


def _build_color_program():
    color_vertex_shader = shaders.compileShader(
        """
        #version 130
        uniform mat4 mvp;

        in vec3 position;
        in vec3 color_in;
        out vec3 color;

        void main() {
            vec4 camera_space_position = mvp * vec4(position, 1.0);
            gl_Position = camera_space_position;
            color = color_in;
        }""",
        GL.GL_VERTEX_SHADER
    )
    points_color_fragment_shader = shaders.compileShader(
        """
        #version 130
        
        in vec3 color;
        out vec3 out_color;

        void main() {
            out_color = color;
        }""",
        GL.GL_FRAGMENT_SHADER
    )
    return shaders.compileProgram(
        color_vertex_shader, points_color_fragment_shader
    )


class CameraTrackRenderer:

    def __init__(self,
                 cam_model_files: Tuple[str, str],
                 tracked_cam_parameters: data3d.CameraParameters,
                 tracked_cam_track: List[data3d.Pose],
                 point_cloud: data3d.PointCloud):
        """
        Initialize CameraTrackRenderer. Load camera model, create buffer objects, load textures,
        compile shaders, e.t.c.

        :param cam_model_files: path to camera model obj file and texture. The model consists of
        triangles with per-point uv and normal attributes
        :param tracked_cam_parameters: tracked camera field of view and aspect ratio. To be used
        for building tracked camera frustrum
        :param point_cloud: colored point cloud
        """

        self._cam_fov_y = tracked_cam_parameters.fov_y
        self._cam_ratio = tracked_cam_parameters.aspect_ratio

        self._points_pos_buffer_object = vbo.VBO(np.array(point_cloud.points, dtype=np.float32))
        self._points_color_buffer_object = vbo.VBO(np.array(point_cloud.colors, dtype=np.float32))

        self._color_program = _build_color_program()

        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | GLUT.GLUT_DEPTH)
        GL.glEnable(GL.GL_DEPTH_TEST)

    @staticmethod
    def _build_view(tr, rot):
        return np.linalg.inv(np.block([[rot, tr.reshape(-1, 1)], [0, 0, 0, 1]])).astype(np.float32)

    @staticmethod
    def _build_proj(fov_y, ratio, near, far):
        fy = 1 / np.tan(fov_y / 2)
        fx = fy / ratio
        a = -(far + near) / (far - near)
        b = -2 * far * near / (far - near)
        return np.block([[np.diag([fx, fy]), np.zeros((2, 2))],
                         [np.zeros((2, 2)), np.array([[a, b],
                                                      [-1, 0]])]]).astype(np.float32)

    def display(self, camera_tr_vec, camera_rot_mat, camera_fov_y, tracked_cam_track_pos_float):
        """
        Draw everything with specified render camera position, projection parameters and 
        tracked camera position

        :param camera_tr_vec: vec3 position of render camera in global space
        :param camera_rot_mat: mat3 rotation matrix of render camera in global space
        :param camera_fov_y: render camera field of view. To be used for building a projection
        matrix. Use glutGet to calculate current aspect ratio
        :param tracked_cam_track_pos_float: a frame in which tracked camera
        model and frustrum should be drawn (see tracked_cam_track_pos for basic task)
        :return: returns nothing
        """

        # a frame in which a tracked camera model and frustrum should be drawn
        # without interpolation
        tracked_cam_track_pos = int(tracked_cam_track_pos_float)

        # mvp = np.eye(4)

        cur_ratio = GLUT.glutGet(GLUT.GLUT_WINDOW_WIDTH) / GLUT.glutGet(GLUT.GLUT_WINDOW_HEIGHT)
        mvp = self._build_proj(camera_fov_y, cur_ratio, 0.001, 50) \
            @ self._build_view(camera_tr_vec, camera_rot_mat) \
            @ _CL2GL
        # print(camera_tr_vec)
        # print(camera_rot_mat)
        # print(self._build_view(camera_tr_vec, camera_rot_mat))
        # print(self.normalize(mvp @ np.array([0, 0, 0, 1])))
        # view_ = self._build_view(camera_tr_vec, camera_rot_mat) @ self._CL2GL @ np.array([1, 1, 1, 1])
        # print(view_)
        # print(np.linalg.norm(view_[:3]))

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        self._render_cloud(mvp)

        GLUT.glutSwapBuffers()

    def _render_cloud(self, mvp):
        shaders.glUseProgram(self._color_program)

        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self._color_program, 'mvp'),
            1, True, mvp)

        buffers = [
            ('position', self._points_pos_buffer_object),
            ('color_in', self._points_color_buffer_object)
        ]

        for attrib, buffer in buffers:
            buffer.bind()
            loc = GL.glGetAttribLocation(self._color_program, attrib)
            GL.glEnableVertexAttribArray(loc)
            GL.glVertexAttribPointer(loc, 3, GL.GL_FLOAT, False, 0, buffer)

        GL.glDrawArrays(GL.GL_POINTS, 0, self._points_pos_buffer_object.size // 3)

        for attrib, buffer in buffers:
            loc = GL.glGetAttribLocation(self._color_program, attrib)
            GL.glDisableVertexAttribArray(loc)
            buffer.unbind()

        shaders.glUseProgram(0)
