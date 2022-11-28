#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Description:   1、利用棋盘图进行相机外参标定
                2、利用深度图进行相机偏航角和俯仰角计算
@Date     :2022/11/28 15:14:50
@Author      :xiadachen
@version      :1.0
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.spatial.transform import Rotation

PI = 3.1415926
mtx = [[360.970795, 0, 320.923035], [0, 360.970795, 240.677246], [0, 0, 1]]
dist = [0.320683, -0.454324, 0.000425, 0.000172, 0.192664]

mtx = np.array(mtx)
dist = np.array(dist)


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    corner = (int(corner[0]), int(corner[1]))
    pt1 = (int(imgpts[0][0][0]), int(imgpts[0][0][1]))
    pt2 = (int(imgpts[1][0][0]), int(imgpts[1][0][1]))
    pt3 = (int(imgpts[2][0][0]), int(imgpts[2][0][1]))
    img = cv2.line(img, corner, pt1, (255, 0, 0), 5)
    img = cv2.line(img, corner, pt2, (0, 255, 0), 5)
    img = cv2.line(img, corner, pt3, (0, 0, 255), 5)
    return img


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert isRotationMatrix(R)

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


# 标定图像
def calibration_photo(photo_path):
    # 设置要标定的角点个数
    x_nums = 8  # x方向上的角点个数
    y_nums = 5
    # 设置(生成)标定图在世界坐标中的坐标
    world_point = np.zeros(
        (x_nums * y_nums, 3), np.float32
    )  # 生成x_nums*y_nums个坐标，每个坐标包含x,y,z三个元素
    world_point[:, :2] = np.mgrid[:x_nums, :y_nums].T.reshape(
        -1, 2
    )  # mgrid[]生成包含两个二维矩阵的矩阵，每个矩阵都有x_nums列,y_nums行
    world_point = world_point * 0.27
    # .T矩阵的转置
    # reshape()重新规划矩阵，但不改变矩阵元素
    # 设置世界坐标的坐标
    axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)
    # 设置角点查找限制
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    image = cv2.imread(photo_path)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 查找角点
    ok, corners = cv2.findChessboardCorners(
        gray,
        (x_nums, y_nums),
    )

    if ok:
        # 获取更精确的角点位置
        exact_corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        # for corner in exact_corners:
        #     pt = (int(corner[0][0]), int(corner[0][1]))
        #     gray = cv2.circle(gray, pt, 1, (0, 0, 255))
        # cv2.imshow("img", gray)
        # cv2.waitKey(0)
        # 获取外参
        _, rvec, tvec, inliers = cv2.solvePnPRansac(
            world_point, exact_corners, mtx, dist
        )

        r = Rotation.from_rotvec(rvec.T)
        print(r.as_euler("xyz", degrees=True))
        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
        print(exact_corners[0][0])
        # print(tvec)
        # 可视化角点
        # img = draw(image, corners, imgpts)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)


def depth2mi(depthValue):

    return depthValue * 0.001


def depth2xyz(u, v, depthValue):
    # 深度相机内参，如果经过D2C，应该使用color相机内参
    fx = 513.494385
    fy = 513.218140
    cx = 317.008484
    cy = 246.202484

    depth = depth2mi(depthValue)

    z = float(depth)
    x = float((u - cx) * z) / fx
    y = float((v - cy) * z) / fy

    result = [x, y, z]
    return result


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "(%d, %d)" % (self.x, self.y)

    def line2(self, another):
        if self.x == another.x:
            step = 1 if self.y < another.y else -1
            y = self.y
            while y != another.y:
                yield Point(self.x, y)
                y += step
        elif self.y == another.y:
            step = 1 if self.x < another.x else -1
            x = self.x
            while x != another.x:
                yield Point(x, self.y)
                x += step
        else:
            d_x = self.x - another.x
            d_y = self.y - another.y
            s_x = 1 if d_x < 0 else -1
            s_y = 1 if d_y < 0 else -1

            if d_y:
                delta = 1.0 * d_x / d_y
                for i in range(0, d_x):
                    yield Point(self.x + i * s_x, self.y + i * s_x / delta)
            elif d_x:
                delta = 1.0 * d_y / d_x
                for i in range(0, d_y):
                    yield Point(self.y + i * s_y / delta, self.y + i * s_y)


def calibration_photo_from_depth(depth_path, pointA, pointB):

    depth = cv2.imread(depth_path, -1)
    points = []
    for point in pointA.line2(pointB):
        x, y, z = depth2xyz(point.x, point.y, depth[point.y, point.x])

        points.append([x, y, z])

    points = np.array(points)
    output = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

    return output


# 标定图像保存路径
photo_path = r"D:\Users\OrbbecViewer_v1.4.2_202211141809_ENGINEERING_win_x64_release\output\depth2color\16182680\color\000001_color_16182685.jpg"

depth_path = r"D:\Users\ca\3.0s-30\20221124164049522_000000_Depth_640x480_F094000_400mm_4NS_0.4_SHUFFLE_6688605000.png"

if __name__ == "__main__":

    calibration_photo(photo_path)

    pointA = Point(200, 20)
    pointB = Point(200, 100)
    m, n, p, x0, y0, z0 = calibration_photo_from_depth(depth_path, pointA, pointB)

    print(math.atan(p / n) / PI * 180, math.atan(p / m) / PI * 180)
