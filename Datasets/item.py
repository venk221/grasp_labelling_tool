# The common item types returned by both datasets

import numpy as np


def clamp(val, top, btm):
    return min(max(val, btm), top)


class Rectangle:
    def __init__(self, pts, score=0.0):
        self.__pts = pts
        self.__score = clamp(float(score), 1.0, 0.0)

    def score(self):
        return self.__score

    def set_score(self, score):
        self.__score = clamp(float(score), 1.0, 0.0)

    def pts(self):
        return self.__pts
    
    def points(self):
        return self.__pts

    def set_pts(self, pts):
        self.__pts = pts

    def copy(self):
        return Rectangle(self.__pts.copy(), self.__score)

    def center(self):
        return np.mean(self.__pts, axis=0).astype(np.int64)

    def angle(self):
        dx = self.__pts[1, 1] - self.__pts[0, 1]
        dy = self.__pts[1, 0] - self.__pts[0, 0]
        return (np.arctan2(-dy, dx) + np.pi / 2) % np.pi - np.pi / 2
    
    def width(self):
        dy = self.__pts[2, 1] - self.__pts[1, 1]
        dx = self.__pts[2, 0] - self.__pts[1, 0]
        return np.sqrt(dx ** 2 + dy ** 2)
    
    def length(self):
        dx = self.__pts[1, 1] - self.__pts[0, 1]
        dy = self.__pts[1, 0] - self.__pts[0, 0]
        return np.sqrt(dx ** 2 + dy ** 2)


# Creates a Rectangle from rectangle parameters
def rect_shape_to_pts(center, angle, length, width, score=0.0):
    # Move points by this vector
    delta = np.array([np.cos(angle), np.sin(angle)])
    dA = delta * width / 2
    dB = np.flip(delta) * length / 2
    grasp_pts = np.array([[center[0] - dA[0] + dB[0], center[1] - dA[1] - dB[1]],
                          [center[0] - dA[0] - dB[0], center[1] - dA[1] + dB[1]],
                          [center[0] + dA[0] - dB[0], center[1] + dA[1] + dB[1]],
                          [center[0] + dA[0] + dB[0], center[1] + dA[1] - dB[1]]])
    return Rectangle(grasp_pts, score)


class Item:
    def __init__(self, rgb_img, dep_img, grasp_arr):
        self.rgb_img = rgb_img
        self.dep_img = dep_img
        self.grasp_arr = grasp_arr


