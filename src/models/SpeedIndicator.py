import cv2
import pytesseract
from PIL import Image
import numpy as np
import math
from src.utils.VisionUtils import VisionUtils

class Line:
    def __init__(self, begin: tuple[float, float], end: tuple[float, float]):
        self.begin = begin
        self.end = end
        self.dx = end[0] - begin[0]
        self.dy = end[1] - begin[1]
        self.angle = math.atan2(self.dy, self.dx)

    def draw(self, img, radius_ends: int, color: tuple[int, int, int]):
        cv2.circle(img, tuple(self.begin), radius_ends, color, -1)
        cv2.circle(img, tuple(self.end), radius_ends, color, -1)
        cv2.line(img, tuple(self.begin), tuple(self.end), color, 2)

class Needle:
    def __init__(self, img):
        self.img = img
        self.contour = VisionUtils.find_largest_contour(img, (170, 70, 50), (180, 255, 255))      # red one
        self.line = Line(*VisionUtils.get_terminal_points(self.contour))

    def draw(self, img):
        self.line.draw(img, radius_ends=8, color=(0,255,0))

class Ticks:
    def __init__(self, img):
        self.img = img
        self.contours = VisionUtils.find_contours(img, (0, 0, 200), (180, 40, 255))              # white ones
        self.lines = map(lambda contour: Line(*VisionUtils.get_terminal_points(contour)), self.contours)

    def draw(self, img):
        for line in self.lines:
            line.draw(img, radius_ends=4, color=(128, 128, 128))

class SpeedIndicator:

    def __init__(self, img):
        self.needle = Needle(img)
        self.ticks = Ticks(img)

    def draw(self, img): 
        # draw needle
        self.needle.draw(img)
        self.ticks.draw(img)
