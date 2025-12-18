import numpy as np
import cv2
import inspect


class VisionUtils:

    @classmethod
    def log(cls, message: str):
        print(f'[{cls.__name__}@{inspect.currentframe().f_code.co_name}] {str}')

    @staticmethod
    def find_contours(img, _hsv_lower: tuple[float, float, float], _hsv_upper: tuple[float, float, float]):
        hsv_lower = np.array(_hsv_lower)
        hsv_upper = np.array(_hsv_upper)

        mask = cv2.inRange(img, hsv_lower, hsv_upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            __class__.log("No red object found.")
            return None
        
        return contours
    
    @staticmethod
    def find_largest_contour(img, _hsv_lower: tuple[float, float, float], _hsv_upper: tuple[float, float, float]):
        contours = VisionUtils.find_contours(img, _hsv_lower, _hsv_upper)
        largest = max(contours, key=cv2.contourArea)
        return largest
    
    @staticmethod
    def find_line_contour(img, _hsv_lower: tuple[float, float, float], _hsv_upper: tuple[float, float, float]):
        
        contours = VisionUtils.find_contours(img, _hsv_lower, _hsv_upper)
        needle_candidates = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)

            # 너무 작은 컨투어는 걸러냄
            if area < 1:
                continue

            # aspect ratio 검증: 길쭉해야 needle 가능성 있음
            aspect = max(w, h) / (min(w, h) + 1e-5)

            if aspect > 3.0:   # 길쭉함 필터
                needle_candidates.append(c)

        if len(needle_candidates) == 0:
            __class__.log("No thin/linear contour found.")
            return None

        # 그 중 가장 큰 길이(또는 면적)를 가진 contour 선택
        largest = max(needle_candidates, key=cv2.contourArea)
        return largest

    @staticmethod    
    def get_terminal_points(countour) -> tuple[tuple[float, float], tuple[float, float]]:
        pts = countour.reshape(-1, 2)
        max_dist = 0
        end1, end2 = None, None
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = np.linalg.norm(pts[i] - pts[j])
                if d > max_dist:
                    max_dist = d
                    end1, end2 = pts[i], pts[j]
        return end1, end2
    