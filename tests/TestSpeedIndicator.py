
import cv2
import pytesseract
from PIL import Image
import numpy as np
import math
from src.models.SpeedIndicator import SpeedIndicator
from src.utils.Resources import Resources

img_src = cv2.imread(Resources.get_image_path('dashboard.png'))
img_result = img_src.copy()
hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)

speed_indicator = SpeedIndicator(hsv)
speed_indicator.draw(img_result)

#combine
cv2.imshow("Dashboard", np.hstack((img_src, img_result)))
cv2.waitKey(0)
cv2.destroyAllWindows()
