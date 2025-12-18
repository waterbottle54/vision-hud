
import cv2
import pytesseract
from PIL import Image
import numpy as np
import math
from src.models.SpeedIndicator import SpeedIndicator
from src.utils.Resources import Resources

img_src = cv2.imread(Resources.get_image_path('dashboard.png'), cv2.IMREAD_REDUCED_COLOR_2)
img_result = img_src.copy()
hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)

speed_indicator = SpeedIndicator(hsv)
speed_indicator.draw(img_result)

# calc speed
max_speed = 140.0
speed = (speed_indicator.needle.line.angle / math.pi * max_speed)

# draw speed
cv2.putText(img_result, f'Speed: {speed:.1f} km/h', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#combine
cv2.imshow("Dashboard", np.hstack((img_src, img_result)))
cv2.waitKey(0)
cv2.destroyAllWindows()
