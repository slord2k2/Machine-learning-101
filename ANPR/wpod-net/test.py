import cv2
import numpy as np
# import easyocr
import pytesseract
from PIL import Image
img = cv2.imread('saved_image.jpg')
text=pytesseract.image_to_string(img,lang='eng')
print(text)
