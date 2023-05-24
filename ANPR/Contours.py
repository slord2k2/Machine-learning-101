import cv2
import numpy as np
# import easyocr
import pytesseract
from PIL import Image
# remove warning message
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# required library
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# from local_utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json
import glob

# Load the image and convert to grayscale
img = cv2.imread('Plate_examples/germany_car_plate.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to the grayscale image
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Apply Canny edge detection to the blurred image
canny = cv2.Canny(blur, 100, 200)

# Find contours in the Canny image
contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours by area in descending order
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# Initialize the number plate contour
screenCnt = None

# Loop over the contours
for contour in contours:
    # Approximate the contour to a polygon
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

    # If the polygon has four vertices, it is likely the number plate
    if len(approx) == 4:
        screenCnt = approx
        break

# If we found the number plate contour, draw it and extract the plate region
if screenCnt is not None:
    cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    # Get the coordinates of the plate region
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    # Extract the plate region from the original image
    plate_img = gray[x1:x2+1, y1:y2+1]

    # Display the number plate region
    img=cv2.resize(img,(500,300))
    cropped_image=cv2.resize(plate_img,(400,200))
    # reader=easyocr.Reader['en']
    # result=reader.readtext(cropped_image)
    
    # ########  Image preprocessing ########
    
    
    
    plate_image = cv2.convertScaleAbs(plate_img)
# plt.imshow(plate_image)
#     # convert to grayscale and blur the image
    # gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(plate_image,(7,7),0)
    
#     # Applied inversed thresh_binary 
    binary = cv2.threshold(blur, 180, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    plate_image = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)  
    
    
    
    
    # OCR
    text=pytesseract.image_to_string(cropped_image,lang='eng')
    print(f"Plate Number is: {text}")
    cv2.imshow('car', img)
    # cv2.imwrite('saved_image.jpg', cropped_image)
    cv2.imshow('Cropped', plate_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('Number plate not found!')
