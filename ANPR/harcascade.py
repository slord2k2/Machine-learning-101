import cv2
import numpy as np
import pytesseract
from PIL import Image
# remove warning message
faceCascade = cv2.CascadeClassifier('haarcascade\haarcascade_russian_plate_number.xml')

img = cv2.imread('Plate_examples/india_car_plate.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors = 5, minSize=(25,25))

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cropped_image = img[y: y+h, x:x+w]
    cv2.imshow('Cropped Plate', cropped_image)
    img[y: y+h, x:x+w] = cropped_image
# text=pytesseract.image_to_string(cropped_image,lang='eng')
# print(f"Plate Number is: {text}")
cv2.imshow('Original Image with Plate',img)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()


#  import cv2
# import numpy as np



# faceCascade = cv2.CascadeClassifier('haarcascade\haarcascade_russian_plate_number.xml')

# img = cv2.imread('Plate_examples/russia_car_plate.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors = 5, minSize=(25,25))

# # for (x,y,w,h) in faces:
# #     cv2.rectangle(gray,(x,y),(x+w,y+h),(0,255,0),2)
# #     plate = gray[y: y+h, x:x+w]
# #     plate = cv2.blur(plate,ksize=(20,20))
# #     # put the blurred plate into the original image
# #     gray[y: y+h, x:x+w] = plate
# for (x,y,w,h) in faces:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     plate = img[y: y+h, x:x+w]
#     # plate = cv2.blur(plate,ksize=(20,20))
#     # put the blurred plate into the original image
#     img[y: y+h, x:x+w] = plate

# cv2.imshow('plates',img)
# if cv2.waitKey(0) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()