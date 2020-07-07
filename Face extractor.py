import cv2 
import os

face_classifier = cv2.CascadeClassifier('HAAR/haarcascade_frontalface_default.xml')

def face_extractor(img):
    # Function detects faces and ret urns the cropped face
    # If no face detected, it returns the input image
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ( ):
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face
