import cv2
import pickle
import matplotlib as plt
import numpy as np

# Charger les classifiers (cascades)
face_classifier = cv2.CascadeClassifier('HAAR/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('HAAR/haarcascade_eye.xml')

accuracy = []

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('trainner.yml')

labels = {"nom_personne": 1}

with open("labels.pickle", "rb") as f: #wb reading bytes, f : file
   og_labels= pickle.load(f)
   labels = {v:k for k,v in og_labels.items()}

# Defining a function that will do the detections
def detect(gray, frame):
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2) #(x,y) up left, (x+w, y+h) lower right
        roi_gray = gray[y:y+h, x:x+w] #region of intrest 
        roi_color = frame[y:y+h, x:x+w]

        # eyes = eye_classifier.detectMultiScale(roi_gray, 1.7, 6)
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        id_, resultat = recognizer.predict(roi_gray)
        # print(resultat)
        confidence = int( 100 * (1 - resultat/500) )
        accuracy.append(confidence)
        print(labels[id_],confidence)
        
        # if confidence >= 70:     

        #     cv2.putText(frame, labels[id_], (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # else:
        #     cv2.putText(frame, "Inconnu", (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2, cv2.LINE_AA)

        
    return frame
# data-set/test/Elon/images (100).jpg
image_path = 'data-set/train/Brad/2Q__ (5).jpg' 
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_cropped = cv2.resize(gray, (200,200))
canvas = detect(gray_cropped, image)

cv2.imshow('Reconnaissance faciale', canvas)

# print('max:' + str(np.max(accuracy)))

cv2.waitKey(0)
cv2.destroyAllWindows()