import cv2
import pickle
import numpy as np


# Charger les classifiers (cascades)
face_classifier = cv2.CascadeClassifier('HAAR/haarcascade_frontalface_default.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()

accurancy = []

recognizer.read('trainner.yml')

labels = {"nom_personne": 1}

with open("labels.pickle", "rb") as f: #wb reading bytes, f : file
   og_labels= pickle.load(f)
   labels = {v:k for k,v in og_labels.items()}

# Defining a function that will do the detection
def detect(gray, frame):
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2) #(x,y) up left, (x+w, y+h) lower right
        roi_gray = gray[y:y+h, x:x+w] #region of intrest 
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_classifier.detectMultiScale(roi_gray, 1.7, 6)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        id_, resultat = recognizer.predict(roi_gray)
        # print(roi_gray)
        confidence = int( 100 * (1 - resultat/500) )
        accurancy.append(confidence)
        print(labels[id_],confidence)
        
        if confidence >= 80:     
            # print(id_)
            # print(labels[id_])
            cv2.putText(frame, labels[id_], (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2, cv2.LINE_AA)

        else:
            cv2.putText(frame, "Inconnu", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2, cv2.LINE_AA)

        
    return frame

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0) #0 for computer cam, 1 for external cam

while True:
    _, frame = video_capture.read() #gives 2 returns, not intressted in the first one
                                    #capturs the last frame of the web cam
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Face detector', canvas)
    
    key = ord('0'); 
    if cv2.waitKey(1) & key == ord(' '):
        break



video_capture.release()
cv2.destroyAllWindows()
