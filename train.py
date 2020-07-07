import os #permet d'interagir avec le systeme d'exploitation
import cv2
import numpy as np #permet d'avoir la matrice des images
from PIL import Image
import pickle   #sauvegarder les ids

print('--------------Trainning----------------------')

base_dir = os.path.dirname(os.path.abspath(__file__)) # recuperer le path du dossier racine

Image_dir = os.path.join(base_dir, "faces") # recuperer le path du dossier faces
# Image_dir = '/home/chemseddean/Desktop/L3/S6/PFE/code/Face_Recognition/faces'

face_classifier = cv2.CascadeClassifier('HAAR/haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {}  #dictionnaire
y_labels = []
x_train = []

for root, dirs, files in os.walk(Image_dir):    #parcourir tout le dossier faces 
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):    #finds pictures in the directory
            path = os.path.join(root, file) 
            label = os.path.basename(root) #retourne le nom du fichier 
            
            # print(label, path)

            if not label in label_ids:
                label_ids[label] = current_id
                current_id+=1
                

            id_ = label_ids[label]
            

            pil_image = Image.open(path).convert("L") #fonction de la bib Pillow Pil, convertir les images en grayscale
            image_array = np.array(pil_image, "uint8")  #transformer les images en vecteur numpy avec codification uint8
            # uint8 data type contains all whole numbers from 0 to 255. As with all unsigned numbers

            faces = face_classifier.detectMultiScale(image_array, 1.3, 5)
            for (x, y, w, h) in faces:  #(x,y) up left, (x+w, y+h) lower right
                roi = image_array[y:y+h, x:x+w] #region of intrest 
                
                x_train.append(roi)
                print(str(x_train) + " classe: " +str(id_) + "\n")
                y_labels.append(id_)


# print(y_labels, x_train)
# print(np.array(y_labels))

with open("labels.pickle", "wb") as f: #wb wrting bytes, f : file
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))

recognizer.save("trainner.yml")

print('Successfuly trainned')