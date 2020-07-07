#!/usr/bin/env python
import PySimpleGUI as sg
import cv2
import numpy as np
import pickle
import os
from PIL import Image

# Charger les classifiers (cascades)
face_classifier = cv2.CascadeClassifier('HAAR/haarcascade_frontalface_default.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()



recognizer.read('trainner.yml')

labels = {"nom_personne": 1}

with open("labels.pickle", "rb") as f: #wb reading bytes, f : file
   og_labels= pickle.load(f)
   labels = {v:k for k,v in og_labels.items()}

def train():
    print=sg.Print
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
                    # print(str(x_train) + " classe: " +str(id_) + "\n")

                    print(str(x_train) + " classe: " +str(id_) + "\n")
                        
                    y_labels.append(id_)


    # print(y_labels, x_train)
    # print(np.array(y_labels))

    with open("labels.pickle", "wb") as f: #wb wrting bytes, f : file
        pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))

    recognizer.save("trainner.yml")
    print('Succesfully trainned, you can quit now')

def recognition_webcam(gray, frame):
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2) #(x,y) up left, (x+w, y+h) lower right
        roi_gray = gray[y:y+h, x:x+w] #region of intrest 
        roi_color = frame[y:y+h, x:x+w]


        id_, resultat = recognizer.predict(roi_gray)
        # print(roi_gray)
        confidence = int( 100 * (1 - resultat/400) )
        print(labels[id_],confidence)
        
        if confidence >= 85:     
            # print(id_)
            # print(labels[id_])
            cv2.putText(frame, labels[id_], (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2, cv2.LINE_AA)

        else:
            cv2.putText(frame, "Inconnu", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2, cv2.LINE_AA)

        
    return frame

def recognition_image(gray, frame):
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2) #(x,y) up left, (x+w, y+h) lower right
        roi_gray = gray[y:y+h, x:x+w] #region of intrest 
        roi_color = frame[y:y+h, x:x+w]


        id_, resultat = recognizer.predict(roi_gray)
        # print(roi_gray)
        confidence = int( 100 * (1 - resultat/400) )
        # print(labels[id_],confidence)
        
        if confidence >= 70:     
            # print(id_)
            # print(labels[id_])
            cv2.putText(frame, labels[id_], (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2, cv2.LINE_AA)
            #source, text, position, fontstyle, size, color, weight, line type
        else:
            cv2.putText(frame, "Inconnu", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2, cv2.LINE_AA)

        
    return frame

def face_extractor(img):
    # Function detects faces and ret urns the cropped face
    # If no face detected, it returns the input image
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ( ):
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w] #Segmentation 

    return cropped_face

def image_uploader(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_cropped = cv2.resize(gray, (200,200))
    canvas = recognition_image(gray_cropped, image)
    # cv2.imshow('Reconnaissance faciale', canvas)
    return canvas

def main():

    sg.theme('Black')

    # define the window layout
    layout = [
              [sg.Image(filename='', key='image')],
              [sg.Button('Webcam facial recognition', size=(20, 2), font='Helvetica 14'),
              sg.Button('Pictrue facial recognition', size=(20, 2), font='Helvetica 14'),
               sg.Button('Add a new person', size=(20, 2), font='Any 14'),
               sg.Button('Exit', size=(10, 2), font='Helvetica 14'), ]]

    layout_2 = [[sg.Text('Give the person\'s name')], [sg.Input(key='-IN-')], [sg.Button('Add', size=(10, 1), font='Any 14')]]
    
    layout_3 = [[sg.Text('PASTE IMAGE PATH OR BROWSE FOR AN IMAGE', size=(53, 1))], [sg.Input(key='-IN-'), sg.FileBrowse('Browse')], [sg.Button('Test', size=(10, 1), font='Any 14')]]

    layout_4 = [[sg.Text('ERROR ! GIVE A VALID PATH TO AN IMAGE')], [sg.Button('OK', size=(10, 1), font='Any 14')]]

    
    window = sg.Window('Facial recognition',
                       layout, location=(280, 160))
    
    window_2 = sg.Window('Add a new person', layout_2)

    window_3 = sg.Window('Pictrue facial recognition', layout_3)
    window_4 = sg.Window('', layout_4)
    

    
    video_capture = cv2.VideoCapture(0)
    recording = False
    

    while True:
        event, value = window.read(timeout=20)
        
        if event == 'Exit' or event is None:
            
            return

        elif event == 'Webcam facial recognition':
            recording = True

        elif event == 'Pictrue facial recognition': 
                event, value = window_3.read() 

                path = value['-IN-'] 

                if event == 'Test' and path != '':
                    try:
                        window_3.close()
                        canvas = image_uploader(path)
                        imgbytes = cv2.imencode('.png', canvas)[1].tobytes()
                        window['image'].update(data=imgbytes)
                    except:
                        event_, value_ = window_4.read()
                        if event_ == 'OK':
                            window_4.close()
                        

        elif event == 'Add a new person':
            event, value = window_2.read()
            
            user = value['-IN-']

            if event == 'Add' and user != '':
                dir = 'faces/'+str(value['-IN-'])
                if not os.path.exists(dir):
                    os.mkdir(dir)
                window_2.close()
                count = 0

                while True:
    
                    ret, frame = video_capture.read()
                    if face_extractor(frame) is not None:
                        count += 1
                        face = cv2.resize(face_extractor(frame), (200, 200)) #normalisation 
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                        # Save file in specified directory with unique name
                        file_name_path = 'faces/'+ user +'/'+ str(count) + '.jpg'
                        cv2.imwrite(file_name_path, face)

                        # Put count on images and display live count
                        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                        cv2.imshow('Face Cropper', face)
                        
                    else:
                        print("Face not found")
                        pass

                    if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key
                        break
                        
                video_capture.release()
                cv2.destroyAllWindows()      
                print("Collecting Samples Complete")
                
                train()
                # exec(open("train.py").read())

        
        
        if recording:
            _, frame = video_capture.read() #gives 2 returns, not intressted in the first one
                                    #capturs the last frame of the web cam
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            canvas = recognition_webcam(gray, frame)
            imgbytes = cv2.imencode('.png', canvas)[1].tobytes()
            window['image'].update(data=imgbytes)
            
            
            key = ord('0'); 
            if cv2.waitKey(1) & key == ord(' '):
                break


main()
