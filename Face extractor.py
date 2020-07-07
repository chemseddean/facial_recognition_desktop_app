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


# print('-----------------Face extracting-----------------')
# #Brad

# folder = 'data-set/train/Brad'
# for filename in os.listdir(folder):
#     if filename.endswith("jpg"): 
#         img_path = os.path.join(folder,filename)
#         try:
#             img = cv2.imread(img_path)
#             if face_extractor(img) is not None:
#                 face = cv2.resize(face_extractor(img), (200, 200))
#                 face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                
#                 file_name_path = 'teest/Brad/' + str(filename)
#                 cv2.imwrite(file_name_path, face)
#         except:
#             print(print(img_path+' error '))

# #Daniel
# folder = 'data-set/train/Daniel'
# for filename in os.listdir(folder):
#     if filename.endswith("jpg"): 
#         img_path = os.path.join(folder,filename)
#         try:
#             img = cv2.imread(img_path)
#             if face_extractor(img) is not None:
#                 face = cv2.resize(face_extractor(img), (200, 200))
#                 face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#                 os.mkdir
#                 file_name_path = 'teest/Daniel/'+str(filename)
#                 cv2.imwrite(file_name_path, face)
#         except:
#             print(print(img_path+' error '))

# #Elon
# folder = 'data-set/train/Elon'
# for filename in os.listdir(folder):
#     if filename.endswith("jpg"): 
#         img_path = os.path.join(folder,filename)
#         try:
#             img = cv2.imread(img_path)
#             if face_extractor(img) is not None:
#                 face = cv2.resize(face_extractor(img), (200, 200))
#                 face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

#                 file_name_path = 'teest/Elon/'+str(filename)
#                 cv2.imwrite(file_name_path, face)
#         except:
#             print(print(img_path+' error '))
  
# #Ema

# folder = 'data-set/train/Ema'
# for filename in os.listdir(folder):
#     if filename.endswith("jpg"): 
#         img_path = os.path.join(folder,filename)
#         try:
#             img = cv2.imread(img_path)
#             if face_extractor(img) is not None:
#                 face = cv2.resize(face_extractor(img), (200, 200))
#                 face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

#                 file_name_path = 'teest/Ema/'+str(filename)
#                 cv2.imwrite(file_name_path, face)
#         except:
#             print(print(img_path+' error '))

# #Emilia

# folder = 'data-set/train/Emilia'
# for filename in os.listdir(folder):
#     if filename.endswith("jpg"): 
#         img_path = os.path.join(folder,filename)
#         try:
#             img = cv2.imread(img_path)
#             if face_extractor(img) is not None:
#                 face = cv2.resize(face_extractor(img), (200, 200))
#                 face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

#                 file_name_path = 'teest/Emilia/'+str(filename)
#                 cv2.imwrite(file_name_path, face)
#         except:
#             print(print(img_path+' error '))

# #Maisie

# folder = 'data-set/train/Maisie'
# for filename in os.listdir(folder):
#     if filename.endswith("jpg"): 
#         img_path = os.path.join(folder,filename)
#         try:
#             img = cv2.imread(img_path)
#             if face_extractor(img) is not None:
#                 face = cv2.resize(face_extractor(img), (200, 200))
#                 face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

#                 file_name_path = 'teest/Maisie/'+str(filename)
#                 cv2.imwrite(file_name_path, face)
#         except:
#             print(print(img_path+' error '))

# #Obama

# folder = 'data-set/train/Obama'
# for filename in os.listdir(folder):
#     if filename.endswith("jpg"): 
#         img_path = os.path.join(folder,filename)
#         try:
#             img = cv2.imread(img_path)
#             if face_extractor(img) is not None:
#                 face = cv2.resize(face_extractor(img), (200, 200))
#                 face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

#                 file_name_path = 'teest/Obama/'+str(filename)
#                 cv2.imwrite(file_name_path, face)
#         except:
#             print(print(img_path+' error '))

# #Tom

# folder = 'data-set/train/Tom'
# for filename in os.listdir(folder):
#     if filename.endswith("jpg"): 
#         img_path = os.path.join(folder,filename)
#         try:
#             img = cv2.imread(img_path)
#             if face_extractor(img) is not None:
#                 face = cv2.resize(face_extractor(img), (200, 200))
#                 face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

#                 file_name_path = 'teest/Tom/'+str(filename)
#                 cv2.imwrite(file_name_path, face)
#         except:
#             print(print(img_path+' error '))

# #Trump

# folder = 'data-set/train/Trump'
# for filename in os.listdir(folder):
#     if filename.endswith("jpg"): 
#         img_path = os.path.join(folder,filename)
#         try:
#             img = cv2.imread(img_path)
#             if face_extractor(img) is not None:
#                 face = cv2.resize(face_extractor(img), (200, 200))
#                 face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

#                 file_name_path = 'teest/Trump/'+str(filename)
#                 cv2.imwrite(file_name_path, face)
#         except:
#             print(print(img_path+' error '))

# #Zuckerberg

# folder = 'data-set/train/Zuckerberg'
# for filename in os.listdir(folder):
#     if filename.endswith("jpeg"): 
#         img_path = os.path.join(folder,filename)
#         try:
#             img = cv2.imread(img_path)
#             if face_extractor(img) is not None:
#                 face = cv2.resize(face_extractor(img), (200, 200))
#                 face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

#                 file_name_path = 'teest/Zuckerberg/'+str(filename)
#                 cv2.imwrite(file_name_path, face)
#         except:
#             print(print(img_path+' error '))

# #arnold
# # folder = 'data-set/train/arnold_schwarzenegger'
# # for filename in os.listdir(folder):
# #     if filename.endswith("jpeg") or filename.endswith("jpg"): 
# #         img_path = os.path.join(folder,filename)
# #         try:
# #             img = cv2.imread(img_path)
# #             if face_extractor(img) is not None:
# #                 face = cv2.resize(face_extractor(img), (200, 200))
# #                 face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

# #                 file_name_path = 'teest/arnold_schwarzenegger/'+str(filename)
# #                 cv2.imwrite(file_name_path, face)
# #         except:
# #             print(print(img_path+' error '))


# # #will
folder = 'data-set/train/will smith'
for filename in os.listdir(folder):
    if filename.endswith("jpeg") or filename.endswith("jpg"): 
        img_path = os.path.join(folder,filename)
        try:
            img = cv2.imread(img_path)
            if face_extractor(img) is not None:
                face = cv2.resize(face_extractor(img), (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                file_name_path = 'faces/will smith/'+str(filename)
                cv2.imwrite(file_name_path, face)
        except:
            print(print(img_path+' error '))

# # #keanu

# # folder = 'data-set/train/keanu reeves'
# # for filename in os.listdir(folder):
# #     if filename.endswith("jpeg") or filename.endswith("jpg"): 
# #         img_path = os.path.join(folder,filename)
# #         try:
# #             img = cv2.imread(img_path)
# #             if face_extractor(img) is not None:
# #                 face = cv2.resize(face_extractor(img), (200, 200))
# #                 face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

# #                 file_name_path = 'faces/keanu reeves/'+str(filename)
# #                 cv2.imwrite(file_name_path, face)
# #         except:
# #             print(print(img_path+' error '))

# # #simon pegg

# # folder = 'data-set/train/simon pegg'
# # for filename in os.listdir(folder):
# #     if filename.endswith("jpeg") or filename.endswith("jpg"): 
# #         img_path = os.path.join(folder,filename)
# #         try:
# #             img = cv2.imread(img_path)
# #             if face_extractor(img) is not None:
# #                 face = cv2.resize(face_extractor(img), (200, 200))
# #                 face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

# #                 file_name_path = 'faces/simon pegg/'+str(filename)
# #                 cv2.imwrite(file_name_path, face)
# #         except:
# #             print(print(img_path+' error '))


# # #Alpacino

# # folder = 'data-set/train/Alpacino'
# # for filename in os.listdir(folder):
# #     if filename.endswith("jpeg") or filename.endswith("jpg"): 
# #         img_path = os.path.join(folder,filename)
# #         try:
# #             img = cv2.imread(img_path)
# #             if face_extractor(img) is not None:
# #                 face = cv2.resize(face_extractor(img), (200, 200))
# #                 face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

# #                 file_name_path = 'faces/Alpacino/'+str(filename)
# #                 cv2.imwrite(file_name_path, face)
# #         except:
# #             print(print(img_path+' error '))



# # #Benzema

# # folder = 'data-set/train/Benzema'
# # for filename in os.listdir(folder):
# #     if filename.endswith("jpeg") or filename.endswith("jpg"): 
# #         img_path = os.path.join(folder,filename)
# #         try:
# #             img = cv2.imread(img_path)
# #             if face_extractor(img) is not None:
# #                 face = cv2.resize(face_extractor(img), (200, 200))
# #                 face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

# #                 file_name_path = 'faces/Benzema/'+str(filename)
# #                 cv2.imwrite(file_name_path, face)
# #         except:
# #             print(print(img_path+' error ')) 


# # #Cristiano Ronaldo

# # folder = 'data-set/train/Cristiano Ronaldo'
# # for filename in os.listdir(folder):
# #     if filename.endswith("jpeg") or filename.endswith("jpg"): 
# #         img_path = os.path.join(folder,filename)
# #         try:
# #             img = cv2.imread(img_path)
# #             if face_extractor(img) is not None:
# #                 face = cv2.resize(face_extractor(img), (200, 200))
# #                 face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

# #                 file_name_path = 'faces/Cristiano Ronaldo/'+str(filename)
# #                 cv2.imwrite(file_name_path, face)
# #         except:
# #             print(print(img_path+' error '))

# # #Mbappe


# # folder = 'data-set/train/Mbappe'
# # for filename in os.listdir(folder):
# #     if filename.endswith("jpeg") or filename.endswith("jpg"): 
# #         img_path = os.path.join(folder,filename)
# #         try:
# #             img = cv2.imread(img_path)
# #             if face_extractor(img) is not None:
# #                 face = cv2.resize(face_extractor(img), (200, 200))
# #                 face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

# #                 file_name_path = 'faces/Mbappe/'+str(filename)
# #                 cv2.imwrite(file_name_path, face)
# #         except:
# #             print(print(img_path+' error '))


# # #Robert Dinero

# # folder = 'data-set/train/Robert Dinero'
# # for filename in os.listdir(folder):
# #     if filename.endswith("jpeg") or filename.endswith("jpg"): 
# #         img_path = os.path.join(folder,filename)
# #         try:
# #             img = cv2.imread(img_path)
# #             if face_extractor(img) is not None:
# #                 face = cv2.resize(face_extractor(img), (200, 200))
# #                 face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

# #                 file_name_path = 'faces/Robert Dinero/'+str(filename)
# #                 cv2.imwrite(file_name_path, face)
# #         except:
# #             print(print(img_path+' error '))


# # #Zidane
# # folder = 'data-set/train/Zidane'
# # for filename in os.listdir(folder):
# #     if filename.endswith("jpeg") or filename.endswith("jpg"): 
# #         img_path = os.path.join(folder,filename)
# #         try:
# #             img = cv2.imread(img_path)
# #             if face_extractor(img) is not None:
# #                 face = cv2.resize(face_extractor(img), (200, 200))
# #                 face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

# #                 file_name_path = 'faces/Zidane/'+str(filename)
# #                 cv2.imwrite(file_name_path, face)
# #         except:
# #             print(print(img_path+' error '))



# # folder = 'data-set/test/Obama'
# # for filename in os.listdir(folder):
# #     if filename.endswith("jpeg") or filename.endswith("jpg"): 
# #         img_path = os.path.join(folder,filename)
# #         try:
# #             img = cv2.imread(img_path)
# #             if face_extractor(img) is not None:
# #                 face = cv2.resize(face_extractor(img), (200, 200))
# #                 face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

# #                 file_name_path = 'data-set/'+str(filename)
# #                 cv2.imwrite(file_name_path, face)
# #         except:
# #             print(print(img_path+' error '))






# print('Face extraction completed')