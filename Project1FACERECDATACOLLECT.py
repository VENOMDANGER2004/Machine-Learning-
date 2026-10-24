#Read a Image/Video from Web Cam Using Opencv
#Face Detection in Video
#Click 20 pictures of the person who comes in front of the camera and save them as numpy

import cv2
import os
import numpy as np
#Create a Camera Object
cam=cv2.VideoCapture(0)

#Ask the name
file_name= input("Enter person name: ")
dataset_path="./data/"
offset=20
os.makedirs(dataset_path,exist_ok=True)
#Model
model=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#Create a list to save face data
faceData=[]
skip=0

#Read Image From Camera Object
while True:
    success, img = cam.read()
    Best=cv2.GaussianBlur(img,(7,7),0)
    if not success:
        print("Failed to read camera image")
    #Store the gray images
    gray = cv2.cvtColor(Best, cv2.COLOR_BGR2GRAY)

    faces=model.detectMultiScale(Best,1.1,2)
    #pick the face with the largest box area
    faces=sorted(faces, key=lambda f:f[2]*f[3])
    #pick the largest face
    if len(faces)>0:
       f=faces[-1]
       x,y,w,h=f
       cv2.rectangle(img,(x, y),(x + w, y + h),(255, 0, 0),2)
       cropped_face=img[y:y+h,x:x+w]

       #Crop and save the largest face
       resized_face=cv2.resize(cropped_face,(100,100))
       skip += 1
       if skip % 10==0:
           faceData.append(resized_face)
           print("Saved so far: "+str(len(faceData)))


    cv2.imshow("Camera Image", img)
    #cv2.imshow("Cropped Face",cropped_face)


    key=cv2.waitKey(1)
    if key == ord('q'):
        break

#Write the face Data on the disk
faceData=np.asarray(faceData)
print(faceData.shape)
m=faceData.shape[0]
faceData=faceData.reshape(m,-1)

print(faceData.shape)

#Save on the Disk as np array
filepath=dataset_path+file_name+".npy"
np.save(filepath,faceData)
print("Data Saved Successfully" + filepath)

#RElease Camera and Destroy Window
cam.release()
cv2.destroyAllWindows()