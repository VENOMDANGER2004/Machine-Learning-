import cv2
import numpy as np
import os
#Data
dataset_path="./data/"
faceData=[]
labels=[]
classID=0
nameMap={}
offset=10
for f in os.listdir(dataset_path):
    if f.endswith(".npy"):

        nameMap[classID]=f[:-4]

        #X-Values
        dataItem=np.load(dataset_path+f)
        m=dataItem.shape[0]
        faceData.append(dataItem)

        #Y-Values
        target= classID * np.ones((m,))
        classID=classID+1
        labels.append(target)

X=np.concatenate(faceData,axis=0)
Y=np.concatenate(labels,axis=0)
print(X.shape)
print(Y.shape)
print(nameMap)

#Algorithm
def dist(p,q):
    return np.sqrt(np.sum((p-q)**2))
def knn(X,Y,xt,k=5):
    m=X.shape[0]
    dlist=[]

    for i in range(m):
        d=dist(X[i],xt)
        dlist.append((d,Y[i]))

    dlist= np.array(sorted(dlist))
    dlist= np.array(dlist[:k])
    labels=dlist[:,1]

    labels,cnts=np.unique(labels,return_counts=True)
    idx=cnts.argmax()
    pred=labels[idx]

    return int(pred)

#Predictions
cam=cv2.VideoCapture(0)
model=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
while True:
    success, img = cam.read()

    if not success:
        print("Failed to read camera image")

    img = cv2.GaussianBlur(img, (5, 5), 0)
    faces=model.detectMultiScale(img,1.2,4)

    #render a box around each face and predicts its name
    for f in faces:
       x,y,w,h=f
       print(f)
       cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
       #Crop and save the largest face
       cropped_face=img[y-offset:y+h+offset,x-offset:x+w+offset]
       if cropped_face.shape[0]==0 or cropped_face.shape[1]==0:
           continue

       resized_face=cv2.resize(cropped_face,(100,100))

       #Predict the name using KNN
       classPred=knn(X,Y,resized_face.flatten())
        #Name
       namePred=nameMap[classPred]
       print(namePred)
        #Display the Name and Box
       cv2.putText(img,namePred,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_AA)
       cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)


    cv2.imshow("Image Window",img)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
