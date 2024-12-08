from sklearn.neighbors import KNeighborsClassifier
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy
import pickle
images=[]

#loop the folder which contains the images for the training
for a in range(1,21):
    i=cv2.imread('p'+str(a)+'.jpg')
    if i is not None:
      i=cv2.resize(i,(500,500),interpolation=cv2.INTER_LINEAR)
      images.append(i.flatten())

# store color names present in the images 
colors=['red','green','blue','blue','blue','black','white','red','green','pink','pink','pink','yellow','yellow','orenge','orenge','orenge','violate','violate','violate']

knn=KNeighborsClassifier(n_neighbors=2)


knn.fit(images,colors)
 
with open('ml_model','w') as f:
   pickle.dump(knn,f)

 
 
