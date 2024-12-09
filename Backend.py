import pickle
from flask import Flask , render_template, url_for, request, redirect
import cv2
import os
 
app=Flask(__name__)
app.config['upload_folder']="uploads/"

def p(im):
   i=cv2.imread(im)
   if i is not None:
      i=cv2.resize(i,(500,500),interpolation=cv2.INTER_LINEAR)
      img=i.flatten()
      with open("model",'rb') as f:
          mode=pickle.load(f)
      im=mode.predict([img])
      return im
   return None
 
@app.route('/',methods=['GET','POST'])
def a():
   return render_template("home.html")
 


@app.route('/predict',methods=['POST']) 
def predict():
   img=request.files['image']  
   im=os.path.join(app.config['upload_folder'],'upload.jpg')
   img.save(im)
   im=p(im)
   if im!=None:
      return render_template("result.html",post=im)
   return redirect('/')  
if __name__=="__main__":
   app.run(debug=True)
