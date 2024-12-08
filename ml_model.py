import pickle
from flask import Flask,render_template,url_for,request,redirect
import cv2
app=Flask(__name__)
 
@app.route('/',methods=['GET','POST'])
def a():
   return render_template("web.html")
 


@app.route('/predict',methods=['POST']) 
def predict():
   img=request.form['image']
   i=cv2.imread(img+'.jpg')
   if i is not None:
      i=cv2.resize(i,(500,500),interpolation=cv2.INTER_LINEAR)
      img=i.flatten()
      with open("model",'rb') as f:
          mode=pickle.load(f)
      im=mode.predict([img])
      return render_template("reweb.html",post=im)
   return redirect('/')  
if __name__=="__main__":
   app.run(debug=True)