import cv2
import numpy as np
import streamlit as st
from keras.models import model_from_json
from google.protobuf.internal import builder

#load the model
with open('models/model.json','r') as model_json_file:
    model_json= model_json_file.read()
    model= model_from_json(model_json)
    model.load_weights("model/model.h5")

#define the emotions
emotions=['angry','disgust','fear','happy','sad','surprised','neutral']

#create a function to predict the emotion

def predict_emotion(img):
    img=cv2.resize(img, [48,48])
    img= np.reshape(img, [1,48,48,1])
    prediction= model.predict(img)
    return emotions[np.argmax(prediction)]

#create function to detect face in frame
def detect_face(frame):
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade= cv2.CascaseClassifier("/home/rubi/Desktop/emotion-detection/models/haarcascade_frontalface_default.xml")
    faces= face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces)>0:
        (x,y,w,h)= face[0]
        return gray[y:y+h, x:x+w], faces[0]
    else:
        return None

#create the streamlitapp
st.set_page_config(page_title='Emotion Detection', page_icon=':guardsman:', layout='wide')
st.title('Emotion detection using OpenCV and Keras')  

#add checkbox to toggle the webcam
webcam_enabled= st.checkbox("Enable webcam")  

if webcam_enabled:
    webcam=cv2.VideoCapture(0)
    while True:
        _,frame= webcam.read()
        face, rect= detect_face(frame)
        if face is not None:
            emotion= predict_emotion(face)
            (x,y,w,h) =rect
            cv2.putText(frame,emotion,(x,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.8(0,255,0),2)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow("webcam", frame)
        key= cv2.waitkey(1)
        if key==ord('q'):
            webcam.release()
            cv2.destroyAllWindows()
            break
else:
    cv2.destroyAllWindows()
 