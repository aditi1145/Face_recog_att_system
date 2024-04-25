from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pickle
import cv2
import csv
import os
import time
from datetime import datetime

COL_NAMES = ['NAME', 'TIME', 'DATE']

st.set_page_config(
    page_icon=":raised_hand_with_fingers_splayed:",
    page_title="Face Recognition",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Face Recognition Interface")

with open('data/names.pkl', 'rb') as w:
    LABELS=pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES=pickle.load(f)


knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgBackground=cv2.imread("background.png")

facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

if st.button("Start Recognition"):
    video=cv2.VideoCapture(0)
    st_frame = st.empty()
    cap = True
    if st.button("Capture"):
        cap = False
    while video.isOpened() and cap: 
        ret,frame = video.read()
        if ret:
            col1, col2, col3 = st.columns(3)
            with col2:
                gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces=facedetect.detectMultiScale(gray, 1.3 ,5)
                for (x,y,w,h) in faces:
                    crop_img = frame[y:y+h, x:x+w, :]
                    resized_img=cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)
                    output=knn.predict(resized_img)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
                    cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
                    cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
                    st.session_state["output"] = str(output[0])
                
                st_frame.image(
                    frame, caption="Detections", channels="BGR", use_column_width=False
                )
if st.button("Record"):
    ts=time.time()
    date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
    timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
    exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
    with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
        writer=csv.writer(csvfile)
        if not exist:
            writer.writerow(COL_NAMES)
        writer.writerow([st.session_state["output"], timestamp, date])
        csvfile.close()
    st.write(f"Attendance for '{st.session_state["output"]}' added to DB")