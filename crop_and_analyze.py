import sys
sys.path.append("/home/pi/.virtualenvs/cv3/lib/python3.5/site-packages/")
import boto3
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import os



SAVE_PATH = '/home/pi/result/'
FILE_NAME = 'stream.jpg'
BUCKET_NAME = 'simple-recognizer-repo'
s3 = boto3.resource('s3')


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size = (640, 480))
face_detect_count = 0

time.sleep(0.1)

for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    image = frame.array
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            OUTPUT_DIR = SAVE_PATH + str(face_detect_count) + FILE_NAME
            print(OUTPUT_DIR)
            cv2.imwrite(OUTPUT_DIR, image[y:y + h, x:x+w])
            try:
                s3.Object(BUCKET_NAME, str(face_detect_count) + FILE_NAME).upload_file(OUTPUT_DIR)
                print('successfully uploaded')
            except:
                print('uploda has failed')
            face_detect_count = face_detect_count + 1
    
    cv2.imshow('img',image)
    key = cv2.waitKey(1) & 0xFF
    
    rawCapture.truncate(0)





