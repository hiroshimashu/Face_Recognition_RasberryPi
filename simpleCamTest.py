import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import sys
import os



SAVE_PATH = '/home/pi/result/'
FILE_NAME = 'stream.jpg'

def take_photos_and_crop():
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
                cv2.imwrite(SAVE_PATH + str(face_detect_count) + FILE_NAME, image[y:y + h, x:x+w])
                face_detect_count = face_detect_count + 1
    cv2.imshow('img',image)
    rawCapture.truncate(0)
    
    if key == ord("q"):
        break

take_photos_and_crop()