from __future__ import print_function
from random import randint
import cv2 as cv
import argparse
import numpy as np
from PIL import Image
import tkinter as tk


def detectAndDisplay(frame, overlayimage):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    # screen dimension
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()   
    # height, width, number of channels in image
    height = frame.shape[0]
    width = frame.shape[1]
    
    #ratio
    rtio = (screen_width/2)/width

    #new frame dimension 
    frame_ndim = (int(width*rtio), int(height*rtio))

    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        # remove eclipse on frame
        #frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)

        overlay_img_dim = (w, h)
        overlay_img = cv.resize(cv.imread(overlayimage), overlay_img_dim)

        # face dimensions
        point1_dim = (x,y)
        point2_dim = (x+w,y+h)

        alpha_s = overlay_img[:, :, 2] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            frame[y:y+h, x:x+w, c] = (alpha_s * overlay_img[:, :, c] + alpha_l * frame[y:y+h, x:x+w, c])

        #  inject frames but it gives background, so not the be4st
        #frame[y:y+h,x:x+h] = overlay_img
        
        #removing eye circles
        #faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        #eyes = eyes_cascade.detectMultiScale(faceROI)
        #for (x2,y2,w2,h2) in eyes:
        #    eye_center = (x + x2 + w2//2, y + y2 + h2//2)
        #    radius = int(round((w2 + h2)*0.25))
        #    #frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)

    # rezsizing frame
    frame = cv.resize(frame, frame_ndim)
    cv.imshow('Capture - Face detection', frame)

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascades/haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()

face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade

face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()

#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)

camera_device = args.camera
#-- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
mycounter = 0
overlay_file = "overlay1.png"
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    
    mycounter = mycounter + 1
    if ((mycounter % 10) > 8):
        # randomize input image
        val = randint(1, 5)
        overlay_file = "overlay" + str(val) + ".png"
        print(overlay_file)
    print(mycounter)
    detectAndDisplay(frame, overlay_file)

    if cv.waitKey(10) == 27:
        break