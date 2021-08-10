from centroidTracker import CentroidTracker
import numpy as np
import dlib
import imutils
import time
import cv2
import os
import math

def rect_2_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x 
    h = rect.bottom() - y
    return(x,y,w,h)

def shape_np(shape, dtype = 'int'):
    coords = np.zeros((68,2), dtype = dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords
    
frame_wid = 1024
frame_hei = 576
fps = 30

ct = CentroidTracker()
face_det = dlib.get_frontal_face_detector()
shape_pred = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
video_capture = cv2.VideoCapture(0)
time.sleep(1.5)

while True:
    ret, frame = video_capture.read()
    frame = imutils.resize(frame, width = frame_wid, height = frame_hei)
    size = frame.shape
    center = (size[1]/2, size[0]/2)
    min_dist_center = float('inf')

    rectangles = []

    rects = face_det(frame, 0)

    if len(rects) > 0:
        for rect in rects:
            dist = math.sqrt((rect.center().x - center[0])**2 + (rect.center().y - center[1])**2)
            if dist < min_dist_center:
                min_dist_center = dist
                min_box = rect

    for rect in rects:
        (x, y, w, h) = rect_2_bb(rect)

        rectangles.append((x, y, w, h))

        if rect == min_box:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        shape = shape_pred(frame, rect)
        shape = shape_np(shape)

        for (x, y) in shape:
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        
    objects = ct.update(rectangles)

    for (objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] , centroid[1] ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


    

video_capture.release()
cv2.destroyAllWindows()

