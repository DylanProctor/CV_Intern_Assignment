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

model_points = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65),
    (-255.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
])

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

        if rect == min_box:
            pts = shape

        img_pts = np.array([
            pts[30],
            pts[8],
            pts[36],
            pts[45],
            pts[48],
            pts[54]
        ], dtype = 'double')

        focal_len = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array([
            [focal_len, 0, center[0]],
            [0, focal_len, center[1]],
            [0, 0, 1]
        ], dtype = 'double')

        dist_coeffs = np.zeros((4,1))
        (sucess, rot_vec, trans_vec) = cv2.solvePnP(model_points, img_pts, camera_matrix, dist_coeffs, flags = cv2.SOLVEPNP_ITERATIVE)

        (nose_end, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rot_vec, trans_vec, camera_matrix, dist_coeffs)

        p1 = (int(img_pts[0][0]), int(img_pts[0][1]))
        p2 = (int(nose_end[0][0][0]), int(nose_end[0][0][1]))
        cv2.line(frame, p1, p2, (255, 0, 0), 2)
        
    objects = ct.update(rectangles)

    for (objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] , centroid[1] ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break



    

video_capture.release()
cv2.destroyAllWindows()

