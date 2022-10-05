import cv2
import numpy as np
import math
import keras
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
diff = 20
img_size = 300
counter = 0
folder = 'Images/Neutral'
labels = ["Approve", "Dissapprove", "Neutral"]
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img) #hand detector from lib

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox'] #get bbox
        img_white = np.ones((img_size,img_size, 3), np.uint8)*255

        img_crop = img[y-diff : y+h+diff, x-diff : x+w+diff]
        img_crop_shape = img_crop.shape

        #img_white[0:img_crop_shape[0], 0:img_crop_shape[1]] = img_crop
        ratio = h/w

        if ratio > 1:
            k = img_size/h
            w_cal = math.ceil(k*w)
            img_resize = cv2.resize(img_crop, (w_cal, img_size))
            img_resize_shape = img_resize.shape

            w_gap = math.ceil((img_size-w_cal)/2)
            img_white[:, w_gap:w_cal+w_gap] = img_resize
            prediction, index = classifier.getPrediction(img)
            print(prediction, index)

        else:
            k = img_size / w
            h_cal = math.ceil(k * h)
            img_resize = cv2.resize(img_crop, (img_size, h_cal))
            img_resize_shape = img_resize.shape

            h_gap = math.ceil((img_size - h_cal) / 2)
            img_white[h_gap:h_cal + h_gap, :] = img_resize

        cv2.imshow("Cropped", img_crop)
        cv2.imshow("Img_white", img_white)

    cv2.imshow("Stream", img)
    cv2.waitKey(1)

    #if key == ord('s'):
    #    counter += 1
    #    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', img_white)
    #    print(counter)