import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300
counter = 0

labels = ["A", "B", "C", "D", "E", "F", "G", "H",
          "I", "J", "K", "L", "M", "N", "O", "P",
          "Q", "R", "S", "T", "U", "V", "W", "X",
          "Y", "Z", "_"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]


        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            imgCrop = cv2.resize(imgCrop, (imgSize, imgSize))
            imgWhite[0:imgSize, 0:imgSize] = imgCrop
            prediction, index = classifier.getPrediction(imgWhite, draw = False)
            print(prediction, index)

            cv2.putText(imgOutput, labels[index], (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 2,(255, 0, 255), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset),
                          (x + w+offset, y + h+offset), (255, 0, 255), 4)
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
        cv2.destroyAllWindows()


