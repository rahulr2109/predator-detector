import numpy as np
import pandas as pd
from fastapi import FastAPI
from PIL import Image
from joblib import load
from fastapi.middleware.cors import CORSMiddleware
import keras.utils as image
from io import BytesIO
from ultralytics import YOLO
import time
import torch
import cv2
import os
import csv

CONFIDENCE_THRESHOLD = 0.85
COLOR = (0, 255, 0)

ans = NULL


def detect(image, model, display=False):
    start = time.time()
    # pass the image through the model and get the detections
    ans = (model.predict(image)[0].names[0])
    detections = model.predict(image)[0].boxes.data
    # print(detections)

    # check to see if the detections tensor is not empty
    if detections.shape != torch.Size([0, 6]):

        # initialize the list of bounding boxes and confidences
        boxes = []
        confidences = []
        classes = []

        # loop over the detections
        for detection in detections:
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = detection[4]

            # filter out weak detections by ensuring the confidence
            # is greater than the minimum confidence
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            # if the confidence is greater than the minimum confidence, add
            # the bounding box and the confidence to their respective lists
            boxes.append(detection[:4])
            confidences.append(detection[4])
            classes.append(detection[5])

        print(f"{len(boxes)} Number plate(s) have been detected.")
        # initialize a list to store the bounding boxes of the
        # number plates and later the text detected from them
        number_plate_list = []

        # loop over the bounding boxes
        for i in range(len(boxes)):
            # extract the bounding box coordinates
            xmin, ymin, xmax, ymax = int(boxes[i][0]), int(
                boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])
            # append the bounding box of the number plate
            number_plate_list.append(
                [[xmin, ymin, xmax, ymax], confidences[i], classes[i]])

            # draw the bounding box and the label on the image
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), COLOR, 2)
            text = "{}: {:.2f}%".format(classes[i], confidences[i] * 100)
            cv2.putText(image, text, (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)

            if display:
                # crop the detected number plate region
                number_plate = image[ymin:ymax, xmin:xmax]
                # display the number plate
                cv2.imshow(f"Number plate {i}", number_plate)

        end = time.time()
        # show the time it took to detect the number plates
        print(
            f"Time to detect the number plates: {(end - start) * 1000:.0f} milliseconds")
        # return the list containing the bounding
        # boxes of the number plates
        return number_plate_list
    # if there are no detections, show a custom message
    else:
        print("No number plates have been detected.")
        return []


model = YOLO("best.pt")

file_path = "test.jpeg"

image = cv2.imread(file_path)

l = detect(image, model)

# if (l == []):
#     print("No animal have been detected.")
# else:
#     for item in l:
#         print(item)

# cv2.imshow('Image', image)
# cv2.waitKey(0)
