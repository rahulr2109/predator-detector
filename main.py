import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
from joblib import load
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from ultralytics import YOLO
import time
import torch
import cv2
import os
import csv

CONFIDENCE_THRESHOLD = 0.85
COLOR = (0, 255, 0)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("best.pt")


class Response(BaseModel):
    animal: str = None


@app.post("/predict", response_model=Response)
async def predict(image: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_data = await image.read()

        # Convert image data to PIL Image
        image_pil = Image.open(BytesIO(image_data))

        # Convert PIL Image to OpenCV format
        image_cv = np.array(image_pil)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        # Perform detection using YOLO model
        detections = model.predict(image_cv)[0].boxes.data

        # Initialize response
        response = Response()

        # Check if detections are not empty
        if detections.shape != torch.Size([0, 6]):
            # Loop over the detections
            for detection in detections:
                # Extract confidence and class
                confidence = detection[4]
                class_id = int(detection[5])
                class_name = model.names[class_id]

                # Filter out weak detections
                if confidence >= CONFIDENCE_THRESHOLD:
                    # Draw bounding box on the image
                    xmin, ymin, xmax, ymax = map(int, detection[:4])
                    cv2.rectangle(image_cv, (xmin, ymin),
                                  (xmax, ymax), COLOR, 2)
                    cv2.putText(image_cv, f"{class_name}: {confidence:.2f}%", (
                        xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)

                    # Set detected class as response
                    response.animal = class_name
                    break  # Break after detecting the first object above threshold

        # Convert OpenCV image back to PIL format for response
        image_pil_with_boxes = Image.fromarray(
            cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

        # Save image with detections (optional)
        # image_pil_with_boxes.save("detections.jpg")

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
