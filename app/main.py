from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from app.music_sheet_classifier import MusicSheetClassifier

app = FastAPI()
classifier = MusicSheetClassifier()

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image format"})
    
    result = classifier.classify(img)
    return result
