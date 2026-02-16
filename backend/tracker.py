import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "face_landmarker.task"

class ResultStore:
    def __init__(self):
        self.latest = None 

    def callback(self, result, output_image, timestamp_ms):
        self.latest = (result, timestamp_ms)
    
class landmark_extract:
    def __init__(self):
        self.right_eye = [33, 159, 158, 133, 153, 145]
        self.left_eye = [362, 380, 374, 263, 386, 385]

    def eye_data(self, landmarks, indices, height, width):
        data_array = []

        for idx in indices:
            lm = landmarks[idx]
            
            data = {
                "index": idx,
                "normalized": (lm.x, lm.y, lm.z),
                "pixel": (int(lm.x * width), int(lm.y * height))
            }

            data_array.append(data)

        return data_array
    
    def extract_data(self, landmarks, height, width):
        right_eye_data = self.eye_data(landmarks, self.right_eye, height, width)
        left_eye_data = self.eye_data(landmarks, self.left_eye, height, width)

        return right_eye_data, left_eye_data


class eye_tracker:
    def __init__(self, model_path: str, num_faces: int = 1):
        self.model_path = model_path

        self.store = ResultStore()
        
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.store.callback,
            num_faces=1) 

        self.landmarker = vision.FaceLandmarker.create_from_options(options)
