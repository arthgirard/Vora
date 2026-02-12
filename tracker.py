
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



    
class App:
    def __init__(self, settings):
        self.settings = settings # settings est l'objet de eye_tracker
        self.extractor = landmark_extract()
    def run(self): 

        cap = cv2.VideoCapture(0)

        t0 = time.monotonic()

        while True:
            ret, frame_cv = cap.read()   # ret = success flag, frame = image
            # si ret(success flag) ne foncitonne pas, arreter le loop
            if not ret:
                break

            # Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
            # fps = cap.get(cv2.CAP_PROP_FPS)
            # print("FPS:", fps)

    
            # You’ll need it to calculate the timestamp for each frame. On utilise la librairie time et on creer un timestamp pour chaque millisecondes
            self.timestamps = int((time.monotonic()-t0) * 1000)
    
            # Convert the frame received from OpenCV to a MediaPipe’s Image object.
            frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Send live image data to perform face landmarking.
            # The results are accessible via the `result_callback` provided in
            # the `FaceLandmarkerOptions` object.
            # The face landmarker must be created with the live stream mode.
            self.settings.landmarker.detect_async(mp_image, self.timestamps)

            latest = self.settings.store.latest
            if latest is not None:
                result, ts = latest
                if result.face_landmarks:
                        h, w = frame_cv.shape[:2]
                        landmarks = result.face_landmarks[0]

                        left_eye, right_eye = self.extractor.extract_data(landmarks, h, w)

                        # draw every eye
                        for point in left_eye:
                            xl, yl = point["pixel"]
                            cv2.circle(frame_cv, (xl, yl), 2, (0, 255, 0), -1)

                        for point in right_eye:
                            xr, yr = point["pixel"]
                            cv2.circle(frame_cv, (xr, yr), 2, (0, 255, 0), -1)

                        print(f"left: {xl}, {yl} right: {xr}, {yr}")


            cv2.imshow("Webcam", frame_cv) #montre la video  frame== cap.read()
            if cv2.waitKey(1) & 0xFF == ord('q'): #pour quitter, appuyer sur "q"
                break



def main():
    instance = App(eye_tracker(model_path, 1))
    instance.run()
    
if __name__ == "__main__":
    main()


#
# def main():
#     class ResultStore:
#         def __init__(self):
#             self.latest = None  # (result, timestamp_ms)
#
#         def callback(self, result, output_image, timestamp_ms):
#             self.latest = (result, timestamp_ms)
#     #a la place de ca:
#     #def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
#     #   print('face landmarker result: {}'.format(result))
#     
#     store = ResultStore()
#
#     BaseOptions = mp.tasks.BaseOptions
#     FaceLandmarker = mp.tasks.vision.FaceLandmarker
#     FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
#     FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
#     VisionRunningMode = mp.tasks.vision.running_mode
#
#
#     options = FaceLandmarkerOptions(
#         base_options=BaseOptions(model_asset_path=model_path),
#         running_mode=VisionRunningMode.LIVE_STREAM,
#         result_callback=store.callback,
#         num_faces=1)
#     # def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
#     # print('face landmarker result: {}'.format(result))
#
#
#
#     #etape 1: preparer les donnees
#     # Use OpenCV’s VideoCapture to start capturing from the webcam.
#
#     # Create a loop to read the latest frame from the camera using VideoCapture#read()
#
#     # Convert the frame received from OpenCV to a MediaPipe’s Image object.
#
#     landmarker = vision.FaceLandmarker.create_from_options(options)
#     cap = cv2.VideoCapture(0)
#
#     t0 = time.monotonic()
#
#     while True:
#         ret, frame_cv = cap.read()   # ret = success flag, frame = image
#         # si ret(success flag) ne foncitonne pas, arreter le loop
#         if not ret:
#             break
#
#         # Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         print("FPS:", fps)
#
#     
#         # You’ll need it to calculate the timestamp for each frame. On utilise la librairie time et on creer un timestamp pour chaque millisecondes
#         timestamps = int((time.monotonic()-t0) * 1000)
#
#         # Convert the frame received from OpenCV to a MediaPipe’s Image object.
#         frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
#
#         # Send live image data to perform face landmarking.
#         # The results are accessible via the `result_callback` provided in
#         # the `FaceLandmarkerOptions` object.
#         # The face landmarker must be created with the live stream mode.
#         results = landmarker.detect_async(mp_image, timestamps)
#
#
#         if store.latest is not None:
#             result, ts = store.latest
#             if result.face_landmarks:
#                 h, w = frame_cv.shape[:2]
#                 landmarks = result.face_landmarks[0]  # first face
#
#                 for lm in landmarks:
#                     x = int(lm.x * w)
#                     y = int(lm.y * h)
#                     # OpenCV color is BGR:
#                     cv2.circle(frame_cv, (x, y), 2, (0, 255, 0), -1)  # green dot
#
#
#
#         cv2.imshow("Webcam", frame_cv) #montre la video  frame== cap.read()
#         # press 'q' to quit
#         if cv2.waitKey(1) & 0xFF == ord('q'): #pour quitter, appuyer sur "q"
#             break
#
#
#
# if __name__ == "__main__":
#     main()




















