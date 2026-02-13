import cv2 as cv
import mediapipe as mp
import numpy as np
import time

# imports custom
from tracker import eye_tracker, landmark_extract
from calibration import EyeCalibration

class BlinkCounter:
    def __init__(self, video_path, model_path='face_landmarker.task'):
        self.video_path = video_path

        # Init avec tracker.py
        self.tracker = eye_tracker(model_path)
        self.extractor = landmark_extract()

        # Detection logic
        self.ear_threshold = 0.3 # Valeur par défaut, sera modifiée avec la calibration -A
        self.consec_frames = 2
        self.blink_counter = 0
        self.frame_counter = 0
        
        # Calibration
        self.calibrator = EyeCalibration(frames_to_capture=40)

    def eye_aspect_ratio(self, eye_points):
        # eye_points est une liste de dicts dans traceker.py
        p = [pt['pixel'] for pt in eye_points]
        
        # Distances
        A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
        B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
        C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
        return (A + B) / (2.0 * C)

    def update_count(self, ear):
        if ear < self.ear_threshold:
            self.frame_counter += 1
        else:
            if self.frame_counter >= self.consec_frames:
                self.blink_counter += 1
            self.frame_counter = 0

    def process_video(self):
        cap = cv.VideoCapture(self.video_path)
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        window_name = "Blink Counter Modular"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(window_name, 960, 540)
        
        t0 = time.monotonic()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'): break

            # Conversion MediaPipe
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Calcul du timestamp pour le mode async
            timestamp = int((time.monotonic() - t0) * 1000)
            # Async
            self.tracker.landmarker.detect_async(mp_image, timestamp)
            
            # Récolte des données dans tracker.py
            latest = self.tracker.store.latest

            if latest:
                result, _ = latest
                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    
                    # Extraction des données
                    r_data, l_data = self.extractor.extract_data(landmarks, h, w)

                    # Calcul du EAR
                    r_ear = self.eye_aspect_ratio(r_data)
                    l_ear = self.eye_aspect_ratio(l_data)
                    avg_ear = (r_ear + l_ear) / 2.0

                    # Logique de calibration
                    if not self.calibrator.is_complete():
                        finished = self.calibrator.process(frame, avg_ear, key)
                        if finished:
                            self.ear_threshold = self.calibrator.get_threshold()
                            print(f"Seuil: {self.ear_threshold}")
                        
                        # Feedback visuel pour la calibration
                        for pt in r_data + l_data:
                            cv.circle(frame, pt['pixel'], 1, (0, 255, 255), -1)

                    else:
                        self.update_count(avg_ear)
                        color = (0, 0, 255) if avg_ear < self.ear_threshold else (0, 255, 0)

                        # Dessin
                        for pt in r_data + l_data:
                            cv.circle(frame, pt['pixel'], 1, color, -1)
                        
                        # Affichage des infos
                        cv.putText(frame, f"Clignements: {self.blink_counter}", (30, 50), 
                                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv.putText(frame, f"EAR: {avg_ear:.2f} | Seuil: {self.ear_threshold:.2f}", (30, 90), 
                                   cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv.imshow(window_name, frame)

        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    app = BlinkCounter(0)
    app.process_video()
