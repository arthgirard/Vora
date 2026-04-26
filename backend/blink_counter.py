# eye_aspect_ratio, update_count et process_video sont trois fonctions fortement modifiées provenant de https://github.com/Pushtogithub23/Eye-Blink-Detection-using-MediaPipe-and-OpenCV/tree/master
# dessin sur le visage codé avec l'aide de Gemini

import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import base64
import threading as _threading

# imports custom
from tracker import eye_tracker, landmark_extract
from ergo_timer import ErgoTimer
from database_manager import Manager

import threading
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class BlinkCounter:
    def __init__(self, model_path='face_landmarker.task'):

        # dernière frame encodée en base64, lue par la boucle async du frontend
        self._frame_lock = _threading.Lock()
        self.latest_frame = None

        # Init avec tracker.py 
        self.tracker = eye_tracker(model_path)
        self.extractor = landmark_extract()
        self.blink_database = Manager()
        
        # Logique de détection, version améliorée avec double moyennes mobiles
        self.ear_history = []
        self.slow_history_length = 30 # baseline
        self.fast_history_length = 3 # filtre anti-grain

        # Calibration
        self.recent_blink_peaks = []

        self.earm_threshold_close = 0.08
        self.earm_threshold_open = 0.025

        # Machine à états
        self.is_blinking = False
        self.blink_duration = 0
        self.current_blink_max = 0.0
        self.min_blink_frames = 3
        self.max_blink_frames = 15 # pour faire la distinction entre un clignement spontané et un regard dans une autre direction
       
        self.blink_counter = 0
        self.nb_blink_total_minute = 0
        self.freq_stamp = 0 

        self.frame_counter = 0
        
        # Ajout d'une variable de contrôle pour que le thread s'arrête proprement avec Flet
        self.running = True
        
        # Initialisation du timer ergo
        self._seuil_clignements = 10
        self.ergo_timer = ErgoTimer(db_manager=self.blink_database, seuil=self._seuil_clignements)

    @property
    def seuil_clignements(self):
        return self._seuil_clignements

    @seuil_clignements.setter
    def seuil_clignements(self, value):
        self._seuil_clignements = value
        # Sécurité pour s'assurer que ergo_timer existe avant de modifier son attribut
        if hasattr(self, 'ergo_timer'):
            self.ergo_timer.LOW_BLINK_THRESHOLD = value

    def eye_aspect_ratio(self, eye_points):
        # Distances
        p = []
        for pt in eye_points:
            nx, ny, nz = pt['normalized']
            px = nx * self.w
            py = ny * self.h
            pz = nz * self.w #proportionnel à la largeur selon MediaPipe
            p.append(np.array([px, py, pz]))

        A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
        B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
        C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
        return (A + B) / (2.0 * C)

    def update_count(self, earm):
        if not self.is_blinking:
            if earm > self.earm_threshold_close:
                self.is_blinking = True
                self.blink_duration = 1
                self.current_blink_max = earm 
        else:
            self.blink_duration += 1
            # On cherche la valeur la plus haute atteinte pendant ce clignement
            if earm > self.current_blink_max:
                self.current_blink_max = earm
                
            if earm < self.earm_threshold_open:
                if self.min_blink_frames <= self.blink_duration <= self.max_blink_frames:
                    self.blink_counter += 1
                    
                    # On garde en mémoire les 10 derniers clignements
                    self.recent_blink_peaks.append(self.current_blink_max)
                    if len(self.recent_blink_peaks) > 10:
                        self.recent_blink_peaks.pop(0)
                        
                    median_peak = np.median(self.recent_blink_peaks)
                    nouveau_seuil = self.earm_threshold_open + ((median_peak - self.earm_threshold_open) * 0.40)
                    
                    self.earm_threshold_close = max(self.earm_threshold_open + 0.03, nouveau_seuil)
                
                self.is_blinking = False
                self.blink_duration = 0
                self.current_blink_max = 0.0

    def get_freq_data(self, start):
        elapse = time.monotonic() - start
        current_minute = int(elapse // 60)

        if current_minute > self.freq_stamp:
            self.nb_blink_minute = self.blink_counter - self.nb_blink_total_minute

            is_reliable = 1
            if self.ergo_timer.absence_start is not None:
                is_reliable = 0
            
            is_low = 1 if self.nb_blink_minute < self.seuil_clignements else 0
            # On logue avec le marquer
            self.blink_database.minute_log(current_minute, self.nb_blink_minute, is_reliable, is_low)
        
            self.nb_blink_total_minute = self.blink_counter
            self.freq_stamp = current_minute

    def process_video(self):
        cap = cv.VideoCapture(0)
            
        # reduit le buffer opencv a 1 pour eviter le lag de la video
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

        self.w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        t0 = time.monotonic()
        last_timestamp = -1
        start_timer_freq = time.monotonic()

        # Ajout de la condition 'self.running' pour lier l'exécution à l'état du frontend
        while cap.isOpened() and self.running:

            self.get_freq_data(start_timer_freq) 
             
            ret, frame = cap.read()
            if not ret: break

            # Conversion MediaPipe
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Calcul du timestamp pour le mode async
            timestamp = int((time.monotonic() - t0) * 1000)
            if timestamp <= last_timestamp:
                timestamp = last_timestamp + 1
            last_timestamp = timestamp

            # Async 
            self.tracker.landmarker.detect_async(mp_image, timestamp) 
            
            # Récolte des données dans tracker.py
            latest = self.tracker.store.latest
            face_detected = False
            if latest:
                result, _ = latest
                if result.face_landmarks:
                    face_detected = True
                    landmarks = result.face_landmarks[0]
                    
                    # Extraction des données
                    r_data, l_data = self.extractor.extract_data(landmarks, self.h, self.w)

                    # Calcul du EAR
                    r_ear = self.eye_aspect_ratio(r_data)
                    l_ear = self.eye_aspect_ratio(l_data)
                    avg_ear = (r_ear + l_ear) / 2.0

                    # Gestion de l'historique et calcul du EARM
                    self.ear_history.append(avg_ear)
                    
                    # On garde la liste à une taille maximale de 30
                    if len(self.ear_history) > self.slow_history_length:
                        self.ear_history.pop(0)
                    
                    # Si la liste est pleine, on commence les calculs
                    if len(self.ear_history) == self.slow_history_length:
                        slow_sma = sum(self.ear_history) / float(self.slow_history_length)
                        fast_sma = sum(self.ear_history[-self.fast_history_length:]) / float(self.fast_history_length)

                        earm = slow_sma - fast_sma

                        self.update_count(earm)
                        
                        color = (0, 0, 255) if self.is_blinking else (0, 255, 0)
                        cv.putText(frame, f"Clignements : {self.blink_counter}", (30, 50), 
                                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv.putText(frame, f"Seuil Fermeture: {self.earm_threshold_close:.3f}", (30, 90), 
                                   cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    else:
                        # Remplissage initial de l'historique
                        cv.putText(frame, "Initialisation de la camera...", (30, 50), 
                                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    # Dessin des points
                    for pt in r_data + l_data:
                        cv.circle(frame, pt['pixel'], 1, (0, 255, 0), -1)

            # encodage de la frame et dépôt pour lecture par le frontend
            _, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 70])
            b64 = base64.b64encode(buffer).decode('utf-8')
            with self._frame_lock:
                self.latest_frame = b64

            self.ergo_timer.update(face_detected)

        cap.release()
        
        # S'assurer que le statut de fonctionnement est mis à jour à la fermeture
        self.running = False

    # Ajout d'une méthode pour arrêter explicitement la caméra depuis le frontend
    def stop(self):
        self.running = False
