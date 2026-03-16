import os
import cv2 as cv
import numpy as np

class EyeCalibration:
    def __init__(self, frames_to_capture=100): # <-- Allongé à 100
        self.frames_to_capture = frames_to_capture
        
        # États
        self.STATE_IDLE = 0
        self.STATE_MEASURE_NOISE = 1 
        self.STATE_FINISHED = 2
        
        self.current_state = self.STATE_IDLE
        
        # Données
        self.earm_measurements = []
        self.computed_threshold = None
        
        # Couleurs UI
        self.YELLOW = (0, 255, 255)
        self.GREEN = (86, 241, 13)
        self.BLUE = (252, 41, 3)

    def is_complete(self):
        return self.current_state == self.STATE_FINISHED

    def get_threshold(self):
        return self.computed_threshold

    def _draw_ui(self, frame, text, subtext, color):
        """Helper pour dessiner l'interface"""
        h, w = frame.shape[:2]
        
        # Fond beaucoup plus transparent et moins large
        overlay = frame.copy()
        cv.rectangle(overlay, (0, h//2 - 40), (w, h//2 + 40), (0, 0, 0), -1)
        cv.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Texte principal
        (tw, th), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv.putText(frame, text, ((w - tw) // 2, h // 2 - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Sous-texte
        if subtext:
            (stw, sth), _ = cv.getTextSize(subtext, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv.putText(frame, subtext, ((w - stw) // 2, h // 2 + 25),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    def process(self, frame, current_earm, key_pressed):
        
        if self.current_state == self.STATE_IDLE:
            # nouvelle méthode pour le earm
            self._draw_ui(frame, "CALIBRATION DES ANGLES", "Appuyez sur 'c' pour commencer", self.YELLOW)
            if key_pressed == ord('c'):
                self.earm_measurements = [] 
                self.current_state = self.STATE_MEASURE_NOISE 

        elif self.current_state == self.STATE_MEASURE_NOISE: 
            self.earm_measurements.append(current_earm)
            prog = len(self.earm_measurements)
            
            self._draw_ui(frame, "REGARDEZ FIXEMENT LA CAMERA", f"Mesure : {prog}/{self.frames_to_capture} (Ne clignez pas)", self.BLUE)
            
            if len(self.earm_measurements) >= self.frames_to_capture:
                # calcul des angles
                max_angle_noise = np.percentile(self.earm_measurements, 95)
                
                base_noise = max(0.01, max_angle_noise)
                self.computed_threshold = base_noise + 0.005 # il faut majorer pour éviter d'être pile dessus 
                
                print(f"Calibration terminée. Bruit extrême : {base_noise:.4f}")
                print(f"Nouveau seuil EARM calculé : {self.computed_threshold:.4f}")
                
                self.current_state = self.STATE_FINISHED
                return True 

        return False
