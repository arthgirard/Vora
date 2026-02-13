import cv2 as cv
import numpy as np

class EyeCalibration:
    def __init__(self, frames_to_capture=50):
        self.frames_to_capture = frames_to_capture
        
        # États
        self.STATE_IDLE = 0
        self.STATE_OPEN = 1
        self.STATE_WAIT = 2
        self.STATE_CLOSED = 3
        self.STATE_FINISHED = 4
        
        self.current_state = self.STATE_IDLE
        
        # Données
        self.open_ears = []
        self.closed_ears = []
        self.computed_threshold = None
        
        # Compteurs pour les délais
        self.wait_counter = 0
        self.WAIT_FRAMES = 40 

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
        
        # Fond sombre semi-transparent pour lisibilité
        overlay = frame.copy()
        cv.rectangle(overlay, (0, h//2 - 50), (w, h//2 + 50), (0, 0, 0), -1)
        cv.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Texte principal
        (tw, th), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv.putText(frame, text, ((w - tw) // 2, h // 2 - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Sous-texte
        if subtext:
            (stw, sth), _ = cv.getTextSize(subtext, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv.putText(frame, subtext, ((w - stw) // 2, h // 2 + 35),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    def process(self, frame, current_ear, key_pressed):
        """
        Fonction principale appelée à chaque frame.
        Retourne True si la calibration vient de se terminer.
        """
        
        if self.current_state == self.STATE_IDLE:
            self._draw_ui(frame, "CALIBRATION REQUISE", "Appuyez sur 'c' pour commencer", self.YELLOW)
            if key_pressed == ord('c'):
                self.open_ears = [] # Reset
                self.current_state = self.STATE_OPEN

        elif self.current_state == self.STATE_OPEN:
            self.open_ears.append(current_ear)
            prog = len(self.open_ears)
            self._draw_ui(frame, "GARDEZ LES YEUX OUVERTS", f"Mesure : {prog}/{self.frames_to_capture}", self.GREEN)
            
            if len(self.open_ears) >= self.frames_to_capture:
                self.current_state = self.STATE_WAIT
                self.wait_counter = 0

        elif self.current_state == self.STATE_WAIT:
            self.wait_counter += 1
            # On demande à l'utilisateur de fermer les yeux, mais on n'enregistre pas encore
            self._draw_ui(frame, "FERMEZ LES YEUX MAINTENANT !", "Preparation...", self.BLUE)
            
            if self.wait_counter >= self.WAIT_FRAMES:
                self.closed_ears = [] # Reset
                self.current_state = self.STATE_CLOSED

        elif self.current_state == self.STATE_CLOSED:
            self.closed_ears.append(current_ear)
            prog = len(self.closed_ears)
            self._draw_ui(frame, "MAINTENEZ LES YEUX FERMES", f"Mesure : {prog}/{self.frames_to_capture}", self.BLUE)

            if len(self.closed_ears) >= self.frames_to_capture:
                # CALCUL FINAL
                avg_open = np.mean(self.open_ears)
                avg_closed = np.mean(self.closed_ears)
                self.computed_threshold = (avg_open + avg_closed) / 2.0
                
                print(f"Calibration terminée. Seuil calculé : {self.computed_threshold:.4f}")
                self.current_state = self.STATE_FINISHED
                return True # Indique que c'est fini

        return False