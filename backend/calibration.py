import cv2 as cv
import numpy as np

class EyeCalibration:
    def __init__(self, frames_noise=100, frames_blinks=150):
        self.frames_noise = frames_noise
        self.frames_blinks = frames_blinks
        
        # États de  machine
        self.STATE_IDLE = 0
        self.STATE_MEASURE_NOISE = 1 
        self.STATE_MEASURE_BLINKS = 2
        self.STATE_FINISHED = 3
        
        self.current_state = self.STATE_IDLE
        
        # Stockage des données
        self.noise_measurements = []
        self.blink_measurements = []
        
        # Seuils finaux calculés
        self.threshold_close = None
        self.threshold_open = None
        
        # Couleurs UI
        self.YELLOW = (0, 255, 255)
        self.GREEN = (86, 241, 13)
        self.BLUE = (252, 41, 3)
        self.PURPLE = (255, 0, 255)

    def is_complete(self):
        return self.current_state == self.STATE_FINISHED

    def get_thresholds(self):
        return self.threshold_close, self.threshold_open

    def _draw_ui(self, frame, text, subtext, color):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv.rectangle(overlay, (0, h//2 - 40), (w, h//2 + 40), (0, 0, 0), -1)
        cv.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        (tw, th), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv.putText(frame, text, ((w - tw) // 2, h // 2 - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        if subtext:
            (stw, sth), _ = cv.getTextSize(subtext, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv.putText(frame, subtext, ((w - stw) // 2, h // 2 + 25),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def process(self, frame, current_earm, key_pressed):
        
        if self.current_state == self.STATE_IDLE:
            self._draw_ui(frame, "CALIBRATION UNIVERSELLE", "Appuyez sur 'c' pour démarrer", self.YELLOW)
            if key_pressed == ord('c'):
                self.current_state = self.STATE_MEASURE_NOISE 

        elif self.current_state == self.STATE_MEASURE_NOISE: 
            self.noise_measurements.append(current_earm)
            prog = len(self.noise_measurements)
            
            self._draw_ui(frame, "REGARDEZ LES 4 COINS DE L'ECRAN", 
                          f"Mesure mouvements : {prog}/{self.frames_noise} (Ne clignez pas)", self.BLUE)
            
            if prog >= self.frames_noise:
                # Le seuil d'ouverture doit être juste au-dessus du bruit maximal généré par le mouvement
                max_noise = np.percentile(self.noise_measurements, 98)
                self.threshold_open = max(0.02, max_noise + 0.01)
                self.current_state = self.STATE_MEASURE_BLINKS

        elif self.current_state == self.STATE_MEASURE_BLINKS:
            self.blink_measurements.append(current_earm)
            prog = len(self.blink_measurements)
            
            self._draw_ui(frame, "CLIGNEZ DES YEUX 3 FOIS", 
                          f"Mesure amplitude : {prog}/{self.frames_blinks}", self.PURPLE)
            
            if prog >= self.frames_blinks:
                valid_blinks = [val for val in self.blink_measurements if val > self.threshold_open * 2]
                
                if not valid_blinks:
                    print("[ERREUR] Aucun clignement franc détecté. Valeurs par défaut appliquées.")
                    self.threshold_close = self.threshold_open + 0.10
                else:
                    avg_blink_peak = np.percentile(valid_blinks, 75)
                    
                    # Le seuil de fermeture se place entre le bruit de mouvement et le pic du clignement
                    self.threshold_close = self.threshold_open + 0.04 # 0.04 de marge

                    # Si l'utilisateur fait de très petits clignements pendant la calibration, on s'assure que le seuil ne dépasse pas la moitié de son clignement
                    if self.threshold_close > (avg_blink_peak * 0.5):
                        self.threshold_close = avg_blink_peak * 0.5
                
                print(f"Bruit max (Mouvement) : {np.percentile(self.noise_measurements, 98):.4f}")
                print(f"Pic moyen (Clignement) : {np.median(np.sort(valid_blinks)[-10:]) if valid_blinks else 'N/A'}")
                print(f"Seuil Ouverture (Hystérésis bas) : {self.threshold_open:.4f}")
                print(f"Seuil Fermeture (Hystérésis haut) : {self.threshold_close:.4f}")
                
                self.current_state = self.STATE_FINISHED
                return True 

        return False
