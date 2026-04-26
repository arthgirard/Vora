import time
import os
from notifypy import Notify

audio_path = os.path.join("..", "assets", "notif.wav")
icon_path = os.path.join("..", "assets", "icon.png")

class ErgoTimer:
    def __init__(self, db_manager=None, seuil=10):
        # Système de notifications
        self.db_manager = db_manager
        self.notifier = Notify()
        self.notifier.title = "Il est temps de prendre une pause."
        self.notifier.message = "Regardez à 20 pieds pendant au moins 20 secondes, vos yeux ont besoin de repos."
        self.notifier.audio = audio_path
        self.notifier.icon = icon_path
        
        self.screen_time = 0.0
        self.absence_start = None
        self.last_update = None
        
        # Constantes pour la règle 20-20-20
        self.BREAK_REQUIRED_SECONDS = 20*60 # 20 minutes (1200 secondes)
        self.RESET_ABSENCE_SECONDS = 20.0 # 20 secondes d'absence pour valider la pause
        
        self.needs_break = False
        self.notification_sent = False
        
        self._low_blink_threshold = seuil # seuil : en bas de 10=yeux fatigués

    @property
    def LOW_BLINK_THRESHOLD(self):
        return self._low_blink_threshold

    @LOW_BLINK_THRESHOLD.setter
    def LOW_BLINK_THRESHOLD(self, value):
        self._low_blink_threshold = value

    def update(self, face_detected: bool):
        current_time = time.monotonic()
        
        # Initialisation lors de la première image pour éviter un décalage de temps de chargement
        if self.last_update is None:
            self.last_update = current_time
            return
            
        delta = current_time - self.last_update
        self.last_update = current_time

        if self.db_manager:
            last_count = self.db_manager.get_last_reliable_blink_count()
            if last_count is not None and last_count < self.LOW_BLINK_THRESHOLD:
                self.needs_break = True
                if not self.notification_sent:
                    self.notifier.message = f"Fréquence basse ({last_count} clign/min). Vos yeux sont fatigués."
                    self.notifier.send()
                    self.notification_sent = True

        if face_detected:
            # L'utilisateur regarde l'écran
            self.absence_start = None # On annule toute absence en cours
            self.screen_time += delta
            
            # Vérifier si on atteint les 20 minutes
            if self.screen_time >= self.BREAK_REQUIRED_SECONDS:
                self.needs_break = True
                if not self.notification_sent:
                    self.notifier.send()
                    self.notification_sent = True
        else:
            # L'utilisateur ne regarde plus l'écran
            if self.absence_start is None:
                # Début du chronomètre d'absence
                self.absence_start = current_time
            
            absence_duration = current_time - self.absence_start
            
            # Si l'absence dure 20 secondes ou plus, la pause est validée !
            if absence_duration >= self.RESET_ABSENCE_SECONDS:
                self.screen_time = 0.0
                self.needs_break = False
                self.absence_start = None
                self.notification_sent = False
