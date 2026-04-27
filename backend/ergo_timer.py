import time
import os
from notifypy import Notify

# Utilisation de chemins absolus basés sur __file__ pour une fiabilité totale
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
audio_path = os.path.join(base_dir, "assets", "notif.wav")
icon_path = os.path.join(base_dir, "assets", "favicon.png")

class ErgoTimer:
    def __init__(self, db_manager=None, seuil=10):
        # Système de notifications
        self.db_manager = db_manager
        self.notifier = Notify()
        self.notifier.title = "Il est temps de prendre une pause."
        self.notifier.message = "Règle 20-20-20 : Regardez à 20 pieds pendant au moins 20 secondes, vos yeux ont besoin de repos."
        self.notifier.audio = audio_path
        self.notifier.icon = icon_path
        
        self.screen_time = 0.0
        self.absence_start = None
        self.last_update = None
        
        # Variables de session
        self.session_start = 0.0
        self.last_checked_minute = 0
        
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
            self.session_start = current_time
            return
            
        delta = current_time - self.last_update
        self.last_update = current_time

        # Verification de la frequence de clignements de la derniere minute ecoulee
        current_minute = int((current_time - self.session_start) // 60)

        if self.db_manager and current_minute > 0 and current_minute > self.last_checked_minute:
            # Délai de grâce d'une seconde pour éviter la 'race condition' avec l'écriture de la base
            if (current_time - self.session_start) % 60 > 1.0:
                self.last_checked_minute = current_minute
                last_count = self.db_manager.get_last_reliable_blink_count()
                
                if last_count is not None and last_count < self.LOW_BLINK_THRESHOLD:
                    self.notifier.title = "Fatigue visuelle"
                    self.notifier.message = f"Fréquence basse ({last_count} clign/min). Vos yeux sont fatigués."
                    self.notifier.send()

        if face_detected:
            # L'utilisateur regarde l'écran
            self.absence_start = None # On annule toute absence en cours
            self.screen_time += delta
            
            # Vérifier si on atteint les 20 minutes
            if self.screen_time >= self.BREAK_REQUIRED_SECONDS:
                self.needs_break = True
                if not self.notification_sent:
                    self.notifier.title = "Il est temps de prendre une pause."
                    self.notifier.message = "Règle 20-20-20 : Regardez à 20 pieds pendant au moins 20 secondes, vos yeux ont besoin de repos."
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
