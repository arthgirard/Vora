
import sqlite3
import random
from datetime import datetime, timedelta
from database_manager import Manager

def generate_test_db(dbname="test_data.db"):
    # Initialisation du manager (crée la table si elle n'existe pas)
    db = Manager(dbname=dbname)
    
    print(f"Génération de données de test dans {dbname}...")
    
    # Nettoyage de la table pour repartir à zéro (optionnel)
    db.cursor.execute("DELETE FROM blink_logs")
    
    # On génère des données pour les 7 derniers jours
    end_date = datetime.now()
    
    for day_offset in range(7, -1, -1):  # De il y a 7 jours jusqu'à aujourd'hui
        current_date = end_date - timedelta(days=day_offset)
        date_str = current_date.strftime("%y%m%d")
        
        # On simule entre 1 et 3 sessions par jour
        num_sessions = random.randint(1, 3)
        
        for session_num in range(1, num_sessions + 1):
            session_id = f"{date_str}{session_num:03d}"
            
            # Chaque session dure entre 10 et 45 minutes
            duration = random.randint(10, 45)
            
            # Heure de début de session aléatoire dans la journée
            start_hour = random.randint(8, 20)
            session_start_time = current_date.replace(hour=start_hour, minute=0, second=0)
            
            for minute in range(1, duration + 1):
                # Calcul du timestamp précis pour chaque log
                log_time = session_start_time + timedelta(minutes=minute)
                ts_string = log_time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Nombre de clignements aléatoire (1-30)
                blink_count = random.randint(1, 30)
                
                # 10% de chance que la donnée soit non-fiable (absence du visage)
                is_reliable = 1 if random.random() > 0.1 else 0
                
                # low_freq = 1 si clignements < 10
                low_freq = 1 if blink_count < 10 else 0
                
                # Insertion manuelle pour pouvoir contrôler les dates passées
                query = """
                    INSERT INTO blink_logs 
                    (timestamp, session_id, minute_mark, blink_count, is_reliable, low_freq) 
                    VALUES (?, ?, ?, ?, ?, ?)
                """
                db.cursor.execute(query, (
                    ts_string, 
                    session_id, 
                    minute, 
                    blink_count, 
                    is_reliable, 
                    low_freq
                ))
        
        # Commit après chaque journée
        db.connect.commit()

    print(f"Terminé ! {dbname} contient maintenant une semaine de données variées.")
    db.connect.close()

if __name__ == "__main__":
    generate_test_db()
