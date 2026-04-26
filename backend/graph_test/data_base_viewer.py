
import sqlite3

def print_database_data(dbname="test_data.db"):
    try:
        # 1. Connexion à la base de données
        conn = sqlite3.connect(dbname)
        cursor = conn.cursor()

        # 2. Récupération de toutes les données
        cursor.execute("SELECT * FROM blink_logs ORDER BY session_id, minute_mark")
        rows = cursor.fetchall()
        
        if not rows:
            print("\n[!] La base de données est vide.")
            return

        # 3. En-tête (Header) - Largeur ajustée à 95 pour la nouvelle colonne
        print("\n" + "="*95)
        print(f"{'ID':<4} | {'Timestamp':<20} | {'Session':<10} | {'Min':<4} | {'Blinks':<6} | {'Reliable':<8} | {'Low Freq':<8}")
        print("-" * 95)

        # 4. Affichage des lignes (Rows)
        for row in rows:
            # row[0]=id, row[1]=timestamp, row[2]=session_id, row[3]=minute_mark
            # row[4]=blink_count, row[5]=is_reliable, row[6]=low_freq
            ts = str(row[1]) if row[1] else "N/A"
            
            # On s'assure d'extraire row[5] et row[6]
            reliable = row[5] if len(row) > 5 else "N/A"
            low_f = row[6] if len(row) > 6 else "N/A"
            
            print(f"{row[0]:<4} | {ts:<20} | {row[2]:<10} | {row[3]:<4} | {row[4]:<6} | {reliable:<8} | {low_f:<8}")

        print("="*95 + "\n")
        conn.close()

    except sqlite3.OperationalError:
        print(f"\n[!] Erreur : Impossible de trouver le fichier '{dbname}' ou la table est mal structurée.")
    except Exception as e:
        print(f"\n[!] Une erreur est survenue : {e}")

if __name__ == "__main__":
    print_database_data()
