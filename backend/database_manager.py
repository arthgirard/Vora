import sqlite3
from datetime import datetime


class Manager:
    def __init__(self, dbname="data_blinks.db"):
        self.connect = sqlite3.connect(dbname, check_same_thread=False)
        self.cursor = self.connect.cursor()

        self.create_table()
        
        self.today_str = datetime.now().strftime("%y%m%d")
        # 2. Determine the next sequence number for today
        self.session_number = self.get_next_session_number(self.today_str)
        
        # 3. Format: YYYYMMDD + 3-zero padded number (e.g., 001, 002)
        self.current_session_id = f"{self.today_str}{self.session_number:03d}"

        
    def get_next_session_number(self, date_prefix):
        # Count how many unique session_ids start with today's date
        query = "SELECT COUNT(DISTINCT session_id) FROM blink_logs WHERE session_id LIKE ?"
        self.cursor.execute(query, (f"{date_prefix}%",))
        count = self.cursor.fetchone()[0]
        return count + 1

    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS blink_logs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                session_id TEXT,
                minute_mark INTEGER,
                blink_count INTEGER,
                is_reliable INTEGER DEFAULT 1,
                low_freq INTEGER DEFAULT 0
            )
        ''')
        self.connect.commit()

    def minute_log(self, minute, count, is_reliable=1, low_freq=0):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Il faut 6 points d'interrogation (timestamp, session, min, count, reliable, low_freq)
        query = """
            INSERT INTO blink_logs 
            (timestamp, session_id, minute_mark, blink_count, is_reliable, low_freq) 
            VALUES (?, ?, ?, ?, ?, ?)
        """
        self.cursor.execute(query, (now, self.current_session_id, minute, count, is_reliable, low_freq))
        self.connect.commit()

    def get_last_reliable_blink_count(self):
        """Récupère le dernier compte non-aberrant pour l'ErgoTimer."""
        query = """
            SELECT blink_count FROM blink_logs 
            WHERE is_reliable = 1 
            ORDER BY timestamp DESC LIMIT 1
        """
        self.cursor.execute(query)
        result = self.cursor.fetchone()
        return result[0] if result else None
