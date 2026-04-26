import sqlite3
import uuid
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

    def get_latest_session_id(self):
        """Récupère l'ID de la dernière session enregistrée dans la DB."""
        query = "SELECT session_id FROM blink_logs ORDER BY timestamp DESC LIMIT 1"
        self.cursor.execute(query)
        result = self.cursor.fetchone()
        if result:
            return result[0]
        return self.current_session_id # Fallback si la DB est vide
        
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

    def get_histogram_query(self, scope="alltime"):
        condition = "is_reliable = 1"
        if scope == "today":
            condition += " AND date(timestamp) = date('now')"    
        return f"SELECT blink_count FROM blink_logs WHERE {condition}"

    def get_low_freq_weekly_query(self, jours):
        return f"""
            SELECT date(timestamp) as x_plot, 
                   SUM(low_freq) as y_plot 
            FROM blink_logs 
            WHERE timestamp >= date('now', '-{jours} day')
                AND is_reliable = 1
            GROUP BY date(timestamp)
            ORDER BY date(timestamp) ASC
        """
   
    def get_scatter_session_query(self):
        return """
            SELECT minute_mark as x_plot, 
                   blink_count as y_plot 
            FROM blink_logs 
            WHERE session_id = (SELECT session_id FROM blink_logs ORDER BY timestamp DESC LIMIT 1)
                AND is_reliable = 1
            ORDER BY minute_mark ASC
        """
