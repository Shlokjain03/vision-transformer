import sqlite3

def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(image_path, predicted_class, confidence):
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('INSERT INTO predictions (image_path, predicted_class, confidence) VALUES (?, ?, ?)', 
              (image_path, predicted_class, confidence))
    conn.commit()
    conn.close()

def get_all_predictions():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('SELECT * FROM predictions ORDER BY id DESC')
    results = c.fetchall()
    conn.close()
    return results
