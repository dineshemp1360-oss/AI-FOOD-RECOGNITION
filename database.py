
import sqlite3
import os

DB_NAME = "database/food_recognition.db"

def create_database():
    os.makedirs("database", exist_ok=True)

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS food_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            food_name TEXT NOT NULL,
            quantity REAL NOT NULL,
            calories REAL NOT NULL,
            protein REAL NOT NULL,
            carbohydrates REAL NOT NULL,
            fat REAL NOT NULL,
            date_time TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

def save_food(food_name, quantity, calories, protein, carbs, fat):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO food_history
        (food_name, quantity, calories, protein,
         carbohydrates, fat, date_time)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
    """, (food_name, quantity, calories, protein, carbs, fat))

    conn.commit()
    conn.close()

def get_food_history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM food_history ORDER BY id DESC")
    data = cursor.fetchall()

    conn.close()
    return data

