import sqlite3

def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # Create equipment table
    c.execute('''
    CREATE TABLE IF NOT EXISTS equipment (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        type TEXT,
        rarity TEXT,
        hp INTEGER DEFAULT 0,
        atk INTEGER DEFAULT 0,
        def INTEGER DEFAULT 0,
        crit REAL DEFAULT 0,
        atk_spd REAL DEFAULT 0,
        evasion REAL DEFAULT 0,
        additional_effects TEXT,
        image_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    print("Database initialized successfully!") 