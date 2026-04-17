import os
import sqlite3

import pandas as pd

# Database file saved in the demo directory
DB_PATH = os.path.join(os.path.dirname(__file__), "local_air_cache.db")

def init_db():
    """Create the database table if it does not already exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Store station name, timestamp, PM2.5, and weather features.
    # UNIQUE(station, timestamp) prevents duplicate inserts for the same point in time.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS air_quality (
            station TEXT,
            timestamp DATETIME,
            pm25 REAL,
            temp REAL,
            pres REAL,
            dewp REAL,
            wspm REAL,
            UNIQUE(station, timestamp)
        )
    ''')
    conn.commit()
    conn.close()

def save_realtime_data(station, timestamp, pm25, temp=0, pres=0, dewp=0, wspm=0):
    """
    Save the latest API data to the database.

    INSERT OR REPLACE updates an existing timestamp if present,
    otherwise it inserts a new row.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO air_quality (station, timestamp, pm25, temp, pres, dewp, wspm)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (station, timestamp, pm25, temp, pres, dewp, wspm))
    conn.commit()
    conn.close()

def get_recent_data(station, limit=1000):
    """
    Load the latest N samples for a station and feed them into the LSTM pipeline.
    """
    conn = sqlite3.connect(DB_PATH)
    # Read the newest rows first, then reverse them back to chronological order.
    query = f'''
        SELECT timestamp as ds, pm25 as y, temp as TEMP, pres as PRES, dewp as DEWP, wspm as WSPM
        FROM air_quality 
        WHERE station = '{station}'
        ORDER BY timestamp DESC 
        LIMIT {limit}
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        return df
        
    # The DESC query returns newest first, so reverse the frame into time order.
    df = df.iloc[::-1].reset_index(drop=True)
    df['ds'] = pd.to_datetime(df['ds'])
    return df
