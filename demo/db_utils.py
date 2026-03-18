# db_utils.py
import sqlite3
import pandas as pd
import os
from datetime import datetime

# Database file saved in demo directory
DB_PATH = os.path.join(os.path.dirname(__file__), 'local_air_cache.db')

def init_db():
    """初始化数据库表：如果不存在就创建"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # 创建一张表：包含站点名、时间、PM2.5、以及天气特征
    # UNIQUE(station, timestamp) 保证同一站点同一时间的数据不会重复插入
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
    将 API 获取到的最新数据存入数据库。
    使用 INSERT OR REPLACE：如果时间点已存在，就更新它；不存在，就插入新行。
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
    从数据库中捞出该站点最近的 N 条数据，喂给 LSTM 进行平滑分解
    """
    conn = sqlite3.connect(DB_PATH)
    # 按时间降序拿最新的 limit 条，然后再通过 pandas 倒序回正常的时间流
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
        
    # 因为是用 DESC 取的，最新的在最上面，需要反转回时间正序
    df = df.iloc[::-1].reset_index(drop=True)
    df['ds'] = pd.to_datetime(df['ds'])
    return df