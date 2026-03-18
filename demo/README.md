# 空气质量预测系统 - Web 应用

## 快速启动

```bash
streamlit run app.py
```

应用将在浏览器自动打开：http://localhost:8501

## 功能说明

### 数据源模式
- **实时 API 模式**：从 OpenWeatherMap 和 QWeather 获取最新数据
- **历史模拟模式**：使用历史数据进行回测

### 预测模型

| 模型 | 速度 | 特点 |
|------|------|------|
| ARIMA | ⚡ 极快 | 统计基线模型 |
| Prophet | ⚡ 快 | 多季节性模型 |
| BiLSTM-Hybrid | 🐢 慢 | 高精度深度学习模型 |
| DiffSTG | 🐢🐢 很慢 | 扩散概率模型 |

**性能建议**：
- 快速预览：ARIMA + Prophet
- 高精度：BiLSTM-Hybrid（单独使用）
- 避免同时加载 BiLSTM + DiffSTG（内存不足）

### 智能预警
开启"Enable 48-Hour Pollution Alert"后，系统会自动检测未来 48 小时的污染风险并显示预警横幅。

## 数据缓存机制

### 缓存作用
系统使用 SQLite 数据库 (`local_air_cache.db`) 缓存 API 数据，主要用于：
1. **减少 API 重复请求**：避免频繁调用外部 API
2. **为 LSTM 信号分解提供长时间序列**：最多保存 1000 小时历史数据
3. **提高分解精度**：数据越长，CEEMDAN/EEMD 分解越准确，端点效应越小

### 数据库结构
```sql
air_quality (
    station TEXT,      -- 站点名（如 'Dongsi'）
    timestamp DATETIME,-- 时间戳
    pm25 REAL,         -- PM2.5 数值
    temp REAL,         -- 温度
    pres REAL,         -- 气压
    dewp REAL,         -- 露点
    wspm REAL          -- 风速
)
```

### 相关代码文件
- **`db_utils.py`** - 数据库操作（创建表、保存数据、读取数据）
- **`loader.py`** - API 数据加载，调用 `db_utils` 存储和读取
- **`predictor.py`** - LSTM 模型从数据库读取长时间序列进行信号分解

### LSTM 信号分解流程
1. 从数据库读取最近 1000 小时数据
2. 使用 CEEMDAN + EEMD 分解为 17 个 IMF 分量
3. 取每个分量的最后 12/48 个点作为 LSTM 输入
4. 17 个模型分别预测，累加得到最终结果

### 查看缓存
```bash
sqlite3 local_air_cache.db "SELECT station, datetime(timestamp, 'unixepoch', 'localtime') as time, pm25, temp FROM air_quality ORDER BY timestamp DESC LIMIT 10;"
```

## 文件说明

- `app.py` - 主应用入口
- `predictor.py` - 模型推理引擎
- `loader.py` - 实时 API 数据加载
- `loader_simulation.py` - 历史数据模拟
- `aqi_classifier.py` - AQI 等级分类
- `alert_system.py` - 智能预警系统
- `db_utils.py` - 数据库缓存管理
- `local_air_cache.db` - 本地数据缓存
