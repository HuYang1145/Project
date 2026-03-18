# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

---

## 🖥️ 系统环境（固定信息）

### 硬件与操作系统
- **操作系统**: Windows 11 Home China (Build 10.0.26200)
- **Shell**: Bash (Git Bash) - 必须使用 Unix 语法（如 `/dev/null` 而非 `NUL`）
- **用户**: Huyang
- **项目路径**: `D:\文件\大四下\Project`
- **GPU**: 可选，用于加速 LSTM 和 Diffusion 模型推理

### Python 环境
- **Conda 环境**: `project`（必须在此环境中运行）
- **包管理器**: pip（禁止使用 pipenv）
- **VSCode 配置**: 已禁用终端自动激活 conda 环境

### 个人偏好与规范
- **代码风格**:
  - 中文注释和文档字符串
  - 函数命名: snake_case
  - 类命名: PascalCase
  - 常量: UPPER_CASE
- **代码修改流程**:
  1. 先展示修改计划（Diff）
  2. 等待用户明确回复"确认修改"
  3. 禁止未经允许直接覆盖代码
- **训练脚本要求**:
  - 必须生成可视化结果（用于论文）
  - 图表保存为高分辨率 PNG（dpi=300）
  - 评估指标保存为 CSV（UTF-8 编码）

---

## 📋 项目概述

这是一个**空气质量预测系统**，集成了四种预测模型：
- **ARIMA**: 传统时间序列统计模型（最快）
- **Prophet**: Facebook 多变量时间序列模型（快）
- **BiLSTM-Hybrid**: CEEMDAN/EEMD分解 + 双向LSTM + LEC局部误差校正（慢）
- **DiffSTG**: 基于扩散概率模型的生成式 AI 预测（最慢）

主应用通过 Streamlit 提供 Web 界面，支持实时 API 模式和历史数据模拟模式。

### 性能特性
| 模型 | 预测速度 | 内存占用 | 推荐场景 |
|------|---------|---------|---------|
| ARIMA | ⚡ 极快 | 最小 | 快速预览、基线对比 |
| Prophet | ⚡ 快 | 小 | 趋势分析、稳定预测 |
| BiLSTM-Hybrid | 🐢 慢 | 大 | 高精度预测、48h预警 |
| DiffSTG | 🐢🐢 很慢 | 最大 | 不确定性量化 |

**⚠️ 重要**: BiLSTM 和 DiffSTG 同时加载会导致内存不足，建议单独使用或与轻量模型组合。

---

## 🎯 核心功能

### 1. Regression-to-Classification 后处理系统 ✅
- **核心模块**: `aqi_classifier.py`
- **策略**: 保留数值预测能力，通过阈值映射实现 AQI 6级分类
- **预测时长**: 7天（168小时）
- **集成方式**: `predictor.predict_aqi_classification()` 接口
- **UI展示**: 主导等级 + 每日分布柱状图 + 健康建议

### 2. 智能雾霾预警系统 ✅
- **核心模块**: `alert_system.py`
- **触发条件**: PM2.5 增加 > 30 µg/m³ 且达到 75+ (轻度污染)
- **预警等级**: 轻度🟡 / 中度🟠 / 重度🔴 / 严重🟣
- **显示位置**: 预测图表上方彩色横幅

### 3. AQI 等级定义（6类）
```python
0: Excellent (优) - PM2.5 ≤ 35
1: Good (良) - 35 < PM2.5 ≤ 75
2: Lightly Polluted (轻度污染) - 75 < PM2.5 ≤ 115
3: Moderately Polluted (中度污染) - 115 < PM2.5 ≤ 150
4: Heavily Polluted (重度污染) - 150 < PM2.5 ≤ 250
5: Severely Polluted (严重污染) - PM2.5 > 250
```

---

## 🚀 核心命令

### 启动应用
```bash
# 方式1: 从根目录启动
streamlit run app.py

# 方式2: 从 demo 文件夹启动
cd demo
streamlit run app.py
```

### 环境管理
```bash
# 激活环境
conda activate project

# 安装依赖
pip install -r requirements.txt
```

### 数据库查看
```bash
# 查看缓存内容
sqlite3 local_air_cache.db "SELECT station, datetime(timestamp, 'unixepoch', 'localtime') as time, pm25, temp FROM air_quality ORDER BY timestamp DESC LIMIT 20;"

# 查看总记录数
sqlite3 local_air_cache.db "SELECT COUNT(*) FROM air_quality;"
```

---

## 📁 项目完整结构

### 根目录文件
```
├── app.py                      # Streamlit 主应用
├── predictor.py                # 四大模型推理引擎
├── loader.py                   # 实时 API 数据加载
├── loader_simulation.py        # 历史数据模拟
├── db_utils.py                 # SQLite 缓存管理
├── aqi_classifier.py           # AQI 分类后处理
├── alert_system.py             # 智能预警系统
├── summarize_results.py        # 结果汇总脚本
├── local_air_cache.db          # SQLite 缓存数据库
├── CLAUDE.md                   # 本文档
├── README.md                   # 项目说明
├── TODO.md                     # 任务清单
└── requirements.txt            # 依赖列表
```

### demo/ 文件夹（应用副本）
```
demo/
├── app.py                      # 应用副本
├── predictor.py                # 推理引擎副本
├── loader.py                   # 数据加载副本
├── loader_simulation.py        # 模拟加载副本
├── aqi_classifier.py           # 分类器副本
├── alert_system.py             # 预警系统副本
├── db_utils.py                 # 数据库工具副本
├── local_air_cache.db          # 缓存数据库副本
└── README.md                   # Demo 说明文档
```

**注意**: demo/ 文件夹包含根目录文件的副本，方便独立运行。

### models/ 目录结构

**models/arima/**
- `Aotizhongxin_ARIMA.pkl` (319MB)

**models/prophet/**
- 12 个站点模型: `{Station}_prophet.joblib`
- 3 个分类模型: `prophet_classification_{3|5|7}days.joblib` (已废弃)

**models/lstm1/** (12小时预测)
- 17 个 CEEMDAN IMF 分量: `CEEMDAN_IMF_{3-19}.pth`
- 17 个 RLMD 子分量: `RLMD_Sub_IMF_{3-19}.pth`
- LEC 误差校正: `LEC_Model.pth`
- 3 个分类模型: `lstm_classification_{3|5|7}days.pth` (已废弃)
- 配置文件: `config.pkl`, `scaler_Dongsi.pkl`

**models/lstm_48h/** (48小时预测 - 新增 2026-03-12)
- 17 个 CEEMDAN IMF 分量: `CEEMDAN_IMF_{3-19}.pth`
- 17 个 RLMD 子分量: `RLMD_Sub_IMF_{3-19}.pth`
- LEC 误差校正: `LEC_Model.pth`
- 配置文件: `config.pkl`, `scaler_Dongsi.pkl`
- ⚠️ **缺失**: 48小时分类模型

**models/diffusion/**
- `checkpoints/`: DiffSTG 模型检查点
- `results/`: 预测结果文件
- `logs/`: 训练日志

### data/ 目录结构
```
data/
├── raw/                        # 原始数据
├── processed/
│   ├── arima/                  # 14 个站点 PM2.5 CSV
│   ├── lstm/                   # 多站点 PM2.5 CSV
│   └── diffusion/AIR_BJ/       # 图数据
└── dataset/AIR_BJ/
    ├── adj.npy                 # 图邻接矩阵
    └── flow.npy                # 流量数据
```

### notebooks/ 目录结构
```
notebooks/
├── arima/
│   ├── train_evaluate_arima.ipynb
│   ├── build_arima_model.ipynb
│   └── preprocess_arima.ipynb
├── prophet/
│   ├── train.ipynb
│   ├── analysis_Prophet.ipynb
│   └── preprocess_prophet.ipynb
├── lstm/
│   ├── train_lstm_48h.py       # ⭐ 48小时回归训练
│   ├── train_lstm_7day.py      # 7天预测训练
│   └── train.ipynb
└── diffusion/
    ├── train_diffusion.ipynb
    ├── preprocess_diffusion.ipynb
    └── algorithm/diffstg/      # 核心模块
        ├── model.py
        ├── ugnet.py
        └── graph_algo.py
```

### results/ 目录结构
```
results/
├── evaluation_summary.txt      # 所有模型评估汇总
├── figures_index.txt           # 图表索引
└── figures_summary/            # 整理后的图表
    ├── lstm_classification_*_confusion.png
    ├── lstm_classification_*_curves.png
    └── prophet_classification_*.png
```

---

## 🔧 技术架构

### 数据流
1. **实时模式**: API → `loader.py` → `db_utils.py` (缓存) → `predictor.py` → Streamlit UI
2. **模拟模式**: 历史数据 → `loader_simulation.py` → `predictor.py` → Streamlit UI

### BiLSTM-Hybrid 架构
- **信号分解**: CEEMDAN (10 trials) + EEMD (5 trials) 联合分解
- **噪声过滤**: 基于排列熵 (PE > 0.90) 过滤高频噪声
- **深度学习**: 双向 LSTM (Hidden Size: 64/96)
- **误差校正**: LEC 局部误差校正 (残差 ≥ 8.0/12.0 时触发)
- **分量数量**: 17 个 IMF 分量 (IMF3-IMF19)

### DiffSTG 架构
- **扩散过程**: 基于概率生成的时空预测
- **不确定性量化**: 提供 90% 置信区间
- **特殊依赖**: 需要将 `notebooks/diffusion/` 添加到 `sys.path`

### 数据库缓存策略
- **数据库**: SQLite (`local_air_cache.db`)
- **表结构**: `air_quality(station, timestamp, pm25, temp, pres, dewp, wspm)`
- **去重机制**: `UNIQUE(station, timestamp)`
- **缓存时长**: 30 分钟自动刷新
- **存储容量**: 最多 1000 小时历史数据

---

## ⚙️ 开发注意事项

### API 配置
- API 密钥硬编码在 `loader.py` 中 (OpenWeatherMap + QWeather)
- 生产环境应迁移到环境变量或 `.env` 文件

### 模型路径配置
- 所有模型路径在 `predictor.py` 的 `MODELS_CONFIG` 中定义
- 基于 `BASE_DIR` 动态构建，支持根目录和 demo 文件夹

### Windows 特定处理
- 使用 Bash (Git Bash) 而非 CMD
- GPU 设置: `os.environ["CUDA_VISIBLE_DEVICES"] = "0"`
- 路径使用 `os.path.join()` 确保兼容性

### 关键依赖
- **数据处理**: pandas, numpy, scipy, scikit-learn
- **信号分解**: EMD-signal, vmdpy, antropy
- **深度学习**: PyTorch (conda 安装), transformers
- **时间序列**: prophet, statsmodels
- **可视化**: streamlit, plotly, matplotlib
- **数据库**: sqlite3 (内置)

---

## 📊 项目状态总结

### ✅ 已完成功能
1. **四大预测模型**: ARIMA, Prophet, BiLSTM-Hybrid, DiffSTG
2. **48小时回归预测**: BiLSTM-Hybrid 48h 模型
3. **Regression-to-Classification**: AQI 6级分类后处理
4. **智能雾霾预警**: 基于 PM2.5 突增检测
5. **双模式数据源**: 实时 API + 历史模拟
6. **SQLite 缓存**: 减少 API 重复请求
7. **Streamlit Web UI**: 交互式可视化界面

### ⚠️ 已知限制
1. **内存占用**: BiLSTM + DiffSTG 同时加载会导致内存不足
2. **48h 分类模型缺失**: 仅有回归模型，无分类模型
3. **7天分类模型已废弃**: 准确率仅 35-37%，效果不佳

### ❌ 计划书未完成项
1. CNN-LSTM 模型
2. Attention-based RNN
3. Prophet-Hybrid 模型
4. 预测不确定性区间 UI 显示
5. 24-48小时自动超标警报系统

---

## 📝 会话维护规则

- 每次会话结束前更新 `TODO.md`
- 记录未完成任务和重要逻辑变更
- 更新 `CLAUDE.md` 中的新规则或约定
- 保持文档与代码同步

---

## 📖 毕业论文写作规范（LaTeX）

### 论文结构
- **位置**: `毕业论文/` 文件夹
- **主文件**: `main.tex`
- **章节文件**: `contents/` 目录下的独立 `.tex` 文件
- **编译命令**: `xelatex main.tex`（需运行两次生成完整目录和引用）

### 学术写作六大铁律（必须严格遵守）

1. **严禁碎片化与项目符号**
   - 绝对不允许使用无意义的 Bullet points（`\item`）罗列属性
   - 所有定义必须融入连贯、严谨的学术段落
   - 注重句子间的起承转合

2. **严禁直接粘贴工程代码**
   - 正文中不能出现 Python/Pandas 代码
   - 使用学术语言描述算法逻辑
   - 核心算法用 LaTeX 伪代码（algorithm2e 宏包）

3. **剔除所有纯工程流水账**
   - 不提及前端 UI、SQLite 缓存、系统 Bug 调试
   - 只讨论数学模型、神经网络架构、算法机制

4. **保持绝对客观中立的学术口吻**
   - 严禁使用"突破性"、"质的飞跃"、"极大地"等主观词汇
   - 客观描述模型的数学特性及其解决的问题
   - 例如："引入双向机制使得模型能够捕捉全局时序依赖"

5. **建立严谨的层级深度**
   - 章节标题：`\section{}`（仅标题英文，正文中文）
   - 子章节：`\subsection{}`、`\subsubsection{}`
   - 构建清晰的递进逻辑

6. **数学公式必须深度语境化**
   - 使用 `\begin{equation}` 环境
   - 公式前后详细解释每个符号的物理/数学意义
   - 阐述公式为何适用于解决 PM2.5 预测问题
   - 严禁干巴巴地罗列公式

### 论文语言规范（严格遵守）
- **章节标题**: 必须使用英文（如 `\chapter{Introduction}`, `\section{Methodology}`）
- **正文内容**: 必须全部使用中文（包括段落、列表、表格说明等）
- **摘要和关键词**: 必须使用中文内容（标题 Abstract/Keywords 是英文，但内容是中文）
- **数学符号**: 使用标准 LaTeX 数学符号
- **变量命名**: 遵循数学惯例（如 $x_t$ 表示时刻 $t$ 的观测值）
- **图表标题**: 使用英文（caption）

### 禁止事项
- ❌ 不使用 emoji 表情符号
- ❌ 不编造实验数据（所有数据必须来自真实实验结果）
- ❌ 不使用 `\chapter{}`（article 文档类不支持，使用 `\section{}`）
- ❌ 不创建冗余的工程文档或 README

---

**最后更新**: 2026-03-16



