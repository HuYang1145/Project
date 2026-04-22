# Time Series Forecasting of Urban Air Pollution with Machine Learning

[中文说明](#中文说明) | [English](#english)

## 中文说明

### 项目简介

本仓库是 QMUL / BUPT 本科毕业设计项目 `Time Series Forecasting of Urban Air Pollution with Machine Learning` 的代码与结果公开版。项目聚焦北京市 PM2.5 浓度预测，在真实部署约束下结合时间序列建模、风险预警与交互式可视化，构建了一个端到端空气质量预测系统。

当前项目定位如下：

- `BiLSTM-Hybrid` 是当前部署设定下的主点预测模型
- `DiffSTG` 保留为不确定性分析 / 概率预测分支
- 仓库主要公开代码、实验脚本与结果摘要
- `data/` 与 `models/` 通过 Git LFS 随仓库分发，克隆后需拉取 LFS 文件

### 仓库内容

本 GitHub 仓库主要包含以下内容：

- `demo/`: Streamlit 演示系统
- `notebooks/`: 各模型的数据处理、训练与分析脚本
- `results/`: 论文相关图表、汇总结果与评估摘要
- `requirements.txt`: Python 依赖

### 数据与模型下载

本项目使用 Git LFS 管理 `data/` 与 `models/` 下的大文件。克隆仓库后，请先安装并初始化 Git LFS：

```bash
git lfs install
git lfs pull
```

如果你只想浏览代码和结果图，也可以不下载完整的 LFS 文件。

### 原始数据来源

部分原始空气质量数据来源于公开网站：

- Quotsoft Beijing Air Pollution Historical Data: <https://quotsoft.net/air/>

如果原始数据已经可从公开来源稳定获取，建议优先使用原始来源；本仓库中的 `data/` 主要保存项目复现与演示所需的数据文件。

### 环境配置

建议使用项目原始环境：

```bash
conda activate project
pip install -r requirements.txt
```

### 运行演示系统

```bash
cd demo
streamlit run app.py
```

系统支持：

- 实时 API 模式
- 历史仿真模式
- 多模型切换预测
- PM2.5 预测曲线可视化
- 48 小时污染预警

### 复现实验结果

生成结果汇总：

```bash
python results/summarize_results.py
```

运行缓存检查：

```bash
python demo/test_cache.py
```

### 说明

- 本仓库为公开展示与复现导向版本，不包含全部原始数据、缓存数据库和训练中间文件
- `BiLSTM-Hybrid` 与 `DiffSTG` 资源占用较高，日常演示时不建议同时加载
- 若仓库中的结果图、指标摘要与论文正文存在差异，应以最终提交论文版本为准

### 引用

如果你复用了本项目的代码、图表或结果，请注明来源，并说明是否进行了修改。

---

## English

### Overview

This repository is the public code-and-results release of the QMUL / BUPT undergraduate final-year project `Time Series Forecasting of Urban Air Pollution with Machine Learning`. The project focuses on Beijing PM2.5 forecasting under practical deployment constraints and builds an end-to-end system covering forecasting, alerting, and interactive visualization.

The current project positioning is:

- `BiLSTM-Hybrid` is the main point-forecasting model in the deployed setting
- `DiffSTG` is retained as the uncertainty-analysis / probabilistic forecasting branch
- This GitHub repository mainly publishes code, experiment scripts, and result summaries
- `data/` and `models/` are distributed with the repository through Git LFS

### Repository Contents

This GitHub repository mainly includes:

- `demo/`: the runnable Streamlit demo system
- `notebooks/`: preprocessing, training, and analysis scripts for different models
- `results/`: thesis-related figures, summaries, and evaluation outputs
- `requirements.txt`: Python dependencies

### Data and Model Downloads

This project uses Git LFS for large files under `data/` and `models/`. After cloning the repository, install and pull LFS objects:

```bash
git lfs install
git lfs pull
```

If you only want to inspect the code and result figures, you do not need to download the full LFS assets.

### Raw Data Source

Part of the raw air-quality data comes from the following public source:

- Quotsoft Beijing Air Pollution Historical Data: <https://quotsoft.net/air/>

If the raw data can be stably obtained from the original public source, it is better to use that source directly. The repository `data/` directory mainly keeps the files needed for reproduction and demo usage.

### Environment Setup

Recommended environment:

```bash
conda activate project
pip install -r requirements.txt
```

### Running the Demo

```bash
cd demo
streamlit run app.py
```

The system supports:

- real-time API mode
- historical simulation mode
- multi-model forecasting
- PM2.5 forecast visualization
- 48-hour pollution alerting

### Reproducing Key Outputs

Generate the summary report:

```bash
python results/summarize_results.py
```

Run the cache check:

```bash
python demo/test_cache.py
```

### Notes

- This is a public-facing repository for presentation and reproducibility; it does not include all raw data, local caches, or intermediate training artifacts
- `BiLSTM-Hybrid` and `DiffSTG` are both resource-heavy and should not normally be loaded together for routine demo usage
- If any discrepancy exists between this repository and the final submitted thesis, the final thesis version should be treated as authoritative

### Citation

If you reuse the code, figures, or results from this project, please cite the project appropriately and indicate any modifications.
