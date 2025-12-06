# 基于 ERNIE 大模型的双阶段文献检索与推荐系统

## 📖 项目简介
本项目实现了一个端到端的智能文献检索系统，旨在解决传统关键词检索面临的“语义失配”问题。系统采用工业界主流的**“双阶段检索” (Two-Stage Retrieval)** 架构：
1.  **语义召回 (Recall)**：基于 **ERNIE 3.0 双塔模型** 与 **HNSW** 向量索引技术，实现海量文献的毫秒级粗排召回。
2.  **精细排序 (Rank)**：基于 **ERNIE 3.0 单塔模型** (Cross-Encoder) 与 Pairwise 训练策略，实现对候选文档的高精度语义重排序。
3.  **可视化系统**：基于 **PyQt5** 开发了交互式桌面应用，直观展示检索效果。

## 📂 目录结构
本项目包含三个核心模块文件夹：

```text
Literature-Retrieval-System/
├── recall/                  # 语义召回模块
│   ├── base_model.py        # 双塔模型网络结构 (ERNIE 3.0)
│   ├── finetune.py          # 召回模型训练脚本 (In-batch Negatives Loss)
│   ├── recall.py            # 向量索引构建与召回预测
│   ├── ann_util.py          # HNSW 索引构建工具
│   ├── data2.py             # 召回数据处理流水线
│   └── evaluate.py          # Recall@N 评估脚本
│
├── rank/                    # 精细排序模块
│   ├── model.py             # 单塔模型网络结构 (ERNIE 3.0 + Pairwise)
│   ├── train_pairwise.py    # 排序模型训练脚本 (Margin Ranking Loss)
│   ├── predict_pairwise.py  # 排序预测脚本
│   └── data.py              # 排序数据处理流水线 (三元组/交互式拼接)
│
├── search_system/           # 系统实现模块
│   └── jiemian.py           # PyQt5 可视化检索主程序
│
└── requirements.txt         # 项目依赖环境
