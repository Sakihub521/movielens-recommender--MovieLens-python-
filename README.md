# 基于 MovieLens 数据集的推荐系统（协同过滤）

一个用于学习推荐系统的入门项目：在 [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/) 数据集上，从零实现记忆型协同过滤（用户-用户 / 物品-物品），并在测试集上用 RMSE 评估效果。

> 本项目为学习笔记，主体代码基于教程《从零开始用 Python 搭建推荐引擎》（原作者 Pulkit Sharma，中文翻译版），在此基础上修复了原教程的 bug 并补充了评估环节。

## 实现内容

- 数据加载：MovieLens 100k（943 用户、1682 电影、10 万条评分）
- 构建用户-电影评分矩阵
- 计算余弦相似度（用户相似度 / 物品相似度）
- 两种协同过滤预测：
  - 用户协同过滤（user-based）
  - 物品协同过滤（item-based）
- 在测试集 `ua.test`（9430 条）上计算 RMSE 评估

## 实验结果

| 方法 | RMSE（越小越好） |
| --- | --- |
| 基线（预测全局均值） | 1.122 |
| 用户协同过滤 | 0.972 |
| 物品协同过滤 | 1.023 |

## 相比原教程的改进

原教程（2018 年）的代码存在两个会导致结果严重偏差的问题，本项目已修复：

1. **`n_items` 用 `unique().shape[0]` 统计数量** —— 电影 ID 不连续时（训练集缺 2 部电影），矩阵开小导致越界报错 → 改为 `.max()` 取最大 ID。
2. **`ratings.mean(axis=1)` 把「没打分=0」也算进分母** —— 用户平均分被稀释约 6 倍，预测整体塌陷 → 改为只对「打过分」的电影求平均。

另外补充了原教程缺失的 **RMSE 评估** 环节，使得模型可以在测试集上验证效果。

## 运行环境与依赖

- Python 3.x
- pandas、numpy、scikit-learn

```bash
pip install pandas numpy scikit-learn
```

## 运行

```bash
python 基于MovieLens数据集的python实例学习.py
```

## 数据集

数据集来自 GroupLens（明尼苏达大学），未包含在仓库中。下载地址：

https://grouplens.org/datasets/movielens/100k/

下载后解压，将数据文件放到 `ml-100k/` 目录下即可。

## 目录结构

```
.
├── 基于MovieLens数据集的python实例学习.py   # 主程序
├── ml-100k/                                # 数据集（需自行下载）
├── README.md
└── .gitignore
```
