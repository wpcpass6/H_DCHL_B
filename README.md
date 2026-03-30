# H-DCHL-B

基于 DCHL 风格实现的异构超图下一 POI 推荐实验工程。

当前工程目标不是复刻原论文代码结构，而是在尽量保持 DCHL 稀疏矩阵实现风格的前提下，引入异构节点与掩码训练机制，验证结构增益与辅助任务的作用。


## 1. 当前模型包含什么

当前版本使用以下四类节点/结构：

- `User`：用户节点，用于建模长期协同偏好。
- `POI`：兴趣点节点，是最终推荐对象。
- `Category`：POI 类别节点，用于建模功能语义。
- `Region`：POI 区域节点，默认通过 geohash 动态划分，用于建模空间语义。

当前四条主分支：

- `User-POI` 协同分支
- `POI->POI` 有向转移分支
- `POI-Category` 异构语义分支
- `POI-Region` 异构语义分支

当前可选的掩码任务：

- `Category mask`
- `Region mask`
- `POI mask`

说明：

- `Category/Region mask` 的作用是强制模型学好异构节点表示。
- `POI mask` 的作用不是重复学习 POI 本身，而是强迫模型利用 `user / trans / category / region` 四类结构去恢复被遮蔽的 POI，从而间接增强异构信息利用。


## 2. 当前代码入口说明

核心文件：

- `preprocess.py`：从 TSMC2014 原始文本生成训练所需数据。
- `dataset.py`：构造数据集对象及超图/异构关系矩阵。
- `model.py`：H-DCHL-B 主模型。
- `train.py`：训练入口。
- `utils.py`：稀疏矩阵、geohash、文件读写等工具。
- `metrics.py`：Recall/NDCG 指标计算。


## 3. 数据准备

当前默认目录约定：

- 原始数据：`datasets/dataset_TSMC2014_TKY.txt`
- 预处理输出：`datasets/TKY`

执行预处理：

```bash
python preprocess.py --raw_path datasets/dataset_TSMC2014_TKY.txt --output_dir datasets/TKY
```

预处理输出主要文件：

- `datasets/TKY/train_poi_zero.pkl`
- `datasets/TKY/test_poi_zero.pkl`
- `datasets/TKY/poi_coos.pkl`
- `datasets/TKY/poi_category.pkl`
- `datasets/TKY/meta.pkl`

说明：

- `Region` 默认不强依赖预处理好的 `poi_region.pkl`；
- 若训练时不传 `--poi_region_path`，则会根据 `poi_coos.pkl` 按 `--region_precision` 动态构造 region。


## 4. 结果输出说明

每次训练都会在 `save_dir/时间戳目录/` 下生成：

- `log_training.txt`：完整训练日志
- `result.txt`：简要结果汇总
- `args.json`：本次运行参数

`result.txt` 会记录两套结果：

1. `best-by-metric`
   - 每个指标在所有 epoch 中分别取最大值
   - 与原始 DCHL 代码风格一致

2. `best-Rec10-row`
   - 按 `Rec10` 最优所在 epoch 取该轮整行结果
   - 更严格，推荐优先用于方案比较


## 5. 当前默认训练配置

`train.py` 里当前默认值已经设为当前阶段最优主线：

- `data_dir=datasets/TKY`
- `meta_path=datasets/TKY/meta.pkl`
- `region_precision=5`
- `seed=2026`
- `num_epochs=50`
- `batch_size=200`
- `emb_dim=128`
- `lr=0.001`
- `decay=0.0005`
- `dropout=0.3`
- `keep_rate=1.0`
- `keep_rate_poi=1.0`
- `num_col_layers=2`
- `num_reg_layers=2`
- `num_cat_layers=1`
- `num_trans_layers=4`
- `mask_rate_cat=0.2`
- `lambda_cat=0.05`
- `mask_rate_reg=0.2`
- `lambda_reg=0.02`
- `mask_rate_poi=0.1`
- `lambda_poi=0.02`
- `lambda_cat_cls=0.0`


## 6. 常用执行命令

### 6.1 当前默认最佳版本（推荐直接跑）

```bash
python train.py
```

这条命令默认等价于：

- `cat + reg + poi mask`
- `region_precision=5`
- 不启用 `category classification head`


### 6.2 关闭所有 mask，对照基础异构超图结构

```bash
python train.py --mask_rate_cat 0 --lambda_cat 0 --mask_rate_reg 0 --lambda_reg 0 --mask_rate_poi 0 --lambda_poi 0
```


### 6.3 仅启用 Category + Region mask（关闭 POI mask）

```bash
python train.py --mask_rate_poi 0 --lambda_poi 0
```


### 6.4 仅启用 Region mask

```bash
python train.py --mask_rate_cat 0 --lambda_cat 0 --mask_rate_poi 0 --lambda_poi 0 --mask_rate_reg 0.2 --lambda_reg 0.02
```


### 6.5 仅启用 Category mask

```bash
python train.py --mask_rate_reg 0 --lambda_reg 0 --mask_rate_poi 0 --lambda_poi 0 --mask_rate_cat 0.2 --lambda_cat 0.05
```


### 6.6 只测试 geohash 粒度影响（例如切回 6）

```bash
python train.py --region_precision 6
```


### 6.7 关闭 category 分类辅助头（当前默认就是关闭）

```bash
python train.py --lambda_cat_cls 0
```


### 6.8 若想重新尝试 category 分类辅助头

```bash
python train.py --lambda_cat_cls 0.01
```

说明：

- 当前实验表明 `category classification head` 效果不稳定，因此默认关闭。


### 6.9 多 seed 对比

示例：

```bash
python train.py --seed 42 --save_dir logs_final
python train.py --seed 2025 --save_dir logs_final
python train.py --seed 2026 --save_dir logs_final
```


## 7. 当前阶段的主要实验结论

### 7.1 有效的改动

- 引入 `Category`、`Region` 异构节点后，结构本身有效。
- `Category mask` 有稳定小幅增益。
- `Region mask` 在 `geohash=6` 下效果一般，但在 `geohash=5` 下明显更好。
- `Category + Region mask` 一起使用优于单独使用其中之一。
- `POI mask` 进一步带来小幅增益，说明通过恢复 POI 表示来间接利用异构节点是有效的。

### 7.2 无效或不推荐保留的改动

- `category classification head` 效果不稳定，甚至可能与主任务冲突，当前默认关闭。
- 激进的动态融合（此前尝试过的全通道 softmax 动态权重）效果明显变差，已删除相关代码。
- `residual reweight` 对结果影响极小，已删除相关代码。


## 8. 当前最值得关注的问题

虽然 `Category/Region` 已经通过异构传播和 mask 训练被学到，但它们的信息仍然主要是“间接服务于 POI 推荐”。

也就是说：

- 异构节点已经被使用；
- 但如何“更充分、且不引入负面影响”地利用这些异构语义，仍然是后续可以继续探索的问题。

目前实验表明：

- 直接把 `category/region` 变成强辅助分类目标，不一定有效；
- 相比之下，通过 `mask` 这种较温和的方式约束异构节点表示更稳。


## 9. 推荐的当前使用方式

如果你只是想复现当前阶段最优版本，请直接运行：

```bash
python train.py
```

如果你想做严格对照，请至少保留下面三组：

1. 无任何 mask
2. `cat + reg mask`
3. `cat + reg + poi mask`（当前推荐版本）


## 10. 备注

- 当前工程偏研究原型，保留了若干实验参数开关，但默认值已经尽量收敛到当前最好配置。
- 若后续需要整理论文级实验，建议优先汇报 `best-Rec10-row`，同时保留 `best-by-metric` 作为补充说明。
