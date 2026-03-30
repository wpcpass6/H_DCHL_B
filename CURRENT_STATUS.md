# CURRENT STATUS

本文档用于记录 `H_DCHL_B` 当前阶段已经完成的实验、主要结论，以及当前建议保留的最佳配置，避免后续重复试错。


## 1. 项目当前目标

当前阶段的目标是：

- 在 DCHL 风格框架上构建异构超图版本 `H-DCHL-B`
- 验证异构节点（`Category`、`Region`）是否能带来结构增益
- 通过温和的 mask 机制促进异构节点信息被模型利用
- 先做研究原型验证，不追求论文级大规模调参


## 2. 当前模型结构

### 2.1 节点类型

- `User`
- `POI`
- `Category`
- `Region`

### 2.2 主要结构分支

- `User-POI` 协同分支：建模用户长期偏好
- `POI->POI` 有向转移分支：建模序列迁移关系
- `POI-Category` 异构分支：建模功能语义
- `POI-Region` 异构分支：建模空间语义

### 2.3 当前保留的 mask 机制

- `Category mask`
- `Region mask`
- `POI mask`

### 2.4 当前关闭/放弃的机制

- `Category classification head`：已验证效果不稳定，默认关闭
- 早期尝试的激进 `dynamic fusion`：效果明显变差，已删除
- `residual reweight`：效果极小，已删除


## 3. 已完成的实验路线

### 3.1 基础结构调参（无 mask）

已完成如下结构超参调参：

- `num_trans_layers`
- `num_col_layers`
- `num_reg_layers`
- `num_cat_layers`
- `dropout`

阶段性最优结构参数收敛为：

- `num_trans_layers = 4`
- `num_col_layers = 2`
- `num_reg_layers = 2`
- `num_cat_layers = 1`
- `dropout = 0.3`

说明：

- `trans` 与 `col` 对指标更敏感，说明时序迁移和长期协同仍然是 next POI 的核心主干
- `reg` 与 `cat` 的层数变化影响较小，说明它们更像语义增强分支而非主驱动分支


### 3.2 训练轮数观察

已验证：

- 早期 15 轮不足以覆盖最优结果
- 将训练轮数扩展到 50 轮后，结果更稳定

当前默认：

- `num_epochs = 50`


### 3.3 汇报方式调整

当前训练脚本会输出两套结果：

1. `best-by-metric`
   - 每个指标在所有 epoch 中分别取最大值
   - 与原始 DCHL 代码风格保持一致

2. `best-Rec10-row`
   - 以 `Rec10` 最优所在 epoch 的整行结果作为更严谨的比较口径

每次训练在日志目录中生成：

- `log_training.txt`
- `result.txt`
- `args.json`


## 4. 已完成的 mask 相关实验及结论

### 4.1 Category mask

结论：

- 有效，但增益不大
- 属于稳定的小幅正增益

解释：

- 直接约束 `Category` 节点表示是有价值的
- 但它更像温和正则，而不是会显著抬高主指标的主模块


### 4.2 Region mask（基于 geohash=6）

结论：

- 效果一般，不够稳定

解释：

- 说明 `Region` 的定义质量可能是问题，而不只是有没有监督的问题


### 4.3 Region 粒度调整：`geohash=6 -> geohash=5`

结论：

- `Region` 分支在 `geohash=5` 下明显优于 `geohash=6`
- 说明原先问题很可能是区域切得太细，导致 `Region` 节点语义不稳定

因此当前默认区域粒度为：

- `region_precision = 5`


### 4.4 Category mask + Region mask

结论：

- 在 `region_precision=5` 下，`cat + reg mask` 整体优于单独使用其中之一
- 说明 `Category` 和 `Region` 两类异构语义节点存在互补性

当前保留：

- `mask_rate_cat = 0.2`
- `lambda_cat = 0.05`
- `mask_rate_reg = 0.2`
- `lambda_reg = 0.02`

说明：

- `lambda_reg=0.02` 比更大的权重更稳，说明 `Region` 信息有用但更噪，不宜压得太重


### 4.5 Category classification head

尝试内容：

- 基于最终用户表示预测下一 `Category`
- 作为多任务辅助目标加入总损失

结论：

- 效果不稳定，整体不如当前最佳 mask 方案
- 说明“把异构节点直接变成强辅助分类任务”会和 `next POI` 主任务产生冲突

处理方式：

- 当前默认关闭：`lambda_cat_cls = 0.0`


### 4.6 Dynamic fusion / Residual reweight

尝试内容：

- 通过用户表示为多个 POI 分支预测动态权重
- 或对 `cat/reg` 做 residual 加权调制

结论：

- `dynamic fusion` 明显变差
- `residual reweight` 与静态融合几乎无差别

解释：

- 当前模型主干（`col + trans`）已经较强
- 语义分支更适合做温和补充，而不是强竞争式重加权

处理方式：

- 两者相关代码均已删除
- 当前保留最稳定的 `static` 融合方式


### 4.7 POI mask

尝试内容：

- 对基础 `POI` embedding 做随机 mask
- 使用 `user / transition / category / region` 四类结构恢复被遮蔽 POI

结论：

- 有效，但仍然是小幅提升
- 相比只做 `cat+reg mask`，加入 `POI mask` 后在单 seed 下表现更好

解释：

- `POI mask` 的价值不在于重复学习 POI，而在于逼模型真正调用异构上下文去恢复目标 POI

当前默认保留：

- `mask_rate_poi = 0.1`
- `lambda_poi = 0.02`


## 5. 当前推荐默认配置

当前建议的默认主线配置如下：

- `data_dir = datasets/TKY`
- `meta_path = datasets/TKY/meta.pkl`
- `region_precision = 5`
- `num_epochs = 50`
- `batch_size = 200`
- `emb_dim = 128`
- `lr = 0.001`
- `decay = 0.0005`
- `dropout = 0.3`
- `keep_rate = 1.0`
- `keep_rate_poi = 1.0`
- `num_col_layers = 2`
- `num_reg_layers = 2`
- `num_cat_layers = 1`
- `num_trans_layers = 4`
- `mask_rate_cat = 0.2`
- `lambda_cat = 0.05`
- `mask_rate_reg = 0.2`
- `lambda_reg = 0.02`
- `mask_rate_poi = 0.1`
- `lambda_poi = 0.02`
- `lambda_cat_cls = 0.0`


## 6. 当前最佳版本的理解

当前最佳版本可以理解为：

> 在异构超图结构基础上，通过 `Category/Region/POI` 三类 mask 任务，温和而稳定地提升异构节点信息利用效率。

它的特点是：

- 不依赖复杂对比学习
- 不依赖额外分类头
- 不依赖复杂动态融合
- 主要靠“结构 + mask”取得稳定小幅改进


## 7. 当前还没有完全解决的问题

当前仍然存在的核心问题：

- 虽然 `Category`、`Region` 节点已经被学习并参与传播，但它们的信息仍然主要是“间接服务于 POI 推荐”
- 如何在不破坏主任务的前提下，更充分利用异构节点信息，仍然是后续值得探索的方向

已经尝试但暂时失败的方向：

- 直接 category 分类辅助头
- 激进动态融合
- residual 通道重加权


## 8. 当前建议的使用方式

如果只是使用当前最佳版本，直接运行：

```bash
python train.py
```

如果要做最常见的对照：

1. 关闭全部 mask
2. 只开 `cat+reg mask`
3. 开 `cat+reg+poi mask`（当前推荐版本）


## 9. 当前阶段总结

一句话总结：

> `H-DCHL-B` 的异构结构本身成立；`Category` 与 `Region` 作为异构节点是有价值的；但最有效的利用方式目前仍然是温和的 mask 重建，而不是额外分类任务或复杂动态融合。
