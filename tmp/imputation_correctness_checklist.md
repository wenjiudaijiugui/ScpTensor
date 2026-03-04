# Imputation 正确性检测清单（临时开发指导）

> 用途：仅用于编码阶段验证，不属于正式文档。

## A. 通用正确性（所有方法）

- [ ] 输出形状与输入一致
- [ ] 非缺失位置数值不变（`max_abs_diff < 1e-12`）
- [ ] 缺失位置被填充（全缺失列按约定 fallback）
- [ ] 不污染源层（`source_layer` 不被改写）
- [ ] `M` 掩码正确标记 `IMPUTED`
- [ ] 同一 `random_state` 可复现
- [ ] 输入无缺失时 passthrough

## B. 方法级检查

- [ ] `knn`：填充值大体在邻居值范围内
- [ ] `lls`：低秩线性数据 RMSE 优于 `row_mean`
- [ ] `bpca`：低秩场景稳定，不发散/不扩散 NaN
- [ ] `missforest`：非线性数据 RMSE 优于线性基线
- [ ] `minprob`：填充值主要位于检测值左尾
- [ ] `qrilc`：左删失场景下左尾分布合理

## C. 机制匹配

- [ ] MCAR：`knn/lls/bpca/missforest` 优于 `minprob/qrilc`
- [ ] MNAR 左删失：`minprob/qrilc` 优于 `knn/lls`
- [ ] MAR：`missforest/knn/lls` 至少一类优于简单基线

## D. 生物学保真（真实 DIA 单细胞数据）

- [ ] 组内重复 CV 不恶化
- [ ] 组间效应（eta²/AUC）不被过度抹平
- [ ] 不引入伪分离（聚类虚高）

## E. 鲁棒性与性能

- [ ] 高缺失率（50/70/90%）不崩溃
- [ ] 全缺失行/列/常数列有定义行为
- [ ] 稀疏与稠密输入行为一致
- [ ] 运行时间与内存在可接受范围
