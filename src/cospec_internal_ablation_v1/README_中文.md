# CoSpec Mamba 光谱编码内部消融

对应论文v16第3.2.2节和表6。此包可独立放入原工程的`src`目录，使用已有的convmamba环境和`COSPEC_SUITE_V1`缓存。

## 五组定义

| 命令标识 | Word中的组别 | 系数计算 | 可学习修正 | 观测分辨约束 |
|---|---|---|---|---|
| B | B 掩码编码 | 原始掩码感知线性编码 | 无 | 无 |
| LS | LS 无岭正则 | 64个固定基，SVD最小范数解 | 无 | 无 |
| R | R 仅岭拟合 | 原始Cholesky岭拟合，α=0.01 | 无 | 无 |
| RDelta | R+Δ 无约束修正 | 与R相同 | 零初始化，0.1×tanh | 无，直接残差相加 |
| D | D 完整算子 | 原始D实现 | 零初始化，0.1×tanh | 原始观测分辨矩阵 |

B、D的模型代码直接沿用归档实现。构造LS、R、RDelta时先完整构造原D，再替换光谱编码行为，保留同一组基函数和投影参数。LS、R实际移除修正层；RDelta与D的可学习参数规模一致。五组均关闭结构控制、区域细化，不引入图网络。

公共空间主干、位置编码、池化、读出与分类头逐种子核对初始化哈希。LS/R/RDelta/D的共同特征投影也核对哈希。若不一致则停止。此控制不等于五组总参数量相同；各组参数量记录在`initialization.json`。

## 正式实验协议

- HongHu与PaviaU，种子1、2、3，五组共30项独立训练，默认顺序执行。
- 15×15窗口、64个基函数、64维特征、四层四向蛇形Mamba；状态维24、卷积宽4、扩展1、dropout 0.1。
- 100轮，全程Adam学习率0.001、weight decay 0、梯度裁剪15；训练batch32，测试batch128。
- 完整视图和随机保留0.6至0.9的缺波段视图，交叉熵各占0.5；共享空间旋转，无重建损失。
- 沿用原缓存的训练/测试中心、归一化和坐标，校验已归档的文件哈希。不重新划分或重新拟合归一化。
- 测试种子20261014，沿用原`round()`保留数及共享排序；完整输入一次，80/60/40/20/10%各三套嵌套掩码，共16个条件。20/10%只测试同一组100轮终点权重。
- 使用第100轮权重。先平均每个种子内的三套掩码，再计算三个训练种子的均值与样本标准差（ddof=1）。不足三种子的表格单元格写TBD。

**LS的数值实现**：float32，`torch.linalg.svd(full_matrices=False)`，仅保留严格满足`s > 1e-6 × s_max`的奇异值，绝对截断阈值0。通过给未观测行补零进行批量SVD，其非零奇异值及最小范数解与仅使用保留行的设计矩阵等价；缺失光谱值通过`torch.where`排除。支持观测数量少于64或秩亏，不以加上一个很小的岭项替代LS。容差、库版本及源码哈希写入`config.json`。该设置可替换Word中“截断容差和数值实现”的TBD。

## 放置位置与启动

先解压，使目录为：

```text
ConvMamba-main/
  src/
    cospec_internal_ablation_v1/
      run.py
      start.sh
      self_check.py
      vendor/
    diagnostics/
      COSPEC_SUITE_V1/
        cache/HongHu/
        cache/PaviaU/
```

在WSL或AutoDL终端激活已经能跑原模型的环境，进入`ConvMamba-main/src`。例如本地WSL：

```bash
source ~/venvs/convmamba/bin/activate
cd ~/hsi-work/ConvMamba-main/src

# CPU实现自检，不使用真实数据，不生成论文精度。
python cospec_internal_ablation_v1/self_check.py

# 检查原数据、掩码、全部30组公共初始化；不训练。
bash cospec_internal_ablation_v1/start.sh --check-only

# 正式启动；训练/测试进度实时显示。
bash cospec_internal_ablation_v1/start.sh
```

默认读取`src/diagnostics/COSPEC_SUITE_V1`，输出到`src/diagnostics/COSPEC_INTERNAL_V1`。服务器目录不同时可显式指定：

```bash
bash cospec_internal_ablation_v1/start.sh \
  --reference /root/autodl-tmp/hsi-work/ConvMamba-main/src/diagnostics/COSPEC_SUITE_V1 \
  --output /root/autodl-tmp/hsi-work/ConvMamba-main/src/diagnostics/COSPEC_INTERNAL_V1 \
  --device cuda:0
```

`--reference`下的每个数据集需要`data.json`、`cube.npy`、`tr.npy`、`te.npy`、`wave.npy`、`stats.npz`。它们是原实验准备好的缓存，不是原始MAT/HDR目录。原数据内容、划分或掩码不一致时会给出路径和错误；不要改校验哈希强行通过。

代码包附带所需的原始CoSpec实现，不要求覆盖原工程代码或下载第三方模型源码。正式运行使用现有PyTorch 2.5.1/CUDA和mamba_ssm环境；启动器不会安装或升级依赖。默认在一个进程内按顺序构造和训练各组模型，完成后释放模型显存。LS对一批中的相同观测掩码只分解一次，减少完整视图和共享测试掩码下的重复SVD。默认五组全部重训，避免未经核验混用既有B、D权重。

需要分批做时，可先限定范围，例如：

```bash
bash cospec_internal_ablation_v1/start.sh \
  --datasets HongHu --variants LS R RDelta --seeds 1 2 3 \
  --output "$PWD/diagnostics/COSPEC_INTERNAL_HONGHU_NEW3"
```

此命令为9项训练，缺少B/D的汇总位置仍为TBD。不同任务清单使用不同输出目录；同一输出目录只允许完全相同协议与清单续跑。默认30项和上述分批方案是二选一，不要同时重复启动。此版本不导入旧B、D检查点，也不把旧数字直接拼接成新结果。

## 停止与续跑

```bash
python cospec_internal_ablation_v1/run.py stop
# 输出目录自定义时，stop也传相同的--output。
```

也可按Ctrl+C。训练在当前轮结束、保存后退出，测试在保存当前批次后退出。重新执行原启动命令即可续跑，已完成任务会跳过。每轮保存模型、Adam状态、全局随机数、数据顺序、旋转随机流和训练掩码随机流；测试保存像素游标与累计混淆矩阵。安全停止与强制关闭进程不同，强制终止时从最后一个完整保存点重做尚未保存的部分。

启动时核对协议、库版本、源码哈希和初始化，避免续跑时混入不同参数。任务中断或失败不会标为完成；失败原因保存在该任务的`outcome.json`。

## 汇总与结果文件

每完成一个任务自动更新汇总，全部完成后自动生成不含原始影像和权重的结果ZIP。也可随时执行：

```bash
python cospec_internal_ablation_v1/run.py summary
python cospec_internal_ablation_v1/run.py export
```

主要输出：

| 路径 | 内容 |
|---|---|
| `summary/table6_oa.csv` | Word表6顺序，完整/80/60/40%，另附20/10% |
| `summary/table6_oa.md` | 可直接阅读的表6 |
| `summary/per_seed.csv` | 每个种子先平均掩码后的OA、AA、κ |
| `summary/paired_differences.csv` | 按同种子求R−LS、RDelta−R、D−RDelta、D−R；另附R−B |
| `summary/completion.json` | 完成数、缺失任务及完整Word实验是否齐备 |
| `initialization_groups/` | 同数据集同种子的公共初始化核对记录 |
| `runs/patch_015/数据集/组别/seed_编号/` | 每组的初始化、历史、断点、16个条件指标与预测 |

默认结果包位于`src/diagnostics/COSPEC_INTERNAL_V1_results.zip`。跑完后可上传这个ZIP分析，不必上传原始高光谱影像或全部权重。汇总不会以9套“种子×掩码”结果计算标准差，也不预设任何组件一定提升。

## 验证范围

`self_check.py`检查：归档源码哈希、各组公共初始化、投影一致性、修正层参数移除、64维欠定LS与NumPy伪逆对照、岭拟合和两种修正的公式、缺失位置无效值排除、共享嵌套掩码、损坏缓存拒绝、训练续跑精确一致、测试续跑不重复统计、配对差值及样本标准差，以及正式15×15/64维/四层配置的五组CPU前向。

`--synthetic`仅供实现检查，固定为2轮/3×3/8维，输出目录必须以`_SYNTHETIC`结尾，不计为论文实验。交付时的测试报告见`validation/CPU_CHECKS.json`；真实HongHu/PaviaU的100轮CUDA训练和正式精度须在有原缓存与GPU的机器上运行。

