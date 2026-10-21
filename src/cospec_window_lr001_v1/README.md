# 7×7、11×11固定学习率补跑

本包只新增HongHu的B、D两组，7×7和11×11各三个种子，共12项100轮训练。
原9×9和15×15的12份已完成结果只读复用，用于最终窗口对照表，不重新训练。

## 在原AutoDL服务器启动

保留原 `src/diagnostics/COSPEC_FULL_LR001_V1` 和
`src/diagnostics/COSPEC_SUITE_V1/cache/HongHu`，以及原cospec环境。
将ZIP上传到 `src`，在终端执行：

```bash
cd /root/autodl-tmp/hsi-work/ConvMamba-main/src
python3 -m zipfile -e cospec_window_lr001_v1.zip .
bash cospec_window_lr001_v1/start.sh
```

启动脚本优先使用原 `/root/autodl-tmp/venvs/cospec/bin/python`。
需要指定其他解释器时可设置 `COSPEC_PYTHON=/你的环境/bin/python`。
无需下载模型、数据或重新安装已有环境。

## 设置

| 项目 | 固定设置 |
|---|---|
| 数据/模型/种子 | HongHu；B、D；1、2、3 |
| 本次训练窗口 | 7×7、11×11 |
| 优化器 | Adam，权重衰减0 |
| 训练 | 100轮，全程学习率0.001；训练batch32 |
| 测试 | batch128，原全部384493个TE中心 |
| 测试条件 | 完整；80%、60%、40%、20%、10%，各三套共享嵌套掩码，共16条件 |
| 训练视图 | 原完整/缺波段双视图交叉熵平均；原随机旋转和原观察采样器 |
| 其余模型参数 | 原dim64、depth4、basis64及原四向蛇形扫描 |
| 裁剪 | 原梯度范数阈值15 |

模型、训练、指标、导出、图块逻辑保持归档版本原文。批量读取、单批预取、
测试输入在16条件间复用也来自上次实际使用的加速版；不会改训练batch、测试batch或数值精度。

先核验原9×9、15×15的100轮学习率记录、16条件结果和HongHu缓存哈希，
然后在当前GPU对B/D×两个窗口共4类任务分别做少量数值及显存峰值预检。
这些预检参数不会用作正式训练初始化。正式训练全部从各自种子初始化开始。
预检需要独占空闲GPU；原210项队列必须已结束。

## 调度和精简界面

默认按训练/测试显存峰值、当前空闲显存和容器RAM动态准入，最多10项并行，
不是强制一直10项。保留显存余量，OOM会降低并行、从本任务断点重试；
数值异常会记入failures.json，不会悄悄调低学习率或跳过样本。

终端只显示四项：总任务、已完成、当前并行及任务名称、预计剩余。
ETA先显示待测，之后按各任务训练/测试成本与实际总吞吐更新；它是估计值。

```bash
# 关闭观看终端不影响后台队列；重新打开界面
bash cospec_window_lr001_v1/watch.sh

# 单次查看
python3 cospec_window_lr001_v1/status.py

# 动态将并行上限改为5；现有任务自然完成，不杀进程
bash cospec_window_lr001_v1/slots.sh 5

# 请求保存当前轮/测试游标后停止
bash cospec_window_lr001_v1/stop.sh

# 恢复，完成项跳过，未完成项从本次断点继续
bash cospec_window_lr001_v1/start.sh
```

训练每个完整epoch原子保存last.pt和随机状态，测试每50批或主动停止保存各条件游标。
异常断电最多丢失当前未保存的轮/测试批次；已完整保存的数据可恢复。
Ctrl+C只退出观看界面；要停止后台训练使用stop.sh。

## 结果

新输出为 `src/diagnostics/COSPEC_WINDOW_LR001_V1`，不会写入原210项结果目录。

- `summary.csv`：本次12项，三掩码先在种子内平均，再计算三个训练种子的均值和样本标准差。
- `window_comparison.csv/json`：与已核验9×9、15×15合并的四窗口对照。
- `table_4_3.csv`：可用于4.3节的四窗口OA表，未完成项显示TBD，不借用历史7/11数值。
- `protocol_audit.json`：设置、数据/掩码一致性、旧窗口学习率核验记录。
- `runs/patch_007/...`、`runs/patch_011/...`：本次权重、逐轮日志、逐条件预测和指标。
- `parent_snapshot`：原9/15的只读JSON快照及哈希，不复制旧权重。

结束后自动生成 `diagnostics/COSPEC_WINDOW_LR001_V1_results.zip`，
另有maps/features ZIP。结果ZIP不包含.pt检查点；保留服务器runs目录以便以后补测。
只有界面显示“全部完成”才表示训练、全部16条件测试、汇总与打包都结束。

如需更换路径可用 `start.sh --reference /旧缓存根目录 --parent-output /原210结果目录 --output /新补跑结果目录`。
所有输出必须与原结果、缓存、代码目录分开。不要用原210项目录作为--output。

## 可选自检

`bash cospec_window_lr001_v1/check.sh` 用合成数据检查12项清单、16掩码、
加速数据等价、固定0.001的100轮CPU训练/断点、测试游标以及实际B/D前后向。
`check.sh --device cuda` 会把最后四个实际模型前后向检查放到GPU。
这类合成检查不构成正式分类精度结果。
