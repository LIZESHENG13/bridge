# BRIDGE

[English](README.md) | **简体中文**

**CIKM 2026 · Full Research Paper（正式研究论文）**

> **CIKM 2026 论文
> [BRIDGE: Behavior-Guided Residual Integration with Dual-Frequency Graph Evidence](https://doi.org/10.1145/3799682.3840961)
> 的官方 PyTorch 代码实现。**

Zesheng Li、Chengchang Pan、Honggang Qi<br>
中国科学院大学

[论文（ACM DL）](https://doi.org/10.1145/3799682.3840961) | [PDF](docs/assets/bridge-paper.pdf) | [项目主页](docs/index.html) | [模型代码](src/models/bridge.py)

## 项目简介

多模态推荐可以利用图像和文本内容增强物品表示，但更强的跨视图对齐并不一定
带来更好的排序效果。BRIDGE 将多模态表示学习与行为引导的分数校准分开：多模态
主干负责召回合理候选，行为证据只在基础模型的 top-*K* 候选集中调整局部顺序。

BRIDGE 包含三个核心模块：

- **DFGE** 将图平滑后的 ID、视觉和文本表示分解到不同频段，同时保留共享结构
  与模态私有的排序信号。
- **BEN** 将仅由训练集构建的共同用户重叠转化为带符号的行为证据。
- **CRI** 在训练和推理阶段都只对基础候选集应用行为引导的残差校准。

## 模型框架

![BRIDGE 模型框架](docs/assets/framework.png)

## 主要结果

BRIDGE 在三个 Amazon 数据集上采用全排序评测。表中结果为五个随机种子的平均值。

| 数据集 | Recall@20 | NDCG@20 |
| --- | ---: | ---: |
| Baby | **0.1128** | **0.0525** |
| Sports | **0.1262** | **0.0594** |
| Electronics | **0.0778** | **0.0385** |

## 项目结构

```text
BRIDGE-release/
├── docs/                 # 项目主页、论文插图和论文 PDF
├── scripts/              # 三个数据集的复现实验脚本
├── src/
│   ├── configs/          # 数据集和模型配置
│   ├── models/           # BRIDGE 与双频编码器
│   ├── common/           # 训练框架和损失函数
│   ├── utils/            # 数据读取与评测工具
│   └── main.py           # 训练入口
├── environment.yaml      # 参考 Conda 环境
├── requirements.txt      # Python 依赖列表
└── LICENSE
```

## 环境配置

参考环境使用 Python 3.10、PyTorch 2.1.2、CUDA 12.1 和 PyTorch Geometric
2.7.0，可通过以下命令创建：

```bash
conda env create -f environment.yaml
conda activate bridge
```

如果已经根据本机 CUDA 版本安装了 PyTorch 和 PyTorch Geometric，其余依赖可参考
`requirements.txt`。

## 数据准备

本仓库不直接分发数据文件。请按以下结构放置处理后的数据：

处理后的数据集下载：[夸克网盘](https://pan.quark.cn/s/ab2edddf790c)。

```text
data/
├── baby/
│   ├── baby.inter
│   ├── image_feat.npy
│   └── text_feat.npy
├── sports/
│   ├── sports.inter
│   ├── image_feat.npy
│   └── text_feat.npy
└── elec/
    ├── elec.inter
    ├── image_feat.npy
    └── text_feat.npy
```

交互文件采用制表符分隔，并包含以下字段：

| 字段 | 说明 |
| --- | --- |
| `userID` | 整数编码的用户编号 |
| `itemID` | 整数编码的物品编号 |
| `x_label` | 数据划分：`0` 为训练集，`1` 为验证集，`2` 为测试集 |

`image_feat.npy` 和 `text_feat.npy` 的行顺序必须与编码后的 `itemID` 对齐。
`user_emb.npy` 与 `user_graph_dict.npy` 仅用于可选的旧版用户图分支，默认
BRIDGE 配置不需要这两个文件。

也可以将 `data` 设置为指向本地数据目录的符号链接。请勿提交与机器绑定的数据路径。

## 训练与评测

在项目根目录运行实验：

```bash
# Amazon Baby
GPU_ID=0 SEED=2020 bash scripts/run_bridge_baby.sh

# Amazon Sports
GPU_ID=0 SEED=2020 bash scripts/run_bridge_sports.sh

# Amazon Electronics
GPU_ID=0 SEED=2020 bash scripts/run_bridge_elec.sh
```

如需指定 Python 解释器：

```bash
PYTHON_BIN=/path/to/python GPU_ID=0 SEED=2020 bash scripts/run_bridge_baby.sh
```

也可以直接调用训练入口：

```bash
cd src
python main.py --model BRIDGE --dataset baby --gpu_id 0 --seed 2020
```

模型超参数位于
[`src/configs/model/BRIDGE.yaml`](src/configs/model/BRIDGE.yaml)，数据集配置位于
[`src/configs/dataset/`](src/configs/dataset/)。

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@inproceedings{li2026bridge,
  author    = {Li, Zesheng and Pan, Chengchang and Qi, Honggang},
  title     = {BRIDGE: Behavior-Guided Residual Integration with Dual-Frequency Graph Evidence},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/3799682.3840961},
  isbn      = {979-8-4007-2539-5/2026/11}
}
```

## 开源协议

代码基于 [GNU General Public License v3.0](LICENSE) 发布。
