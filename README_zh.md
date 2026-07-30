# Enfold

Enfold: Folding World-Generator Computation into Predictive Representations for Efficient Embodied Control的官方代码仓库

[![English](https://img.shields.io/badge/README-English-111111.svg)](./README.md)
[![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-d14836.svg)](./README_zh.md)

[![arXiv](https://img.shields.io/badge/arXiv-2607.26657-b31b1b.svg)](https://arxiv.org/abs/2607.26657)
[![Project Page](https://img.shields.io/badge/Project_Page-Enfold-2ea44f.svg)](https://zwl666666.github.io/enfold/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-f7c843.svg)](https://huggingface.co/richardxyt/Enfold)

本仓库包含 Enfold 在 LIBERO 和 RoboTwin 上的训练与评测代码。

## 索引

- [目录结构](#目录结构)
- [模型准备](#模型准备)
- [数据集下载](#数据集下载)
- [训练](#训练)
- [推理](#推理)
- [致谢](#致谢)

## 目录结构

```text
Enfold/
├── configs/
│   ├── data/                 # LIBERO 与 RoboTwin 数据集配置
│   ├── model/enfold.yaml     # Cosmos、DINOv3 和 ActionDiT 配置
│   ├── task/                 # 任务级别的训练配置
│   ├── sim_libero.yaml       # LIBERO 评测配置
│   └── sim_robotwin.yaml     # RoboTwin 评测配置
├── scripts/
│   ├── train.py
│   ├── initialize_action_dit.py
│   └── precompute_text_embeds.py
├── experiments/
│   ├── libero/               # LIBERO 评测管理器与 worker
│   └── robotwin/             # RoboTwin 评测管理器与 policy
├── src/enfold/               # 核心实现
├── third_party/
│   ├── cosmos-predict2.5/    # Cosmos-Predict2.5 第三方库
│   └── RoboTwin/             # vendored RoboTwin 评测集成
├── checkpoints/              # 外部及生成的模型权重
├── data/                     # 数据集和文本特征缓存
├── runs/                     # 训练输出
└── evaluate_results/         # 评测输出
```

## 模型准备

训练和推理前，需要下载并准备以下外部资源：

- 视频基座：[Cosmos-Predict2.5-2B](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B/tree/main/base/post-trained) 与 [Cosmos tokenizer](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B/blob/main/tokenizer.pth)
- 文本编码器：[Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B) 与 [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- 视觉编码器：[DINOv3 ViT-H/16+](https://github.com/facebookresearch/dinov3)

默认路径见 [`configs/model/enfold.yaml`](./configs/model/enfold.yaml)。若使用其他本地目录，请修改对应配置项。

```text
checkpoints/
├── 81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt  # Cosmos-Predict2.5-2B base post-trained
├── tokenizer.pth                                      # Cosmos tokenizer / VAE
├── Cosmos-Reason1-7B/                                 # Cosmos 文本编码器 checkpoint
├── Qwen2.5-VL-7B-Instruct/                            # Qwen tokenizer 与模型资源
└── dinov3-vith16plus/                                 # DINOv3 ViT-H/16+ checkpoint
```

默认配置字段如下：

| 组件 | 配置字段 |
| --- | --- |
| Cosmos teacher | `model.video_dit_config.cosmos.checkpoint_path` |
| Cosmos tokenizer / VAE | `model.video_dit_config.cosmos.tokenizer_vae_pth` |
| Cosmos-Reason1 与 Qwen 资源 | `model.video_dit_config.cosmos.text_embedding_*` |
| DINOv3 student | `model.student_config.model_id` |

## 数据集下载

Enfold 遵循 Fast-WAM 的数据准备流程，并使用相同的预处理 LeRobot 格式 LIBERO 与 RoboTwin 数据集。

### LIBERO

预处理后的 LIBERO 数据集地址：

- https://huggingface.co/datasets/yuanty/LIBERO-fastwam

请先下载所有压缩文件，再统一解压：

```bash
mkdir -p data/libero_mujoco3.3.2
cd data/libero_mujoco3.3.2

# 下载全部 4 个 tar.gz 文件后执行
for f in *.tar.gz; do
  tar -xzf "$f"
done
```

解压后的目录结构应为：

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

### RoboTwin

预处理后的 RoboTwin 数据集地址：

- https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam

请先下载所有分卷压缩包，再拼接并解压：

```bash
mkdir -p data/robotwin2.0
cd data/robotwin2.0

# 下载全部 robotwin2.0.tar.gz.part-* 文件后执行
cat robotwin2.0.tar.gz.part-* | tar -xzf -
```

解压后的目录结构应为：

```text
data/robotwin2.0/
└── robotwin2.0/
    ├── data/
    ├── meta/
    └── videos/
```

请将 `data/robotwin2.0/dataset_stats.json` 保留在数据根目录，当前 RoboTwin 配置可直接使用。使用新数据集时，也可以重新计算该文件。

## 训练

### 1) 环境安装

```bash
conda create -n enfold python=3.10 -y
conda activate enfold

pip install -U pip
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .

# 安装 Cosmos-Predict2.5 依赖到当前 conda 环境
cd third_party/cosmos-predict2.5

# 如未安装 uv，先执行以下两行
curl -LsSf https://astral.sh/uv/install.sh | sh
source "${HOME}/.local/bin/env"

VIRTUAL_ENV="$CONDA_PREFIX" uv sync --extra=cu128 --active --inexact --python "$CONDA_PREFIX/bin/python"
```

### 2) 初始化 ActionDiT Backbone

训练前，需要通过配置的 Cosmos teacher 初始化 ActionDiT backbone。

```bash
# RoboTwin
bash init_action.sh robotwin --output checkpoints/action_dit_cosmos_init_robotwin.pt

# LIBERO
bash init_action.sh libero --output checkpoints/action_dit_cosmos_init_libero.pt
```

后续训练时，请通过 `model.action_dit_pretrained_path` 传入与 benchmark 对应的初始化文件。

### 3) 预计算 Cosmos 文本特征

训练前预计算 Cosmos-Reason1 文本特征缓存：

```bash
# LIBERO
python scripts/precompute_text_embeds.py task=enfold_libero

# RoboTwin
python scripts/precompute_text_embeds.py task=enfold_robotwin
```

分布式预计算示例：

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/precompute_text_embeds.py task=enfold_libero
```

### 4) 启动训练

根目录的 launcher 会将 benchmark 名称映射到相应的 Hydra task 配置：

```bash
# 单机 8 卡：LIBERO
bash train.sh libero 8 \
  model.action_dit_pretrained_path=checkpoints/action_dit_cosmos_init_libero.pt

# 单机 8 卡：RoboTwin
bash train.sh robotwin 8 \
  model.action_dit_pretrained_path=checkpoints/action_dit_cosmos_init_robotwin.pt
```

可直接追加其他 Hydra override：

```bash
bash train.sh robotwin 8 \
  model.action_dit_pretrained_path=checkpoints/action_dit_cosmos_init_robotwin.pt \
  model.video_dit_config.use_gradient_checkpointing=true
```

多机训练时，在每个节点上使用该节点对应的 `NODE_RANK` 执行：

```bash
NNODES=4 NODE_RANK=0 MASTER_ADDR=<rank0-host> MASTER_PORT=29500 \
  bash scripts/train_zero1_multinode.sh 8 \
  task=enfold_robotwin \
  model.action_dit_pretrained_path=checkpoints/action_dit_cosmos_init_robotwin.pt
```

训练输出（包括 checkpoint 和生成的 dataset statistics）保存在 `runs/<task>/<run_id>/`。

## 推理

推理需要训练好的 checkpoint 及其对应的 `dataset_stats.json`。建议为推理单独创建环境，以避免与训练或其他benchmark 依赖发生冲突；仍需按“模型准备”一节放置所有外部资源。

### LIBERO

#### 环境配置
```bash
conda create -n enfold_libero python=3.10 -y
conda activate enfold_libero

pip install -U pip
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```
根据官方 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 文档安装 LIBERO，再执行以下命令。MuJoCo 版本应与 LIBERO 数据版本保持一致。
```bash
pip install mujoco==3.3.2
```
安装完libero依赖后再安装cosmos依赖。
```bash
# 安装 Cosmos-Predict2.5 依赖
cd third_party/cosmos-predict2.5

# 如未安装 uv，先执行以下两行
curl -LsSf https://astral.sh/uv/install.sh | sh
source "${HOME}/.local/bin/env"

VIRTUAL_ENV="$CONDA_PREFIX" uv sync --extra=cu128 --active --inexact --python "$CONDA_PREFIX/bin/python"
```

#### 推理步骤

评测已配置的全部 LIBERO suite：

```bash
bash eval.sh libero <checkpoint.pt> <dataset_stats.json> \
  MULTIRUN.num_gpus=8 MULTIRUN.max_tasks_per_gpu=1
```

如需使用更少的 GPU，降低 `MULTIRUN.num_gpus` 即可。结果默认写入 `evaluate_results/libero/`。

### RoboTwin

#### 环境配置

```bash
conda create -n enfold_robotwin python=3.10 -y
conda activate enfold_robotwin

pip install -U pip
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .

# 安装 Cosmos-Predict2.5 依赖
cd third_party/cosmos-predict2.5

# 如未安装 uv，先执行以下两行
curl -LsSf https://astral.sh/uv/install.sh | sh
source "${HOME}/.local/bin/env"

VIRTUAL_ENV="$CONDA_PREFIX" uv sync --extra=cu128 --active --inexact --python "$CONDA_PREFIX/bin/python"
```
根据官方[RoboTwin 安装文档](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) 准备 RoboTwin 环境，RoboTwin环境安装后恢复本评测流程使用的版本：
```bash
pip install huggingface-hub==0.36.0
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
```
本仓库只包含 RoboTwin 的评测集成，不包含其所需要的 assets。需要先按照[官方文档](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)进行下载并放到third_party/Robotwin/assets下，然后创建以下链接：

```bash
ln -sfn "$(pwd)/experiments/robotwin/enfold_policy" \
  "$(pwd)/third_party/RoboTwin/policy/enfold_policy"
```

#### 推理步骤

评测 RoboTwin：

```bash
bash eval.sh robotwin <checkpoint.pt> <dataset_stats.json> \
  MULTIRUN.num_gpus=8 MULTIRUN.max_tasks_per_gpu=1 \
  EVALUATION.replan_steps=24
```

## 致谢

本工作建立在 [Fast-WAM](https://github.com/yuantianyuan01/FastWAM) 的基础上。我们也感谢 [Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5)、[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)、[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 和 [DINOv3](https://github.com/facebookresearch/dinov3) 团队公开相关工作。

## BibTeX
