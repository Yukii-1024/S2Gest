# S2Gest 中文使用指南（Linux + NVIDIA GPU + uv）

本文面向 Linux 上使用 NVIDIA RTX 50 系显卡的单卡训练与评估，使用 `uv` 管理独立 Python 虚拟环境。安装流程沿用项目原有的 CUDA 方案：PyTorch 2.2.1 + CUDA 11.8，以及与其匹配的 `causal-conv1d`、`mamba-ssm` 预编译 wheel。

## 1. 创建独立环境

在仓库根目录执行。`uv` 管理 Python 3.12.3，项目依赖安装在 `.venv`，不使用系统 Python 包：

```bash
uv python install 3.12.3
uv venv --python 3.12.3
source .venv/bin/activate
```

RTX 50 系需要可支持该显卡的 NVIDIA 驱动。PyTorch 2.2.1 + CUDA 11.8 是项目原有安装组合；若驱动或该组合无法支持 RTX 50 系，需换用兼容的 PyTorch/CUDA 与扩展 wheel 组合。

## 2. 安装 PyTorch 与 Mamba 扩展

使用项目原有的 CUDA 11.8 PyTorch wheel：

```bash
uv pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
  --index-url https://download.pytorch.org/whl/cu118
uv pip install packaging ninja==1.13.0
```

从官方发布页下载适配 **Python 3.12、PyTorch 2.2.1、CUDA 11.8** 的 Linux 预编译 wheel：

- [`causal-conv1d` v1.4.0](https://github.com/Dao-AILab/causal-conv1d/releases)
- [`mamba-ssm` v2.0.3](https://github.com/state-spaces/mamba/releases)

确保下载文件与当前 Python、PyTorch 和 CUDA 版本匹配，然后在 wheel 所在目录安装（替换为实际文件名）：

```bash
uv pip install ./causal_conv1d-*.whl ./mamba_ssm-*.whl
```

## 3. 最小安装验证

确认运行于虚拟环境且 CUDA 可用：

```bash
which python
python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA:", torch.version.cuda); print("GPU:", torch.cuda.get_device_name(0)); assert torch.cuda.is_available()'
python -c 'from mamba_ssm import Mamba; print("mamba_ssm import OK")'
```

`which python` 应指向仓库 `.venv/bin/python`。

## 4. 数据准备

从数据集官方来源获取数据：

- [NVGesture](https://research.nvidia.com/publication/2016-06_online-detection-and-classification-dynamic-hand-gestures-recurrent-3d)
- [EgoGesture](https://nlpr.ia.ac.cn/iva/yfzhang/datasets/egogesture.html)
- [Briareo](https://aimagelab-legacy.ing.unimore.it/imagelab/page.asp?IdPage=31)
- [Jester](https://20bn.com/datasets/jester)

示例目录：

```text
data/
├── NVGesture/
│   ├── Video_data/
│   └── nvgesture_train_correct_cvpr2016_v2.lst
├── EgoGesture/
│   ├── images/
│   └── labels-final-revised1/
├── Briareo/
│   ├── train/
│   ├── validation/
│   └── test/
└── jester/
    ├── 20bn-jester-v1-videos/
    ├── jester-v1-train.csv
    └── jester-v1-validation.csv
```

实际数据路径以 `hyperparameters/` 中对应 JSON 配置和数据集格式为准；按需修改 `data_path`。

## 5. 单卡训练与评估

选择 `hyperparameters/` 中对应数据集和模态的配置，例如 `hyperparameters/NVGesture/depth.json`，检查数据路径、批大小、epoch 数和学习率。单卡训练：

```bash
python main.py --phase train \
  --hypes hyperparameters/NVGesture/depth.json \
  --gpus 0
```

评估已有权重：

```bash
python main.py --phase test \
  --hypes hyperparameters/NVGesture/depth.json \
  --resume path/to/model.pth \
  --gpus 0 --accuracy --matrix
```

效率基准：

```bash
python main.py --phase test \
  --hypes hyperparameters/NVGesture/depth.json \
  --resume path/to/model.pth \
  --gpus 0 --benchmark
```

恢复训练时，在训练命令中添加 `--resume path/to/checkpoint.pth`。

## 6. 预训练权重与引用

Small、Tiny、Nano 权重链接及结果见 [README.md](README.md) 的 Model Zoo。论文信息及 BibTeX 请参阅其 Citation 部分。项目采用 MIT License，详情见 [LICENSE](LICENSE)。

