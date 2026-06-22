# 实验二：基于 LoRA 与 DPO 的数学推理模型后训练

本项目用于完成实验二的最小可运行流程：

```text
Base Model
-> 手写 LoRA 注入 q_proj / v_proj
-> 使用 Math-Step-DPO-10K 的 chosen 回答做小样本 SFT
-> 手写 DPO loss 做 chosen/rejected 单批次验证
-> 对比 Base 与 SFT-LoRA 输出差异
```

## 项目结构

```text
src/
  lora.py          # 手写 LoRALinear
  inject_lora.py   # 将 LoRA 注入目标 Linear 层
  dataset.py       # 加载并规范化 Math-Step-DPO-10K
  train_sft.py     # LoRA SFT 训练入口
  dpo_loss.py      # 手写 DPO loss 和序列 log-prob
  verify_dpo.py    # DPO 单批次验证入口
  evaluate.py      # Base vs SFT-LoRA 输出对比
tests/
  test_lora.py
  test_dpo_loss.py
outputs/
```

## 云端环境准备

本项目默认在云端 GPU 上运行。本地只负责写代码和同步 GitHub，不需要在本地安装深度学习环境。

建议云端使用 Python 3.10 或 3.11：

```bash
git clone https://github.com/你的用户名/lab2-math-reasoning-lora-dpo.git
cd lab2-math-reasoning-lora-dpo
python -m pip install -r requirements.txt
```

如果使用 AutoDL / 恒源云，建议选择：

```text
GPU: RTX 4090 24GB 或 RTX 3090 24GB
镜像: PyTorch 2.x + CUDA 12.x
```

云端启动后先确认 GPU：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## 可选单元测试

```bash
pytest tests
```

## 训练 LoRA SFT

如果 Hugging Face 数据集自动下载不稳定，可以先断点续传数据文件：

```bash
source /etc/network_turbo
mkdir -p data
curl -L -C - --retry 20 --retry-delay 3 \
  -o data/math_step_dpo_train.parquet \
  https://huggingface.co/datasets/xinlai/Math-Step-DPO-10K/resolve/main/data/train-00000-of-00001.parquet
```

### RTX 4090 / 24GB 推荐配置

可以直接运行一键脚本：

```bash
chmod +x scripts/run_4090_24g_training.sh
scripts/run_4090_24g_training.sh all
```

训练曲线会同时写入 CSV 和 TensorBoard：

```text
outputs/loss_log.csv
outputs/dpo_loss_log.csv
runs/sft/
runs/dpo/
```

查看实时曲线：

```bash
tensorboard --logdir runs --host 0.0.0.0 --port 6006
```

下面这组命令默认使用全量数据，不再抽样。脚本会用 `bf16` 加载模型，并默认开启 gradient checkpointing。`batch_size` 是实际进入显存的 micro batch；如果 `nvidia-smi` 显示显存仍有明显余量，再逐步增大 `batch_size`。

```bash
python -m src.train_sft \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data_file data/math_step_dpo_train.parquet \
  --max_length 1024 \
  --batch_size 8 \
  --gradient_accumulation_steps 2 \
  --epochs 2 \
  --rank 32 \
  --alpha 64 \
  --lr 2e-5 \
  --dtype bf16
```

如果 24GB 显存仍没有接近用满，可以继续加压：

```bash
python -m src.train_sft \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data_file data/math_step_dpo_train.parquet \
  --max_length 1024 \
  --batch_size 12 \
  --gradient_accumulation_steps 1 \
  --epochs 2 \
  --rank 32 \
  --alpha 64 \
  --lr 2e-5 \
  --dtype bf16
```

如果出现 CUDA out of memory，先把 `batch_size` 降到 4；仍然 OOM 再把 `max_length` 降回 768。排查环境时才需要使用 `--samples 20 --batch_size 1` 这类 smoke test 配置。

脚本也支持用环境变量继续加压，例如：

```bash
SFT_BATCH_SIZE=12 SFT_GRAD_ACCUM=1 scripts/run_4090_24g_training.sh sft
```

训练后会生成：

```text
outputs/sft_lora/
  adapter_config.json
  adapter_model.pt
outputs/loss_log.csv
```

## 验证 DPO Loss

先用 SFT 后的 LoRA adapter 做 DPO loss 方向验证：

```bash
python -m src.verify_dpo \
  --adapter_dir outputs/sft_lora \
  --reference_model Qwen/Qwen2.5-0.5B-Instruct \
  --data_file data/math_step_dpo_train.parquet \
  --samples 16 \
  --max_length 512
```

输出文件：

```text
outputs/dpo_check.txt
```

报告里重点说明：

- normal loss 非零；
- 交换 chosen/rejected 后 loss 发生变化；
- DPO loss 公式由代码手写实现。

## 完整 DPO 训练

DPO 同时计算 policy/reference 与 chosen/rejected，显存压力明显高于 SFT。4090 24GB 建议从下面配置开始：

```bash
python -m src.train_dpo \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --init_adapter_dir outputs/sft_lora \
  --output_dir outputs/dpo_lora \
  --data_file data/math_step_dpo_train.parquet \
  --max_length 1024 \
  --batch_size 4 \
  --gradient_accumulation_steps 2 \
  --epochs 2 \
  --rank 32 \
  --alpha 64 \
  --lr 1e-5 \
  --beta 0.1 \
  --dtype bf16
```

如果显存仍有余量，可以继续加压到：

```bash
python -m src.train_dpo \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --init_adapter_dir outputs/sft_lora \
  --output_dir outputs/dpo_lora \
  --data_file data/math_step_dpo_train.parquet \
  --max_length 1024 \
  --batch_size 6 \
  --gradient_accumulation_steps 1 \
  --epochs 2 \
  --rank 32 \
  --alpha 64 \
  --lr 1e-5 \
  --beta 0.1 \
  --dtype bf16
```

如果出现 CUDA out of memory，先把 DPO 的 `batch_size` 降到 2；仍然 OOM 再把 `max_length` 降回 768。

脚本加压 DPO 的示例：

```bash
DPO_BATCH_SIZE=6 DPO_GRAD_ACCUM=1 scripts/run_4090_24g_training.sh dpo
```

训练后会生成：

```text
outputs/dpo_lora/
outputs/dpo_loss_log.csv
```

再用 DPO 后的 adapter 验证：

```bash
python -m src.verify_dpo \
  --adapter_dir outputs/dpo_lora \
  --reference_model Qwen/Qwen2.5-0.5B-Instruct \
  --data_file data/math_step_dpo_train.parquet \
  --samples 16 \
  --max_length 512
```

## 输出对比

```bash
python -m src.evaluate \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_dir outputs/dpo_lora
```

输出文件：

```text
outputs/base_vs_lora.md
```

## GitHub 同步建议

只提交代码、日志和轻量结果，不提交模型权重或 checkpoint：

```bash
git add .
git commit -m "初始化数学推理LoRA与DPO实验项目"
git push
```

如果需要保存实验结果，建议只提交：

```text
outputs/loss_log.csv
outputs/dpo_loss_log.csv
outputs/dpo_check.txt
outputs/base_vs_lora.md
```

不要提交：

```text
outputs/sft_lora/
outputs/dpo_lora/
checkpoints/
models/
*.pt
*.safetensors
```
