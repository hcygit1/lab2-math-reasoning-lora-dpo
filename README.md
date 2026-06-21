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

第一次建议先跑小样本 smoke test：

```bash
python -m src.train_sft \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --samples 20 \
  --max_length 256 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --epochs 1
```

正式实验可以提高样本量：

```bash
python -m src.train_sft \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --samples 300 \
  --max_length 512 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --epochs 1
```

训练后会生成：

```text
outputs/sft_lora/
  adapter_config.json
  adapter_model.pt
outputs/loss_log.csv
```

## 验证 DPO Loss

```bash
python -m src.verify_dpo \
  --adapter_dir outputs/sft_lora \
  --reference_model Qwen/Qwen2.5-0.5B-Instruct \
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

## 输出对比

```bash
python -m src.evaluate \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_dir outputs/sft_lora
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
outputs/dpo_check.txt
outputs/base_vs_lora.md
```

不要提交：

```text
outputs/sft_lora/
checkpoints/
models/
*.pt
*.safetensors
```
