# RSReason: Remote Sensing Reasoning Segmentation via Rotation-aware Large Vision-Language Model

**Changli Wu<sup>1,2,†</sup>, Yonghan Zheng<sup>1,†</sup>, Jiayi Ji<sup>1</sup>, Xiaoshuai Sun<sup>1</sup>, Tat-Seng Chua<sup>3</sup>, Liujuan Cao<sup>1</sup>**

<sup>1</sup>Xiamen University · <sup>2</sup>Shanghai Innovation Institute · <sup>3</sup>National University of Singapore  
<sup>†</sup>Equal contribution

---

## 📖 Abstract

We introduce **Remote Sensing Reasoning Segmentation (RSReasonSeg)**, which segments targets and generates explanatory answers from implicit, reasoning-centric instructions. The **RSReason** benchmark contains **5,138 image–instruction–answer–mask samples**. Our **RSReasonNet** incorporates **Rotation Awareness Optimization (RAO)** to align mask geometry and orientation through Gaussian representations and Wasserstein distance, without additional inference-time overhead.

## 🛠️ Installation

Create a Python 3.10 environment on Linux with CUDA, then install the dependencies:

```bash
git clone https://github.com/sosppxo/RSReason.git
cd RSReason
conda create -n rsreason python=3.10 -y
conda activate rsreason
pip install -r requirements.txt
pip install deepspeed peft tensorboard tqdm pycocotools matplotlib scikit-image
```

## 📂 Benchmark

The RSReason benchmark is coming soon.

## 📦 Model Weights

Prepare compatible LLaVA, CLIP, and SAM ViT-H weights for the current implementation. Set their paths using `--version`, `--vision-tower`, and `--vision_pretrained`.

Pre-trained RSReasonNet weights are coming soon.

## 🚀 Training

Set the model and data paths, then run the following example:

```bash
deepspeed --num_gpus=8 train_ds.py \
  --version /path/to/llava --conv_type llava_llama_2 \
  --vision-tower /path/to/clip --vision_pretrained /path/to/sam_vit_h.pth \
  --dataset_dir ./dataset --dataset reason_seg --sample_rates 1 \
  --reason_seg_data 'RSReason|train' --val_dataset 'RSReason|val' \
  --exp_name rsreason --rao_loss_weight 1.0
```

## 🔍 Evaluation

Load the matching DeepSpeed checkpoint and evaluate on the test split:

```bash
deepspeed --num_gpus=8 train_ds.py \
  --version /path/to/llava --conv_type llava_llama_2 \
  --vision-tower /path/to/clip --vision_pretrained /path/to/sam_vit_h.pth \
  --dataset_dir ./dataset --dataset reason_seg --sample_rates 1 \
  --reason_seg_data 'RSReason|train' --val_dataset 'RSReason|test' \
  --resume ./runs/rsreason/ckpt_model --eval_only
```

Use the same backbone, LoRA settings, and GPU count as the checkpoint. Evaluation reports gIoU, cIoU, and Precision@0.5–0.9 (Precision@0.5 corresponds to Acc@50).

## 🖥️ Demo

The interactive demo is coming soon.

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@misc{wu_rsreason,
  author = {Changli Wu and Yonghan Zheng and Jiayi Ji and Xiaoshuai Sun and Tat-Seng Chua and Liujuan Cao},
  title = {RSReason: Remote Sensing Reasoning Segmentation via Rotation-aware Large Vision-Language Model}
}
```

