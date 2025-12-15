# MobileNetV2-for-CIFAR-10
This assignment focuses on training MobileNet-v2 on CIFAR-10 and then applying model compression techniques to reduce model size while retaining accuracy.

# Steps to replicate the results in colab
1. Clone the repository and upload it to your drive.
2. Setup your environment and workspace in colab and select available runtime environment.
```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/<project root folder>
   !pip install torch torchvision wandb
   import wandb
   wandb.login()
```
3. Train the baseline. This saves: ./checkpoints/mobilenetv2_cifar_baseline.pt
```python
!python train.py \
  --data_dir ./data \
  --epochs 200 \
  --batch_size 128 \
  --lr 0.1 \
  --weight_decay 5e-4 \
  --momentum 0.9 \
  --width_mult 1.0 \
  --dropout 0.2 \
  --pretrained \
  --save_path ./checkpoints \
  --wandb_project cs6886-assignment3
```
4. Quantization sweep (weights + activations, per-channel)

For the weights:

Conv2d layer - Per-tensor quantization

Linear Layers - Per-channel quantization

For the Activation:

Per-tensor quantization, applied after every Conv2d and Linear layer.
```python
!python compress_eval.py \
  --data_dir ./data \
  --batch_size 128 \
  --checkpoint ./checkpoints/mobilenetv2_cifar_baseline.pt \
  --weight_bits_list 8 6 4 2 \
  --activation_bits_list 8 6 4 2 \
  --wandb_project cs6886-assignment3
```
5. Activation size measurement (for activation compression ratio)
```python
!python measure_activations.py \
  --data_dir ./data \
  --batch_size 64 \
  --checkpoint ./checkpoints/mobilenetv2_cifar_baseline.pt \
  --activation_bits 8
  ```
