# 🧠 VAE vs GAN on Fashion-MNIST  
使用 Fashion-MNIST 数据集实现的 VAE 与 GAN 生成模型对比项目

---

## 📌 Project Overview | 项目简介

This project explores and compares two generative models—**Variational Autoencoder (VAE)** and **Generative Adversarial Network (GAN)**—by implementing both using PyTorch and evaluating their performance on the **Fashion-MNIST** dataset.

本项目基于 PyTorch 实现了两种经典的生成模型：**变分自编码器（VAE）** 与 **生成对抗网络（GAN）**，并在 **Fashion-MNIST** 数据集上进行了图像生成实验与效果对比，旨在加深对生成模型结构、潜空间设计、训练稳定性的理解。

---

## 📂 Project Structure | 项目结构

```plaintext
├── VAE.py                 # VAE 模型定义
├── VAE_evaluate.py        # VAE 图像评估与输出
├── GAN.py                 # GAN 模型定义
├── GAN_evaluate.py        # GAN 图像评估与输出
├── build_model_VAE.py     # VAE 构建与训练主文件
├── build_model_GAN.py     # GAN 构建与训练主文件
├── generator.pth          # GAN 生成器模型参数
├── discriminator.pth      # GAN 判别器模型参数
├── VAE_model.pth          # VAE 模型参数
├── results/               # 生成图像结果示例
├── requirements.txt       # Python 依赖环境
└── README.md              # 项目说明文档
