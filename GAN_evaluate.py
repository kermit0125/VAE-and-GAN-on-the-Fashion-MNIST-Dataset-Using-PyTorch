import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.models import inception_v3
from torchvision.utils import make_grid
from scipy.linalg import sqrtm

from GAN import Generator

class FashionMNISTCSV_Test(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        data = self.df.values
        self.labels = data[:, 0].astype(np.int64)
        # 数据归一化到 [0,1]
        self.images = data[:, 1:].astype(np.float32) / 255.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].reshape(28, 28)
        img = torch.tensor(img).unsqueeze(0)
        label = self.labels[idx]
        return img, label

def preprocess_for_inception(x):
    x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    return x

def get_inception_model(device):
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model

def get_activations(model, images, device, batch_size=64):
    model.eval()
    all_features = []
    total = len(images)
    count = 0
    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch = images[i:i+batch_size]
            batch = preprocess_for_inception(batch).to(device)
            features = model(batch)
            all_features.append(features.cpu().numpy())
            count += len(batch)
            print(f"Processed {count}/{total} images...")
    return np.concatenate(all_features, axis=0)

def calculate_fid(real_features, fake_features):
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    csv_file = "fashion-mnist_test.csv"
    test_dataset = FashionMNISTCSV_Test(csv_file)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    generator = Generator(nz=100).to(device)
    checkpoint_path = "generator.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found!")
        return
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval()
    print(f"Loaded generator from {checkpoint_path}")

    all_real = []
    for batch, _ in test_loader:
        all_real.append(batch)
    all_real = torch.cat(all_real, dim=0)

    n_samples = len(test_dataset)
    batch_size = 64
    fake_images = []
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            current_batch = min(batch_size, n_samples - i)
            noise = torch.randn(current_batch, 100, device=device)
            gen_imgs = generator(noise)
            gen_imgs = (gen_imgs + 1) / 2
            fake_images.append(gen_imgs.cpu())
    fake_images = torch.cat(fake_images, dim=0)

    grid_orig = make_grid(all_real[:64], nrow=8, padding=2)
    grid_fake = make_grid(fake_images[:64], nrow=8, padding=2)

    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(grid_orig.permute(1, 2, 0).squeeze(), cmap='gray')
    axes[0].set_title("Original Images")
    axes[0].axis('off')

    axes[1].imshow(grid_fake.permute(1, 2, 0).squeeze(), cmap='gray')
    axes[1].set_title("GAN Fake Images")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "final_test_GAN.png"), dpi=150)
    plt.show()

    inception = get_inception_model(device)
    real_feat = get_activations(inception, all_real, device)
    fake_feat = get_activations(inception, fake_images, device)
    fid_score = calculate_fid(real_feat, fake_feat)
    print(f"\nFID Score of GAN: {fid_score:.4f}")

if __name__ == '__main__':
    main()
