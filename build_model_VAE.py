import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from VAE import VAE, loss_function

class FashionMNISTCSV(Dataset):
    def __init__(self, csv_file): #data loader and normalization
        self.df = pd.read_csv(csv_file)
        data = self.df.values
        self.labels = data[:, 0].astype(np.int64)
        self.images = data[:, 1:].astype(np.float32) / 255.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx): #return label and processed image (1*28*28)
        img = self.images[idx].reshape(28, 28)
        img = torch.tensor(img).unsqueeze(0)
        label = self.labels[idx]
        return img, label

def train_vae(model, dataloader, optimizer, device, epochs=50):
    model.train()
    for epoch in range(1, epochs+1): #The default training time is 50 epochs.
        total_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss = loss_function(recon, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_file = "fashion-mnist_train.csv"
    dataset = FashionMNISTCSV(csv_file)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    
    latent_dim = 20
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 50
    train_vae(model, dataloader, optimizer, device, epochs=epochs)

    save_path = "VAE_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()
