import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torchvision.utils import save_image

from GAN import Generator, Discriminator

class FashionMNISTCSV(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        data = self.df.values
        self.labels = data[:, 0].astype(np.int64)
        self.images = data[:, 1:].astype(np.float32) / 255.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].reshape(28, 28)
        img = torch.tensor(img).unsqueeze(0)
        label = self.labels[idx]
        return img, label

def train_gan(generator, discriminator, dataloader, optimizer_G, optimizer_D, device, epochs=50):
    adversarial_loss = torch.nn.BCELoss()

    for epoch in range(1, epochs + 1):
        for batch_idx, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)
            imgs = imgs.to(device) * 2 - 1
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)
            optimizer_D.zero_grad()

            real_preds = discriminator(imgs)
            loss_real = adversarial_loss(real_preds, valid)

            noise = torch.randn(batch_size, 100, device=device)
            gen_imgs = generator(noise)
            fake_preds = discriminator(gen_imgs.detach())
            loss_fake = adversarial_loss(fake_preds, fake)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()


            optimizer_G.zero_grad()
            fake_preds = discriminator(gen_imgs)
            loss_G = adversarial_loss(fake_preds, valid)
            loss_G.backward()
            optimizer_G.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Batch {batch_idx}/{len(dataloader)}: Loss_D = {loss_D.item():.4f}, Loss_G = {loss_G.item():.4f}")
                save_image(gen_imgs * 0.5 + 0.5, f"results/epoch{epoch}_batch{batch_idx}.png", nrow=8)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_file = "fashion-mnist_train.csv"
    dataset = FashionMNISTCSV(csv_file)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    generator = Generator(nz=100).to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    if not os.path.exists("results"):
        os.makedirs("results")

    epochs = 50
    train_gan(generator, discriminator, dataloader, optimizer_G, optimizer_D, device, epochs=epochs)

    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")
    print("Models saved.")

if __name__ == '__main__':
    main()
