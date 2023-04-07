import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


class GAN:
    def __init__(self, lr=0.0002, betas=(0.5, 0.999)):
        self.img_shape = (1, 28, 28)

        self.latent_dim = 100

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build and compile the discriminator
        self.discriminator = Discriminator().to(self.device)
        self.optim_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)

        # Build the generator
        self.generator = Generator(self.latent_dim).to(self.device)

        # The combined model we will only train the generator
        self.discriminator.eval()

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.criterion = nn.BCELoss()
        self.optim_g = optim.Adam(self.generator.parameters(), lr=lr, betas=betas)

    def train(self, epochs, batch_size=128, sample_interval=50):
        # Load the dataset
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            for i, (imgs, _) in enumerate(train_loader):
                imgs = imgs.to(self.device)
                valid = torch.ones((imgs.size(0), 1)).to(self.device)
                fake = torch.zeros((imgs.size(0), 1)).to(self.device)

                # Train the discriminator with real images
                self.discriminator.zero_grad()
                d_real = self.discriminator(imgs)
                loss_d_real = self.criterion(d_real, valid)
                loss_d_real.backward()

                # Train the discriminator with fake images
                noise = torch.randn(imgs.size(0), self.latent_dim).to(self.device)
                gen_imgs = self.generator(noise)
                d_fake = self.discriminator(gen_imgs.detach())
                loss_d_fake = self.criterion(d_fake, fake)
                loss_d_fake.backward()

                loss_d = loss_d_real + loss_d_fake
                self.optim_d.step()

                # ---------------------
                #  Train Generator
                # ---------------------

                self.generator.zero_grad()
                d_fake_g = self.discriminator(gen_imgs)
                loss_g = self.criterion(d_fake_g, valid)
                loss_g.backward()
                self.optim_g.step()

                if i == len(train_loader) - 1:
                    print("%d [D loss: %f] [G loss: %f]" % (epoch, loss_d.item(), loss_g.item()))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch, nrow=5):
        noise = torch.randn(nrow ** 2, self.latent_dim).to(self.device)
        gen_imgs = self.generator(noise)
        gen_imgs = (gen_imgs + 1) / 2
        save_image(gen_imgs, f"images/{epoch}.png", nrow=nrow)
