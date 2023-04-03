import torch
from torch import nn, optim
from torchvision import transforms


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),
            nn.Sigmoid()  # 将输出限制在 [0, 1] 范围内
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train(model, train_loader, test_loader, pre_frame, lr=1e-3, num_epochs=100, device='cpu'):
    print(f'train on {device}')

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    unloader = transforms.ToPILImage()
    l = nn.MSELoss()
    train_loss = []
    test_loss = []

    for epoch in range(num_epochs):
        running_loss = 0
        for i, data in enumerate(train_loader):
            inputs, outputs = data
            inputs = inputs.to(device)
            outputs = outputs.to(device)

            optimizer.zero_grad()
            outputs_bar = model(inputs)
            loss = l(outputs_bar, outputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        now_test_loss = evaluate(model, test_loader, l, device)
        train_loss.append(epoch_loss)
        test_loss.append(now_test_loss)
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: train loss: {epoch_loss:.4f}, test loss: {now_test_loss:.4f}")

           # preview_frame = create_next_frame(model, pre_frame, unloader)
           # preview_frame.save(f'outputs/{epoch+1} output.jpg')

    return train_loss, test_loss


def evaluate(model, test_loader, l, device='cpu'):
    testing_loss = 0
    model = model.to(device)
    for i, data in enumerate(test_loader):
        inputs, outputs = data
        inputs.to(device)
        outputs.to(device)

        with torch.no_grad():
            outputs_bar = model(inputs)
            loss = l(outputs_bar, outputs)

            testing_loss += loss

    epoch_loss = testing_loss / len(test_loader)
    return epoch_loss


def create_next_frame(model, pre_frame, transform):
    next_frame = model(pre_frame) * 255.0
    image = transform(next_frame[0].cpu().clone())
    return image
