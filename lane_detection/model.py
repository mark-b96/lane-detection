import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=(5, 5)),
            nn.ReLU(True),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=(5, 5)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=(5, 5)),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            # nn.Dropout(0.8),
            nn.Conv2d(16, 32, kernel_size=(5, 5)),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 2)),
            # nn.Dropout(0.8),
            nn.Conv2d(32, 64, kernel_size=(5, 5)),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(5, 5)),
            nn.MaxPool2d((2, 2)),
        )
        self.decoder = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ConvTranspose2d(64, 64, kernel_size=(5, 5)),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=(5, 5)),
            nn.ReLU(True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ConvTranspose2d(32, 16, kernel_size=(5, 5)),
            nn.ReLU(True),
            # nn.Dropout(0.8),
            nn.ConvTranspose2d(16, 16, kernel_size=(5, 5)),
            nn.ReLU(True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            # nn.Dropout(0.8),
            nn.ConvTranspose2d(16, 8, kernel_size=(5, 5)),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, kernel_size=(5, 5)),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
