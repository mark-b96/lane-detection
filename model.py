import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=(3, 3)),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=(3, 3)),
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=(3, 3)),
            nn.ReLU(True),
            # Dropout layer: Randomly zeroes some of the elements of the input tensor with a probability of 0.8
            nn.Dropout(0.8),
            nn.Conv2d(16, 32, kernel_size=(3, 3)),
            nn.ReLU(True),
            nn.Dropout(0.8),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=(3, 3)),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=(3, 3)),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3)),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 3)),
            nn.ReLU(True),
            nn.Dropout(0.8),
            nn.ConvTranspose2d(16, 16, kernel_size=(3, 3)),
            nn.ReLU(True),
            nn.Dropout(0.8),
            nn.ConvTranspose2d(16, 8, kernel_size=(3, 3)),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, kernel_size=(3, 3))
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
