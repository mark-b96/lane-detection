import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class ModelTraining:
    def __init__(self, model, device, epochs, learning_rate, batch_size, model_version):
        self.model = model.to(device)
        self.device = device
        self.epochs: int = epochs
        self.learning_rate: float = learning_rate
        self.batch_size: int = batch_size
        self.model_version: str = model_version

        self.criterion = nn.MSELoss()
        self.optimiser = torch.optim.Adam(
                                params=self.model.parameters(),
                                lr=self.learning_rate
                            )
        self.writer = SummaryWriter(
            f'runs/lane_det_{self.model_version}'
        )
        self.running_loss: int = 0

    def train(self, training_data):
        dataset_len = len(training_data)
        for epoch in range(self.epochs):
            for index, (images, labels) in enumerate(training_data):

                images = images.to(self.device)
                labels = labels.to(self.device)
                predictions = self.model(images)

                loss = self.criterion(predictions, labels)
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                self.running_loss += loss.item()
                if (index+1) % 100 == 0:
                    self.writer.add_scalar(
                        tag='training_loss',
                        scalar_value=loss.item(),
                        global_step=(epoch*dataset_len) + index
                    )
                    self.running_loss = 0
