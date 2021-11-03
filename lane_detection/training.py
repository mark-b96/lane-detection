import torch
import torch.nn as nn


class ModelTraining:
    def __init__(self, model, criterion, device, epochs: int,
                 learning_rate: float, batch_size: int):
        self.device = device
        self.criterion = criterion
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = model.to(device)
        self.optimiser = torch.optim.Adam(
                                params=self.model.parameters(),
                                lr=self.learning_rate
                            )

    def train(self, training_data):
        for epoch in range(self.epochs):
            for images, labels in training_data:

                images = images.to(self.device)
                labels = labels.to(self.device)
                predictions = self.model(images)

                loss = self.criterion(predictions, labels)
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                print(f'loss: {loss.item()}')
