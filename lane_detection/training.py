import torch
import torch.nn as nn
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from mlflow import log_metric, log_param


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
            f'./runs/lane_det_{self.model_version}'
        )
        self.running_loss: int = 0

    def train(self, training_data):
        dataset_len = len(training_data)
        logger.info('Starting training...')
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
                    log_metric("training_loss", loss.item())
                    self.writer.add_scalar(
                        tag='training_loss',
                        scalar_value=loss.item(),
                        global_step=(epoch*dataset_len) + index
                    )
                    logger.info(f'Loss: {loss.item()}')
                    self.running_loss = 0
        logger.info('Training complete!')

    def log_params(self):
        log_param("model_version", self.model_version)
        log_param("epochs", self.epochs)
        log_param("learning_rate", self.learning_rate)
        log_param("batch_size", self.batch_size)
        log_param("device", self.device)
