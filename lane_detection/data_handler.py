import os
import cv2
import torch
import onnx
from torch.utils.data import Dataset
import onnxruntime
import numpy as np
# cv2.namedWindow('img', cv2.WINDOW_GUI_NORMAL)


class CustomDataset(Dataset):
    def __init__(self, image_path, label_path, transform=None):
        self.image_path = image_path
        self.label_path = label_path
        self.transform = transform
        self.image_paths = sorted(os.listdir(self.image_path))
        self.label_paths = sorted(os.listdir(self.label_path))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = cv2.imread(f'{self.image_path}/{self.image_paths[index]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = cv2.imread(f'{self.label_path}/{self.label_paths[index]}')
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)

        # np_output = np.squeeze(img.numpy())
        # np_output = np.transpose(np_output, (1, 2, 0))
        # np_output = cv2.cvtColor(np_output, cv2.COLOR_RGB2BGR)
        # cv2.imshow('img', np_output)

        # cv2.imshow('label', label)
        cv2.waitKey(0)
        return img, label


class DataHandler:
    @staticmethod
    def save_torch_model(model, path: str):
        torch.save(model.state_dict(), path)

    @staticmethod
    def load_torch_model(model, path: str):
        model.load_state_dict(torch.load(path))
        model.eval()
        return model

    @staticmethod
    def save_onnx_model(model, path: str, dummy_input):
        torch.onnx.export(model, dummy_input, path, opset_version=11)

    @staticmethod
    def load_onnx_model(path: str):
        model = onnx.load(path)
        onnx.checker.check_model(model)
        onnx.helper.printable_graph(model.graph)
        ort_session = onnxruntime.InferenceSession(path)
        return ort_session
