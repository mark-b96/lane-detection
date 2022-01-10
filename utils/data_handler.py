import os
import cv2
import torch
import onnx
import json
from torch.utils.data import Dataset
import onnxruntime
import torchvision.transforms as transforms


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
        return img, label


class DataHandler:
    def __init__(self, src_video_path, dst_video_dir):
        self.src_video_path: str = src_video_path
        self.dst_video_dir: str = dst_video_dir
        self.train_transform, self.config = None, None

    def set_train_transform(self, image_resolution):
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(image_resolution),
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5)
                )
            ]
        )

    def set_config(self, config):
        self.config = config

    @staticmethod
    def load_json_file(file_path: str):
        with open(file_path, 'r') as f:
            return json.load(f)

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
