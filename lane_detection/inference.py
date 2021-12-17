import cv2
import torch
import numpy as np


class Inference:
    def __init__(self, transform, model=None):
        self.transform = transform
        self.model = model

    def predict(self, image: np.ndarray):
        test_image = self.transform(image)
        np_output = np.squeeze(test_image.numpy())
        np_output = np.transpose(np_output, (1, 2, 0))
        np_output = cv2.cvtColor(np_output, cv2.COLOR_RGB2BGR)
        test_image = test_image.reshape((1, 3, 80, 80))

        with torch.no_grad():
            output = self.model(test_image)
            np_output = np.squeeze(output.numpy())
            np_output = np.transpose(np_output, (1, 2, 0))
            mean = np.array([0.5, 0.5, 0.5])
            std = np.array([0.5, 0.5, 0.5])
            np_output = ((np_output * std) + mean).clip(0, 1)
            np_output = (np_output*255).astype(np.uint8)

    def predict_onnx(self, image: np.ndarray, ort_session) -> np.ndarray:
        test_image = self.transform(image)
        np_output = np.squeeze(test_image.numpy())
        input_name = ort_session.get_inputs()[0].name
        np_tensor_input = torch.from_numpy(np_output)
        np_tensor_input.unsqueeze_(0)
        np_input = np_tensor_input.numpy()
        # compute ONNX Runtime output prediction
        ort_inputs = {input_name: np_input}
        ort_outs = ort_session.run(None, ort_inputs)

        np_output = np.squeeze(ort_outs[0])
        np_output = np.transpose(np_output, (1, 2, 0))
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        np_output = ((np_output * std) + mean).clip(0, 1)
        np_output = (np_output * 255).astype(np.uint8)
        np_output = cv2.cvtColor(np_output, cv2.COLOR_RGB2BGR)
        return np_output

