import cv2
import numpy as np


class UserInterface:
    @staticmethod
    def get_key():
        return cv2.waitKey(1)

    @staticmethod
    def view_input_img(img):
        np_output = np.squeeze(img.numpy())
        np_output = np.transpose(np_output, (1, 2, 0))
        np_output = cv2.cvtColor(np_output, cv2.COLOR_RGB2BGR)
        cv2.imshow('img', np_output)
        cv2.waitKey(0)
