import cv2
import numpy as np


class Visualiser:
    def __init__(self, frame: np.ndarray, frame_width: int, frame_height: int):
        self.frame = frame
        self.frame_width = frame_width
        self.frame_height = frame_height

    def draw_lane(self, predicted_output: np.ndarray):
        grey = cv2.cvtColor(predicted_output, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(grey, 128, 255, cv2.THRESH_BINARY)
        white = np.where(mask == 255)
        blank = np.zeros((self.frame_height, self.frame_width, 3)).astype(np.uint8)
        blank[white] = (0, 255, 0)
        alpha = 0.5
        self.frame[white] = cv2.addWeighted(self.frame, alpha, blank, 1-alpha, 0)[white]
