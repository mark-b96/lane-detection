import cv2
import math
import os
import numpy as np


class Video:
    def __init__(self, _input_video: str, _video_name: str, _skip_frames: int, _ui):
        self.ui = _ui
        self.input_video = _input_video
        self.video_name = _video_name
        self.skip_frames = _skip_frames
        self.video_capture = cv2.VideoCapture(self.input_video)
        self.fps = math.ceil(self.video_capture.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resolution = (self.frame_width, self.frame_height)
        data_folder = os.path.dirname(self.input_video)
        output_video_path = f"{data_folder}/{self.video_name}_lane.mp4"
        self.is_playing = True

        self.video_writer = cv2.VideoWriter(
            filename=output_video_path,
            fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
            fps=self.fps,
            frameSize=(self.frame_width, self.frame_height)
        )

    def __del__(self):
        self.video_capture.release()

    @staticmethod
    def create_window(window_name: str):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def get_frame(self, skip_frames: int) -> np.ndarray:
        frame_index = 0

        while self.video_capture.isOpened():
            key = self.ui.get_key()
            if key == ord('p'):
                self.is_playing = not self.is_playing
            if self.is_playing or key == ord(' '):
                frame_exists, frame = self.video_capture.read()

                if not frame_exists:
                    break
                if frame_index % skip_frames == 0:
                    yield frame

                frame_index += 1
        cv2.destroyAllWindows()
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def write_frame(self, frame: np.ndarray):
        self.video_writer.write(frame)

    @staticmethod
    def display_frame(window_name: str, frame: np.ndarray):
        cv2.imshow(window_name, frame)
