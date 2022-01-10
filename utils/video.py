import cv2
import math
from pathlib import Path
import numpy as np


class Video:
    def __init__(self, ui_obj, data_obj, skip_frames):
        self.ui_obj = ui_obj
        self.data_obj = data_obj
        self.input_video: str = self.data_obj.src_video_path
        self.video_name: str = Path(self.input_video).stem
        self.skip_frames: int = skip_frames

        self.video_capture = cv2.VideoCapture(self.input_video)
        self.fps = math.ceil(self.video_capture.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.resolution = (self.frame_width, self.frame_height)
        self.inference_config = self.data_obj.config['inference']

        output_video_path = f"{self.data_obj.dst_video_dir}/{self.video_name}_lane.mp4"

        self.is_playing: bool = True
        self.frame_index: int = 0

        self.video_writer = cv2.VideoWriter(
            filename=output_video_path,
            fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
            fps=self.fps,
            frameSize=self.resolution
        )

    def __del__(self):
        self.video_capture.release()

    def create_window(self, window_name: str):
        if not self.inference_config['display_output']:
            return
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def get_frame(self, skip_frames: int) -> np.ndarray:
        while self.video_capture.isOpened():
            key = self.ui_obj.get_key()
            if key == ord('p'):
                self.is_playing = not self.is_playing
            if self.is_playing or key == ord(' '):
                frame_exists, frame = self.video_capture.read()

                if not frame_exists:
                    break
                if self.frame_index % skip_frames == 0:
                    yield frame

                self.frame_index += 1
        cv2.destroyAllWindows()

    def write_frame(self, frame: np.ndarray):
        if not self.inference_config['create_video']:
            return
        self.video_writer.write(frame)

    def display_frame(self, window_name: str, frame: np.ndarray):
        if not self.inference_config['display_output']:
            return
        cv2.imshow(window_name, frame)
