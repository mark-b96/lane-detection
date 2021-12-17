import cv2


class FrameProcessor:
    def __init__(self, visualiser_obj, video_obj, inference_obj, ort_session_obj):
        self.visualiser_obj = visualiser_obj
        self.video_obj = video_obj
        self.inference_obj = inference_obj
        self.ort_session_obj = ort_session_obj

    def process_frame(self):
        predicted_image = self.inference_obj.predict_onnx(
            image=self.visualiser_obj.frame,
            ort_session=self.ort_session_obj
        )

        predicted_image = cv2.resize(predicted_image, self.video_obj.resolution)
        self.visualiser_obj.draw_lane(predicted_output=predicted_image)
        self.video_obj.display_frame('output', self.visualiser_obj.frame)
        self.video_obj.write_frame(self.visualiser_obj.frame)
