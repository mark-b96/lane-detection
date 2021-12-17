import os
import torch
import argparse

from lane_detection.model import CNN
from lane_detection.training import ModelTraining
from lane_detection.inference import Inference

from utils.data_handler import CustomDataset, DataHandler
from utils.video import Video
from utils.user_interface import UserInterface
from utils.visualisation import Visualiser
from utils.frame_processor import FrameProcessor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_arguments() -> argparse.Namespace:
    a = argparse.ArgumentParser()
    a.add_argument(
        '-i',
        type=str,
        help='Path to input video',
        default='/home/mark/Videos/video_2.mp4'
    )
    a.add_argument(
        '--train',
        type=bool,
        help='Run in training mode',
        default=False
    )
    a.add_argument(
        '--weights_dir',
        type=str,
        default='/home/mark/Documents/ml_data/lane_detection/weights'
    )
    a.add_argument(
        '--training_images_dir',
        type=str,
        default='/home/mark/Documents/ml_data/lane_detection/BDD/train/images'
    )
    a.add_argument(
        '--training_labels_dir',
        type=str,
        default='/home/mark/Documents/ml_data/lane_detection/BDD/train/labels'
    )
    a.add_argument(
        '--test_images_dir',
        type=str,
        default='/home/mark/Documents/ml_data/lane_detection/BDD/train/images'
    )
    a.add_argument(
        '--test_labels_dir',
        type=str,
        default='/home/mark/Documents/ml_data/lane_detection/BDD/test/labels'
    )
    a.add_argument(
        '--batch_size',
        type=int,
        help='Batch size used for model training',
        default=8
    )
    a.add_argument(
        '--train_res',
        type=tuple,
        default=(80, 80)
    )

    args = a.parse_args()
    return args


def train_model(data_obj, args):
    model_version = 'v25'
    train_dataset = CustomDataset(
        image_path=args.training_images_dir,
        label_path=args.training_labels_dir,
        transform=data_obj.train_transform
    )

    # imgs = torch.stack([img_t for img_t, _ in train_dataset], dim=3)
    #
    # # test_dataset = CustomDataset(
    # #     image_path=f'{models_root}/custom/test/images',
    # #     label_path=f'{models_root}/custom/test/labels',
    # #     transform=train_transform
    # # )
    # t_mean = imgs.view(3, -1).mean(dim=1)
    # t_std = imgs.view(3, -1).std(dim=1)
    cnn_model = CNN().to(device)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    model_trainer = ModelTraining(
        model=cnn_model,
        device=device,
        epochs=5,
        learning_rate=0.001,
        batch_size=args.batch_size,
        model_version=model_version
    )

    model_trainer.train(training_data=train_loader)

    data_obj.save_torch_model(
        model=cnn_model,
        path=f'{args.weights_path}/model_{model_version}.pth'
    )

    w, h = args.train_res
    data_obj.save_onnx_model(
        model=cnn_model,
        path=f'{args.weights_path}/model_{model_version}.onnx',
        dummy_input=torch.randn(1, 3, w, h)
    )


def run_inference(data_obj, args):
    video_path = args.i
    video_root = os.path.dirname(video_path)
    video_name, video_format = video_path.split('/')[-1].split('.')
    input_video = f'{video_root}/{video_name}.{video_format}'

    ort_session_obj = data_obj.load_onnx_model(
        path=f'{args.weights_dir}/model_v25.onnx'
    )

    inference_obj = Inference(
        transform=data_obj.train_transform,
    )

    ui_obj = UserInterface()
    video_obj = Video(
        input_video_path=input_video,
        video_name=video_name,
        skip_frames=1,
        ui=ui_obj
    )

    video_frames = video_obj.get_frame(skip_frames=1)
    video_obj.create_window('output')

    visualiser_obj = Visualiser(
        frame_width=video_obj.frame_width,
        frame_height=video_obj.frame_height
    )

    frame_proc_obj = FrameProcessor(
        visualiser_obj=visualiser_obj,
        video_obj=video_obj,
        inference_obj=inference_obj,
        ort_session_obj=ort_session_obj
    )

    for frame in video_frames:
        visualiser_obj.set_frame(frame=frame)
        frame_proc_obj.process_frame()


def main():
    args = get_arguments()
    image_resolution = args.train_res
    should_train = args.train

    data_obj = DataHandler()
    data_obj.set_train_transform(image_resolution=image_resolution)

    if should_train:
        train_model(
            data_obj=data_obj,
            args=args,
        )
    else:
        run_inference(
            data_obj=data_obj,
            args=args,
        )


if __name__ == '__main__':
    main()

