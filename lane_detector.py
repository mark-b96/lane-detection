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


def get_arguments() -> argparse.Namespace:
    a = argparse.ArgumentParser()
    a.add_argument(
        '-i',
        type=str,
        help='Path to input video',
        default='/home/mark/Videos/video_2.mp4'
    )
    a.add_argument(
        '-o',
        type=str,
        help='Path to input video',
        default='/home/mark/Videos/'
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

    args = a.parse_args()
    return args


def train_model(data_obj, args):
    training_config = data_obj.config['training']
    model_version = training_config['model_version']
    train_dataset = CustomDataset(
        image_path=args.training_images_dir,
        label_path=args.training_labels_dir,
        transform=data_obj.train_transform
    )

    device = data_obj.get_device()
    cnn_model = CNN().to(device)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True
    )
    model_trainer = ModelTraining(
        model=cnn_model,
        device=device,
        epochs=training_config['epochs'],
        learning_rate=training_config['learning_rate'],
        batch_size=training_config['batch_size'],
        model_version=model_version
    )
    model_trainer.log_params()
    model_trainer.train(training_data=train_loader)

    data_obj.save_torch_model(
        model=cnn_model,
        path=f'{args.weights_dir}/model_{model_version}.pth'
    )

    w, h = args.train_res
    data_obj.save_onnx_model(
        model=cnn_model,
        path=f'{args.weights_dir}/model_{model_version}.onnx',
        dummy_input=torch.randn(1, 3, w, h)
    )


def run_inference(data_obj, args):
    inference_config = data_obj.config['inference']
    model_version = inference_config['model_version']

    ort_session_obj = data_obj.load_onnx_model(
        path=f'{args.weights_dir}/model_v{model_version}.onnx'
    )

    inference_obj = Inference(
        transform=data_obj.train_transform,
    )

    ui_obj = UserInterface()
    video_obj = Video(
        ui_obj=ui_obj,
        data_obj=data_obj,
        skip_frames=1,
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
    should_train = args.train

    data_obj = DataHandler(
        src_video_path=args.i,
        dst_video_dir=args.o
    )
    config_file = data_obj.load_json_file(file_path='./config/config.json')
    data_obj.set_config(config=config_file)
    image_res = tuple(data_obj.config['training']['image_resolution'])
    data_obj.set_train_transform(image_resolution=image_res)

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

