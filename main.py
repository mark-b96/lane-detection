import os
import cv2
import torch
import argparse
import torch.nn as nn
import torchvision.transforms as transforms

from model import CNN
from training import ModelTraining
from inference import Inference
from data_handler import CustomDataset, DataHandler
from video import Video
from user_interface import UserInterface
from visualisation import Visualiser
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_arguments() -> argparse.Namespace:
    a = argparse.ArgumentParser()
    a.add_argument(
        '-i',
        type=str,
        help='Path to input video',
        default='/home/mark/Videos/highway_china_45_55.mp4'
    )
    a.add_argument(
        '--train',
        type=bool,
        help='Run in training mode',
        default=False
    )
    a.add_argument(
        '--weights',
        type=str,
        default='/home/mark/Documents/ml_data/lane_detection/'
    )
    a.add_argument(
        '--batch_size',
        type=int,
        help='Batch size used for model training',
        default=8
    )

    args = a.parse_args()
    return args


def train(model, train_dataset, criterion, batch_size: int):
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model_trainer = ModelTraining(
        model=model,
        device=device,
        criterion=criterion,
        epochs=5,
        learning_rate=0.001,
        batch_size=batch_size,
    )

    model_trainer.train(training_data=train_loader)


def evaluate(model, test_dataset, criterion, batch_size: int):
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for image, label in test_loader:
            pred = model(image)
            test_loss += criterion(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def process_frame(video, inference, frame, ort_session):
    # video.display_frame('input', frame)
    predicted_image = inference.predict_onnx(
        image=frame,
        ort_session=ort_session
    )

    frame_width, frame_height = video.resolution
    vis = Visualiser(
        frame=frame,
        frame_width=frame_width,
        frame_height=frame_height
    )
    predicted_image = cv2.resize(predicted_image, video.resolution)
    vis.draw_lane(predicted_output=predicted_image)
    video.display_frame('output', frame)
    video.write_frame(frame)


def main():
    args = get_arguments()
    batch_size = args.batch_size  # Adjust depending on amount RAM
    image_resolution = (80, 80)
    video_path = args.i
    video_root = os.path.dirname(video_path)
    video_name, video_format = video_path.split('/')[-1].split('.')
    input_video = f'{video_root}/{video_name}.{video_format}'
    models_root = args.weights
    should_train = args.train

    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(image_resolution),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    model_data = DataHandler()

    if should_train:
        train_dataset = CustomDataset(
            image_path='./data/custom/train/images',
            label_path='./data/custom/train/labels',
            transform=train_transform
        )

        test_dataset = CustomDataset(
            image_path='./data/custom/test/images',
            label_path='./data/custom/test/labels',
            transform=train_transform
        )

        cnn_model = CNN().to(device)
        criterion = nn.MSELoss()

        train(
            model=cnn_model,
            train_dataset=train_dataset,
            criterion=criterion,
            batch_size=batch_size,
        )

        model_data.save_torch_model(
            model=cnn_model,
            path='./data/weights/model_v4.pth'
        )

        model_data.save_onnx_model(
            model=cnn_model,
            path='./data/weights/model_v4.onnx',
            dummy_input=torch.randn(1, 3, image_resolution[0], image_resolution[1])
        )

    ort_session = model_data.load_onnx_model(
        path=f'{models_root}weights/model_v4.onnx'
    )

    inference = Inference(
        transform=train_transform,
    )

    ui = UserInterface()
    video = Video(
        _input_video=input_video,
        _video_name=video_name,
        _skip_frames=1,
        _ui=ui
    )

    video_frames = video.get_frame(skip_frames=1)
    video.create_window('output')

    for frame in video_frames:
        process_frame(
            video=video,
            inference=inference,
            frame=frame,
            ort_session=ort_session
        )


if __name__ == '__main__':
    main()

