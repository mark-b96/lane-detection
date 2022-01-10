# lane-detection
Lane detection using a CNN implemented in Pytorch

![Example](data/sample.jpg)

## Installation
```
python3.8 -m pip install -r requirements.txt
```
### Docker
Build the docker image:
```
docker build -t lane-det-img .
```
Run the docker container:
```
sudo docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/mark:/home/mark --hostname $HOSTNAME --name lane-det-container lane-det-img
```

## Usage
```
python3.8 lane_detector.py -i ./data/example.mp4 -o /home/mark/Videos/
```
