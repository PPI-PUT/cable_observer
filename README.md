# DLO Tracking
## Description
This repository is an implementation of the DLO tracking algorithm, which given a mask of the DLO,
returns the B-spline which approximates the shape of the mask.
Presented method is able to run in real-time.

## Usage
### Option 1: Using python3 
* Clone repository
  ```
  git clone https://github.com/pkicki/cable_observer
  ```
* Download videos from [here](https://drive.google.com/drive/folders/1taSxE8XdUbhhhGJnhtnSJg1tgyOTRgRI?usp=sharing) and put them into `cable_observer/videos/` directory
* Install necessary dependencies
  ```
  pip3 install -r requirements.txt
  ```
* Make sure the file `cable_observer.py` is executable
  ```
  chmod +x cable_observer.py
  ```
* Run `cable_observer.py` for tracking on videos
  ```
  ./cable_observer.py --path /path/to/video_file --debug
  ```

###Option 2: Using ROS
* Clone repository into src directory in the ROS workspace
  ```
  git clone https://github.com/pkicki/cable_observer
  ```
* Download videos from [here](https://drive.google.com/drive/folders/1taSxE8XdUbhhhGJnhtnSJg1tgyOTRgRI?usp=sharing) and put them into `cable_observer/videos/` directory
* Install necessary dependencies
  ```
  rosdep install --from-path src/cable_observer
  ```
* Make sure the file `scripts/cable_observer_node.py` is executable
  ```
  chmod +x scripts/cable_observer_node.py
  ```
* Run `cable_observer.launch` for tracking on videos
  ```
  roslaunch cable_observer cable_observer.launch path:=/path/to/video_file debug:=true
  ```
* You may check provided topics `/spline/coeffs` and `/spline/coords` using `rostopic echo /topic/path`

### Available arguments:
  * `path` - path to video file
  * `debug` - run debug mode, default false
  * `camera` - set camera as video input, default false
  * `input` - specify camera number (corresponding to /dev/videoX)

![Screenshot](example.png)
