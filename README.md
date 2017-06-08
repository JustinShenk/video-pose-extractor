# get-humans
Dockerfile and instructions for human pose estimation implementation using Caffe, OpenCV 3.1.0 and Python 2.7.

Dockerized, minimal implementation of Cao et al.'s [repository](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation). 

Includes:
- CPU Caffe implementation
- python bindings + OpenCV for happy hacking
- easy-to-use video-to-images script for pose finding in videos
- dumps limb data into a file for later analysis and modeling

## Getting started

[Install Docker](https://www.docker.com/community-edition#/download)

From the command line:

`docker pull justinshenk/get-humans`

or build it locally

```
git clone https://github.com/justinshenk/get-humans.git
cd get-humans
docker build -t justinshenk/get-humans:cpu -f .
```

### Image ###

`python convert.py -i [image]`

### Video ### 

`python convert.py -i [input_file] -s [start_time] -t [duration] -r [frame_rate]`

Note: Time is in `hh:mm:ss` format.

Data is saved in `positions.npy` and can be loaded using

```
ipython
import numpy as np
raw = np.load('positions.npy')
data = raw.take(0)
```
