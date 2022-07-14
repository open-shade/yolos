# YOLOS-ROS2 Wrapper

This is a ROS2 wrapper for the You Only Look at One Sequence (via Vision Transformers for Object Detection), [YOLOS](https://github.com/hustvl/YOLOS). We utilize `huggingface` and the `transformers` for the [source of the algorithm](https://huggingface.co/hustvl/yolos-small). The main idea is for this container to act as a standalone interface and node, removing the necessity to integrate separate packages and solve numerous dependency issues.

The algorithm utilizes of the vanilla ViT pre-trained on mid-sized ImageNet-1k to the more challenging COCO object detection benchmark.

# Installation Guide


## Using Docker Pull
1. Install [Docker](https://www.docker.com/) and ensure the Docker daemon is running in the background.
2. Run ```docker pull shaderobotics/yolos:${ROS2_DISTRO}-${MODEL_VERSION}```
3. Follow the run commands in the usage section below

## Build Docker Image Natively
1. Install [Docker](https://www.docker.com/) and ensure the Docker daemon is running in the background.
2. Clone this repo with ```git pull https://github.com/open-shade/yolos.git```
3. Enter the repo with ```cd yolos```
4. To pick a specific model version, edit the `ALGO_VERSION` constant in `/yolos/yolos.py`
5. Build the container with ```docker build . -t [name]```. This will take a while. We have also provided associated `cloudbuild.sh` scripts to build on GCP all of the associated versions.
6. Follow the run commands in the usage section below.

# Model Versions

* ```tiny```
* ```base```
* ```small```

More information about these versions can be found in the [paper](https://arxiv.org/abs/2106.00666). `tiny`, `base`, `small`, represent the number of weights stored (i.e. the size of the model).

## Example Docker Command

```bash
docker pull shaderobotics/yolos:foxy-small
```

# Usage
## Run the YOLOS Node 
Run ```docker run -t --net=host shaderobotics/yolos:${ROS_DISTRO}-${MODEL_VERSION}```. Your node should be running now. Then, by running ```ros2 topic list,``` you should see all the possible pub and sub routes.

For more details explaining how to run Docker images, visit the official Docker documentation [here](https://docs.docker.com/engine/reference/run/). Also, additional information as to how ROS2 communicates between external environment or multiple docker containers, visit the official ROS2 docs [here](https://docs.ros.org/en/foxy/How-To-Guides/Run-2-nodes-in-single-or-separate-docker-containers.html#). 

# Parameters
This wrapper utilizes 4 optional parameters to modify the data coming out of the published topics as well as the dataset YOLOS utilizes for comparison. Most parameters can be modified during runtime. However, if you wish to use your own dataset, you must pass that parameter in before runtime. If you are unsure how to pass or update parameters before or during runtime, visit the official ROS2 docs [here](https://docs.ros.org/en/foxy/Concepts/About-ROS-2-Parameters.html?highlight=parameters#setting-initial-parameter-values-when-running-a-node).

The supported, *optional* parameters are...

| Name        | Type    | Default | Use                                                                 |
|-------------|---------|---------|---------------------------------------------------------------------|
| pub_image   | Boolean | True   | Enable or disable the pub of the processed image (with bounding boxes)                |
| pub_boxes    | Boolean | True   | Enable or disable the pub of the bounding boxes as a Detection2DArray           |
| pub_detections   | Boolean | True    | Enable or disable the publishing of detections (whether or not to send back a string with all detections found)   |    

You __do not__ need to specify any parameters, unless you wish to modify the defaults.

# Topics

| Name                   | IO  | Type                             | Use                                                               |
|------------------------|-----|----------------------------------|-------------------------------------------------------------------|
| yolos/image_raw       | sub | [sensor_msgs.msg.Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html)            | Takes the raw camera output to be processed                       |
 | yolos/image           | pub | [sensor_msgs.msg.Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html)            | Outputs the processed image with bounding boxes drawn on the image |
| yolos/detections            | pub | [std_msgs.msg.String](http://docs.ros.org/en/api/std_msgs/html/msg/String.html)              | Outputs all detected classes in the image |
| yolos/detection_boxes | pub | [vision_msgs.msg.Detection2DArray](http://docs.ros.org/en/lunar/api/vision_msgs/html/msg/Detection2DArray.html) | Outputs the detected bounding box location in a unified format    |

# Testing / Demo
To test and ensure that this package is properly installed, replace the Dockerfile in the root of this repo with what exists in the demo folder. Installed in the demo image contains a [camera stream emulator](https://github.com/klintan/ros2_video_streamer) by [klintan](https://github.com/klintan) which directly pubs images to the YOLOS node and processes it for you to observe the outputs.

To run this, run ```docker build . -t --net=host [name]```, then ```docker run -t [name]```. Observing the logs for this will show you what is occuring within the container. If you wish to enter the running container and preform other activities, run ```docker ps```, find the id of the running container, then run ```docker exec -it [containerId] /bin/bash```