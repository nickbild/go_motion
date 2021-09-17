# Go Motion

Simplify stop motion animation with machine learning.

<p align="center">
<img src="https://raw.githubusercontent.com/nickbild/go_motion/main/media/teaser.gif">
</p>

## How It Works

A CSI camera is connected to a Jetson Xavier NX.  This camera continually captures images of a scene.  Using the [trt_pose_hand](https://github.com/NVIDIA-AI-IOT/trt_pose_hand) hand pose detection model, the Jetson is able to determine when a hand is in the image frame.

Each time a hand leaves the frame, a single image is saved as part of the stop motion sequence.  In this way, it is possible to continually manipulate the scene, momentarily removing one's hands from view of the camera after each adjustment, and have a stop motion sequence automatically generated that contains only the relevant image frames.

## Media

YouTube:  
https://www.youtube.com/watch?v=zxzlnXLueIg

Full Setup:

![](https://raw.githubusercontent.com/nickbild/go_motion/main/media/full_setup_sm.jpg)

Jetson Xavier NX:

![](https://raw.githubusercontent.com/nickbild/go_motion/main/media/jetson_nx_sm.jpg)

## Bill of Materials

- 1 x NVIDIA Jetson Xavier NX
- 1 x Raspberry Pi Camera v2

## About the Author

[Nick A. Bild, MS](https://nickbild79.firebaseapp.com/#!/)
