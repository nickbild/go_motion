####
# Nick Bild
# September 2021
# Go Motion
# https://github.com/nickbild/go_motion
#
# Create a stop motion video automatically using machine learning
# by automatically selecting only image frames of a scene after they are
# done being manipulated by human hands.
#
# Hand detection based on examples from:
# https://github.com/NVIDIA-AI-IOT/trt_pose_hand.
####

import json
import cv2
import trt_pose.coco
import math
import os
import numpy as np
import traitlets
import trt_pose.models
import torch
from torch2trt import TRTModule
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import torchvision.transforms as transforms
import PIL.Image
from preprocessdata import preprocessdata
from jetcam.csi_camera import CSICamera
from jetcam.utils import bgr8_to_jpeg
from IPython.display import display
from PIL import Image


NEED_NEXT_FRAME = False
EMPTY_FRAMES = 0
EMPTY_FRAMES_THRESHOLD = 5
SEQ_NO = 0
WIDTH = 224
HEIGHT = 224


def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


def execute(change):
    global NEED_NEXT_FRAME
    global EMPTY_FRAMES
    global EMPTY_FRAMES_THRESHOLD

    image = change['new']
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)
    
    hand_detected = False
    joints = preprocessdata.joints_inference(image, counts, objects, peaks)
    for joint in joints:
        for point in joint:
            if point > 0:
                hand_detected = True
                NEED_NEXT_FRAME = True
                EMPTY_FRAMES = 0
                break
    
    # print("{0}: {1}".format(SEQ_NO, hand_detected))

    # This triggers after a hand has been seen,
    # then removed from the frame.
    if not hand_detected and NEED_NEXT_FRAME:
        EMPTY_FRAMES += 1
    # Wait for EMPTY_FRAMES_THRESHOLD frames meeting the above condition
    # to occur to avoid false triggers.
    if EMPTY_FRAMES >= EMPTY_FRAMES_THRESHOLD:
        im = Image.fromarray(image)
        im.save("img/filter/img_{0}.jpg".format(SEQ_NO))
        NEED_NEXT_FRAME = False
        EMPTY_FRAMES = 0


with open('preprocess/hand_pose.json', 'r') as f:
    hand_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(hand_pose)
num_parts = len(hand_pose['keypoints'])
num_links = len(hand_pose['skeleton'])
model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

if not os.path.exists('model/hand_pose_resnet18_att_244_244_trt.pth'):
    MODEL_WEIGHTS = 'model/hand_pose_resnet18_att_244_244.pth'
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    import torch2trt
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    OPTIMIZED_MODEL = 'model/hand_pose_resnet18_att_244_244_trt.pth'
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

OPTIMIZED_MODEL = 'model/hand_pose_resnet18_att_244_244_trt.pth'
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

parse_objects = ParseObjects(topology,cmap_threshold=0.15, link_threshold=0.15)
draw_objects = DrawObjects(topology)

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

preprocessdata = preprocessdata(topology, num_parts)

camera = CSICamera(width=1280, height=720, capture_fps=30)
camera.running = True

while True:
    orig = camera.value
    resize = cv2.resize(orig, (WIDTH, HEIGHT))

    im = Image.fromarray(orig)
    im.save("img/full/img_{0}.jpg".format(SEQ_NO))
    
    execute({'new': resize})
    SEQ_NO += 1
