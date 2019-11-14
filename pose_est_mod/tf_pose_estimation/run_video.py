import argparse
import logging
import time

import cv2
import numpy as np
import os
import sys

from .tf_pose.estimator import TfPoseEstimator
from .tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def pose_estimation(input_video_path, output_json_dir, number_people_max=1, frame_first=0):
    #parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    #parser.add_argument('--show-process', action='store_true',
    #                    help='for debug purpose, if enabled, speed for inference is dropped.')

    tensorrt = "false"

    model = "mobilenet_v2_large"
    logger.debug('initialization %s : %s' % (model, get_graph_path(model)))
    w, h = model_wh('432x368')
    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h), trt_bool=str2bool(tensorrt))
    cap = cv2.VideoCapture(input_video_path)

    if cap.isOpened() is False:
        logger.error("Error opening input video stream or file: {0}".format(input_video_path))
        sys.exit(1)

    #width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sys.stdout.write("frame: ")
    frame = 0
    detected = False
    while cap.isOpened():
        ret_val, image = cap.read()
        if not ret_val:
            break
        if frame < frame_first:
            frame += 1
            continue

        sys.stdout.write('\rframe: {:5}'.format(frame))
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
        if len(humans) > 0:
            detected = True
        del humans[number_people_max:]
        image = np.zeros(image.shape)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False, frame=frame, output_json_dir=output_json_dir)
        frame += 1
        #cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        #fps_time = time.time()
        #if cv2.waitKey(1) == 27:
        #    break

    sys.stdout.write("\n")

    if frame <= frame_first:
        logger.error('No frame is processed: frame_first = {0}, frame = {1}'.format(frame_first, frame))
        sys.exit(1)

    if not detected:
        logger.error('No human is detected in the video: {0}'.format(input_video_path))
        sys.exit(1)

logger.debug('finished+')

