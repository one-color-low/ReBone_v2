#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# vmdlifting.py - estimate 3D pose by "Lifting-from-the-Deep", and convert the pose data to VMD
#
# This program is derived from demo.py in Lifting-from-the-Deep which is created by Denis Tome'

from __future__ import print_function

def usage(prog):
    print('usage: ' + prog + ' IMAGE_FILE VMD_FILE')
    sys.exit()

#import __init__

from lifting import PoseEstimator
from lifting.utils import draw_limbs
from lifting.utils import plot_pose

import cv2
import matplotlib.pyplot as plt
from os.path import dirname, realpath
from .pos2vmd import positions_to_frames, make_showik_frames, convert_position
from .VmdWriter import VmdWriter
from .refine_position import refine_position
from .adjust_center import adjust_center
from .dump_positions import dump_positions
import argparse;

DIR_PATH = dirname(realpath(__file__))
PROJECT_PATH = realpath(DIR_PATH + '/..')
SAVED_SESSIONS_DIR = PROJECT_PATH + '/data/saved_sessions'
SESSION_PATH = SAVED_SESSIONS_DIR + '/init_session/init'
PROB_MODEL_PATH = SAVED_SESSIONS_DIR + '/prob_model/prob_model_params.mat'

def vmdlifting(image_file, vmd_file, center_enabled=False):
    image_file_path = realpath(image_file)
    cap = cv2.VideoCapture(image_file_path)
    initialized = False
    positions_list = []
    head_rotation_list = []
    visibility_list = []
    frame_num = 0
    print("pose estimation start")
    while (cap.isOpened()):
        ret, image = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # conversion to rgb
        #debug_img = "debug/" + str(frame_num) + ".png"
        #cv2.imwrite(debug_img, image)

        # create pose estimator
        image_size = image.shape

        if not initialized:
            pose_estimator = PoseEstimator(image_size, SESSION_PATH, PROB_MODEL_PATH)
            # load model
            pose_estimator.initialise()
            initialized = True

        # pose estimation
        try:
            pose_2d, visibility, pose_3d = pose_estimator.estimate(image)
        except Exception as ex:
            frame_num +=1
            continue

        if pose_2d is None or visibility is None or pose_3d is None:
            frame_num += 1
            continue

        dump_positions(pose_2d, visibility, pose_3d)
        positions = convert_position(pose_3d)
        #print(positions)
        adjust_center(pose_2d, positions, image)
        #print(positions)
        positions_list.append(positions)
        visibility_list.append(visibility[0])
        print("frame_num: ", frame_num)
        frame_num += 1

    # close model
    pose_estimator.close()

    refine_position(positions_list)

    bone_frames = []
    frame_num = 0
    for positions, visibility in zip(positions_list, visibility_list):
        if positions is None:
            frame_num += 1
            continue
        bf = positions_to_frames(positions, visibility, frame_num, center_enabled)
        bone_frames.extend(bf)
        frame_num += 1

    showik_frames = make_showik_frames()
    writer = VmdWriter()
    writer.write_vmd_file(vmd_file, bone_frames, showik_frames)


def display_results(in_image, data_2d, joint_visibility, data_3d):
    """Plot 2D and 3D poses for each of the people in the image."""
    plt.figure()
    draw_limbs(in_image, data_2d, joint_visibility)
    plt.imshow(in_image)
    plt.axis('off')

    # Show 3D poses
    for single_3D in data_3d:
        # or plot_pose(Prob3dPose.centre_all(single_3D))
        plot_pose(single_3D)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='estimate 3D pose and generate VMD motion')
    parser.add_argument('--center', action='store_true', help='move center bone (experimental)')
    parser.add_argument('IMAGE_FILE')
    parser.add_argument('VMD_FILE')

    arg = parser.parse_args()
    vmdlifting(arg.IMAGE_FILE, arg.VMD_FILE, arg.center)
