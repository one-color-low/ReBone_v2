import datetime
from os.path import dirname, join, basename, abspath
from os import makedirs
import argparse
import glob
import shutil

from .tf_pose_estimation.tfpose import pose_estimation
from .FCRN_DepthPrediction_vmd.depth_pred import depth_pred
from .d3_pose_baseline_vmd.src.d3pose_vmd import baseline
from .VMD_3d_pose_baseline_multi.pos2vmd_main import position2vmd

def video2vmd(input_video_path, output_file_path):
    now_str = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    now_dir = abspath(dirname(__file__))
    json_dir = join(now_dir, "json")
    makedirs(json_dir,exist_ok=True)
    #depth_path = '{0}/{1}_{2}_depth'.format(dirname(json_dir), basename(json_dir), now_str)
    person_path = '{0}/{1}_{2}_idx01'.format(dirname(json_dir), basename(json_dir), now_str)

    pose_estimation(input_video_path, json_dir)
    depth_pred(now_str, input_video_path, json_dir)
    baseline(now_str, person_path)
    position2vmd(now_str, person_path, output_file_path)
    json_files = glob.glob(join(now_dir,"json*"))
    for dir_name in json_files:
        shutil.rmtree(dir_name)

if __name__ == '__main__':
    video2vmd("./input.mp4", "/test.vmd")