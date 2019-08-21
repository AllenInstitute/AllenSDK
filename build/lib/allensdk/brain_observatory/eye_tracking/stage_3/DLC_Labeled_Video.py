import time
t0 = time.time()
import tensorflow as tf
import os
os.environ["DLClight"]="True"
import deeplabcut

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import *
import numpy as np
import collections
import pandas as pd
import sys
import re
import argparse
import logging


ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger = logging.getLogger('dlc-eye-tracking')
logger.setLevel(logging.INFO)
logger.addHandler(ch)
logger.propagate = False

parser = argparse.ArgumentParser()
parser.add_argument("--video_input_file", type=str, required=True, help="path to video file, mp4 or avi")
parser.add_argument("--ellipse_output_video_file", type=str, required=False, help="Create ellipse video file")
parser.add_argument("--points_output_video_file", type=str, required=False, help="Create ellipse video file")
args = parser.parse_args()

video_file_path = args.video_input_file
bucket_data_blobname = video_file_path[:-4] + 'DeepCut_resnet50_universal_eye_trackingJul10shuffle1_1030000_labeled.mp4' 
output_data_file = '/workdir/{}'.format(bucket_data_blobname)

from google.cloud import storage
client = storage.Client()
src_bucket = client.get_bucket('brain-observatory-eye-videos')
tgt_bucket = client.get_bucket('dlc-labeled-videos')
blob = src_bucket.get_blob(video_file_path)
blob.download_to_filename(video_file_path)
path_config_file = '/workdir/model/config.yaml'
initialization_time = time.time() - t0

dlc_analysis_t0 = time.time()
deeplabcut.analyze_videos(path_config_file, [video_file_path])
dlc_analysis_time = time.time() - dlc_analysis_t0

dlc_movie_t0 = time.time()
deeplabcut.create_labeled_video(path_config_file, [video_file_path])
dlc_movie_time = time.time() - dlc_movie_t0



blob2 = tgt_bucket.blob(bucket_data_blobname)
blob2.upload_from_filename(filename=output_data_file)


logger.info('Initialization Time: {}'.format(initialization_time))
logger.info('DLC Analysis Time: {}'.format(dlc_analysis_time))
logger.info('DLC Movie Generation Time: {}'.format(dlc_movie_time))
logger.info('Total Walltime: {}'.format(time.time()-t0))
