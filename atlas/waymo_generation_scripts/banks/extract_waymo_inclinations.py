"""Write data/waymo_top_lidar_inclinations.npy: the 64 beam inclinations (rad) of the Waymo TOP LiDAR.

Taken from the first frame's laser calibration of a raw Waymo TFRecord. The shipped file is the calibration of
the first validation segment (ImageSets/val.txt), segment-10203656353524179475_7625_000_7645_000, which this
reproduces exactly (max abs diff 0). Calibration differs slightly between vehicles (up to ~0.12 deg).

    python extract_waymo_inclinations.py <segment>.tfrecord <out.npy>       (needs waymo_open_dataset, tensorflow)
"""
import sys

import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2

src, out = sys.argv[1], sys.argv[2]
for rec in tf.data.TFRecordDataset(src):
    frame = dataset_pb2.Frame()
    frame.ParseFromString(bytearray(rec.numpy()))
    break
top = [c for c in frame.context.laser_calibrations if c.name == dataset_pb2.LaserName.TOP][0]
inc = np.array(top.beam_inclinations, dtype=np.float64)
np.save(out, inc)
print(f'{out}: {inc.shape[0]} beams, {np.rad2deg(inc.min()):.2f} .. {np.rad2deg(inc.max()):.2f} deg')
