# Copyright (c) 2025 Dooseop Choi. All rights reserved.
#
# This source code is licensed under the GPL License found in the
# LICENSE file in the root directory of this source tree.
# For more information, contact d1024.choi@etri.re.kr


import numpy as np
import math as m
import random
import sys
import pickle
import csv
import os
import time
import matplotlib.pyplot as plt
import cv2
import copy
from random import randint
import argparse

class Pose:

    def __init__(self, heading: float, position: np.ndarray, wlh: np.ndarray):
        '''
        heading (1) : radian
        position (1 x 2) : meter
        wlh (1 x 3) : meter
        '''

        self.heading = heading
        self.position = position[:, :2].reshape(1, 2)  # global position
        self.xyz = position
        self.wlh = wlh

        self.R_e2g = rotation_matrix(heading) # ego-centric to global coordinate
        self.R_g2e = np.linalg.inv(self.R_e2g) # global to ego-centric coordinate
        self.bbox = self.get_bbox()

    def to_agent(self, positions):
        '''
        Global to Agent Centric Coordinate System Conversion

        positions (N x 2)
        output (N x 2)
        '''

        trans = positions - self.position # seq_len x 2
        return np.matmul(self.R_g2e, trans.T).T

    def to_global(self, positions):
        '''
        Agent Centric to Global Coordinate System Conversion

        positions (N x 2)
        output (N x 2)
        '''

        return np.matmul(self.R_e2g, positions.T).T + self.position

    def get_bbox(self):
        '''

           (bottom)          (up)

            front              front
        b0 -------- b3    b4 -------- b7
           |      |          |      |
           |      |          |      |
           |      |          |      |
           |      |          |      |
        b1 -------- b2    b5 -------- b6
             rear              rear
        '''

        # 2D bbox
        w, l, h = self.wlh
        corner_b0 = np.array([l / 2, w / 2]).reshape(1, 2)
        corner_b1 = np.array([-l / 2, w / 2]).reshape(1, 2)
        corner_b2 = np.array([-l / 2, -w / 2]).reshape(1, 2)
        corner_b3 = np.array([l / 2, -w / 2]).reshape(1, 2)
        bbox = np.concatenate([corner_b0, corner_b1, corner_b2, corner_b3], axis=0)  # 4 x 2

        # agent to global coord
        return self.to_global(bbox)

class VossHelper:

    def __init__(self, raw_data_path=None, config_file_name=None):

        self.raw_data_path = raw_data_path
        self.cam0_path, self.lidar_path, self.calib_path = None, None, None
        self.CamInfo = None

        if (raw_data_path is not None):
            self.cam0_path = os.path.join(self.raw_data_path, 'CAM0')
            self.lidar_path = os.path.join(self.raw_data_path, 'PANDAR64')

            # if (config_file_name is not None):
            #     self.calib_path = os.path.join(self.raw_data_path, config_file_name)
            #     self.CamInfo = self.create_calib_parameters()

    def read_csv_file(self, file_dir):
        try:
            return np.genfromtxt(file_dir, delimiter=',')
        except:
            sys.exit(f">> unable to open '{file_dir}'...")

    def read_pc_file(self, file_dir):
        try:
            with open(file_dir, 'rb') as f:
                LiDAR_points = np.fromfile(f, dtype=np.float32)
                LiDAR_points = np.reshape(LiDAR_points, [-1, 4])
                return LiDAR_points[:, :3]
        except:
            sys.exit(f">> unable to open '{file_dir}'...")

    def read_png_file(self, file_dir):
        try:
            return cv2.imread(file_dir)
        except:
            sys.exit(f">> unable to open '{file_dir}'...")

    def get_pose(self, data, frm_idx, obj_id):
        '''
        ------------------------------------------------------------------------------------------------
        frame index| class | obj_id | x[m] | y[m] |  z[m]  | heading[rad] | width[m] | length[m] | height[m] | lidar | image
        ------------------------------------------------------------------------------------------------
        0          | 1     | 2      | 3    | 4    |   5    |      6       |    7     | 8         | 9         | 10     | 11
        ------------------------------------------------------------------------------------------------
        '''

        cur_frm_data = data[data[:, 0] == frm_idx]
        cur_ego_data = cur_frm_data[cur_frm_data[:, 2] == obj_id]

        heading = cur_ego_data[0, 6]
        position = cur_ego_data[0, 3:6].reshape(1, 3)

        return Pose(heading=heading, position=position)

    def get_label_object(self, data, frm_idx):

        cur_frm_data = data[data[:, 0] == frm_idx]
        obj_ids = np.unique(cur_frm_data[:, 2]).tolist()
        obj_ids.remove(-1)
        num_labels = len(obj_ids)

        if (num_labels == 0):
            return []

        obj_dicts = []
        for i, idx in enumerate(obj_ids):

            cur_obj_data = cur_frm_data[cur_frm_data[:, 2] == idx] # 1 x dim
            obj_class = self.code_to_object_class(cur_obj_data[0, 1])
            obj_dict = {'obj_id' : idx,
                        'obj_class' : obj_class,
                        'wlh' : [cur_obj_data[0, 7], cur_obj_data[0, 8], cur_obj_data[0, 9]]}
            obj_dicts.append(obj_dict)

        return obj_dicts

    def code_to_object_class(self, code):

        '''
        code 0 : vehicle
        code 1 : truck
        code 2 : pedestrian
        code 3 : cyclist
        '''

        if (code == 0 or code == 1):
            return 'vehicle'
        elif (code == 2 or code == 3):
            return 'pedestrian'
        else:
            return 'unknown'

    def get_sensor_file_names(self, data, frm_idx):
        cur_frm_data = data[data[:, 0] == frm_idx]
        return cur_frm_data[0, 10], cur_frm_data[0, 11]

    def create_label_file(self):
        '''
        generate 'label.cvs' from data extracted from voss2
        '''

        track_data = self.read_csv_file(os.path.join(self.raw_data_path, 'TRACK_res.csv'))[1:]
        lidar_timestamps = self.read_csv_file(os.path.join(self.raw_data_path, 'PANDAR64.csv'))[1:]
        image_timestamps = self.read_csv_file(os.path.join(self.raw_data_path, 'CAM0.csv'))[1:]
        gps_timestamps = self.read_csv_file(os.path.join(self.raw_data_path, 'GPS_AN.csv'))[1:]

        timestamps = np.unique(track_data[:, 0])
        raw_data = np.concatenate([timestamps.reshape(timestamps.size, 1), np.zeros(shape=(timestamps.size, 3))], axis=1)
        for _, timestamp in enumerate(timestamps):

            minidx = np.argmin(np.abs(gps_timestamps[:, 0] - timestamp))
            z = gps_timestamps[minidx, 6] # altitude (meter)

            minidx = np.argmin(np.abs(lidar_timestamps[:, 0] - timestamp))
            lidar_file_name = lidar_timestamps[minidx, 2] # lidar file name

            minidx = np.argmin(np.abs(image_timestamps[:, 0] - timestamp))
            image_file_name = image_timestamps[minidx, 2] # image file name

            raw_data[_, 1] = z
            raw_data[_, 2] = lidar_file_name
            raw_data[_, 3] = image_file_name

        track_data_ext = np.zeros(shape=(track_data.shape[0], 12))
        track_data_ext[:, :10] = track_data
        for i in range(track_data_ext.shape[0]):
            cur_timestamp = track_data_ext[i][0]
            corr_raw_data = raw_data[raw_data[:, 0] == cur_timestamp]
            track_data_ext[i][-2] = corr_raw_data[0, 2]
            track_data_ext[i][-1] = corr_raw_data[0, 3]

        for index, timestamp in enumerate(timestamps):
            check = (track_data_ext == timestamp)
            track_data_ext[check] = index

        # save as csv file
        file_name = os.path.join(self.raw_data_path, 'label.csv')
        fp = open(file_name, 'w')
        csvWriter = csv.writer(fp, lineterminator='\n')
        csvWriter.writerow(['frame index', 'class', 'obj_id', 'x[m]', 'y[m]', 'z[m]', 'heading[rad]', 'width[m]', 'length[m]', 'height[m]', 'lidar file name', 'image file name'])

        for i in range(track_data_ext.shape[0]):
            cur_data = track_data_ext[i]
            cur_line = [str(int(cur_data[0]))] # frame index
            cur_line.append(str(int(cur_data[1]))) # class
            cur_line.append(str(int(cur_data[2])))  # obj id
            cur_line.append(str(np.around(cur_data[3], decimals=5)))  # x
            cur_line.append(str(np.around(cur_data[4], decimals=5)))  # y
            cur_line.append(str(np.around(cur_data[5], decimals=5)))  # z
            cur_line.append(str(np.around(cur_data[6], decimals=5)))  # heading
            # Note : The current version of VoSS2 has an issue with width and length.
            # cur_line.append(str(np.around(cur_data[7], decimals=5)))  # w
            # cur_line.append(str(np.around(cur_data[8], decimals=5)))  # l
            cur_line.append(str(np.around(cur_data[8], decimals=5)))  # w
            cur_line.append(str(np.around(cur_data[7], decimals=5)))  # l
            cur_line.append(str(np.around(cur_data[9], decimals=5)))  # h
            lidar_file_name = '%08d' % cur_data[10]
            image_file_name = '%08d' % cur_data[11]
            cur_line.append(lidar_file_name)  # lidar
            cur_line.append(image_file_name)  # image
            csvWriter.writerow(cur_line)
        fp.close()

        print(f">> '{file_name}' is generated.." )

    def read_calib_parameters(self):

        # VoSS cfg parsing
        with open(self.calib_path, 'r') as f:
            cfg_lines = f.readlines()

        Caminfo = {}
        for line in cfg_lines:
            line = line.rstrip()
            splited = line.split(' ')
            splited = list(filter(None, splited))

            if len(splited) == 0: continue

            cfg_name = splited[0]
            if cfg_name == "CAM0_general_camera":
                Caminfo['CAM0_general_camera'] = list(map(float, splited[1:]))  # img_w img_h K matrix(3x3)
            elif cfg_name == "CAM0_general_distortion":
                Caminfo['CAM0_general_distortion'] = list(map(float, splited[1:]))  # k1 k2 k3 k4 k5
            elif cfg_name == "CAM0_coordinate_system":
                Caminfo['CAM0_coordinate_system'] = list(map(float, splited[1:]))  # euler angle(roll-pitch-yaw) and translation
        print("Camera Information:", Caminfo)
        return Caminfo

    def project_lidar_pc_to_cam0(self, frm_idx):

        # open label.csv
        data = self.read_csv_file(os.path.join(self.raw_data_path, 'label.csv'))[1:]
        cur_data = data[data[:, 0] == frm_idx]

        # read point cloud and image
        pc = self.read_pc_file(os.path.join(self.lidar_path, '%08d.bin' % cur_data[0, -2]))
        img = self.read_png_file(os.path.join(self.cam0_path, '%08d.png' % cur_data[0, -1]))


        return 0


def Rx(roll):
    return np.array([[1, 0, 0],
                     [0, m.cos(roll), -m.sin(roll)],
                     [0, m.sin(roll), m.cos(roll)]])


def Ry(pitch):
    return np.array([[m.cos(pitch), 0, m.sin(pitch)],
                     [0, 1, 0],
                     [-m.sin(pitch), 0, m.cos(pitch)]])


def Rz(yaw):
    return np.array([[m.cos(yaw), -m.sin(yaw), 0],
                     [m.sin(yaw), m.cos(yaw), 0],
                     [0, 0, 1]])

def rotation_matrix(heading):

    m_cos = np.cos(heading)
    m_sin = np.sin(heading)
    m_R = np.array([m_cos, -1 * m_sin, m_sin, m_cos]).reshape(2, 2)
    return m_R

def in_range_points(points, x, y, z, x_range, y_range, z_range):
    """ extract in-range points """
    points_select = points[np.logical_and.reduce((x > x_range[0], x < x_range[1], y > y_range[0], y < y_range[1], z > z_range[0], z < z_range[1]))]
    return np.around(points_select, decimals=2)


def main():


    vs = VossHelper(raw_data_path='/home/dooseop/DATASET/voss2/20250508-093031_emul',
                    config_file_name='VoSS_IONIQ5_2nd_250416.cfg')
    vs.create_label_file()
    # vs.project_lidar_pc_to_cam0(22)

if __name__ == '__main__':
    main()

