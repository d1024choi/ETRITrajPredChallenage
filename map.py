# Copyright (c) 2025 Dooseop Choi. All rights reserved.
#
# This source code is licensed under the GPL License found in the
# LICENSE file in the root directory of this source tree.
# For more information, contact d1024.choi@etri.re.kr

from libraries import *
import warnings
warnings.filterwarnings("ignore")

class Map:
    '''
    ETRI HDmap data toolkit
    '''
    def __init__(self, dataset_path):

        self.dataset_path = dataset_path
        self.root_path = './ETRImap/'
        self.lanenode, self.lanelink = None, None

        if ('Jan_2021' in dataset_path):
            file_name_lanelink = 'etridb_plus_LAYER_LN_LINK.pkl'
            file_name_lanenode = 'etridb_plus_LAYER_LN_NODE.pkl'
        elif ('May_2025' in dataset_path):
            file_name_lanelink = 'ETRI_OesamDong_LAYER_LN_LINK.pkl'
            file_name_lanenode = 'ETRI_OesamDong_LAYER_LN_NODE.pkl'
        else:
            sys.exit(f"[Error] We don't have map data corresponding to {dataset_path}..")

        rt = self.load_lanelink(os.path.join(self.root_path, file_name_lanelink))
        if (rt is False):
            sys.exit(f"[Error] Unable to properly load the map data '{os.path.join(self.root_path, file_name_lanelink)}'")


        rt = self.load_lanenode(os.path.join(self.root_path, file_name_lanenode))
        if (rt is False):
            sys.exit(f"[Error] Unable to properly load the map data '{os.path.join(self.root_path, file_name_lanenode)}'")

    def load_lanelink(self, file_dir: str) -> bool:

        if (os.path.exists(file_dir)):
            with open(file_dir, 'rb') as f:
                self.lanelink = pickle.load(f)
            return True


        file = open(file_dir.replace('pkl', 'txt'), 'r')
        Lines = file.readlines()

        self.lanelink = {}
        for line in Lines:
            elements = line.split()
            elements = np.array(elements).astype('float')
            num_elmts = len(elements)

            header = elements[:21].astype('int')
            positions = elements[21:].reshape(int((num_elmts - 21) / 2), 2)

            orientations = np.zeros(positions.shape[0])
            orientations[1:] = calc_yaw_from_points(positions[1:] - positions[:-1])
            orientations[0] = orientations[1]

            x_max = np.max(positions[:, 0])
            x_min = np.min(positions[:, 0])
            y_max = np.max(positions[:, 1])
            y_min = np.min(positions[:, 1])

            lane_seg = {'ID': header[0],
                        'Type': header[11],
                        'LLinkID': header[15],
                        'RLinkID': header[16],
                        'SNodeID': header[17],
                        'ENodeID': header[18],
                        'Speed': header[19],
                        'NumPts': header[20],
                        'Pts': positions,
                        'Ori' : orientations,
                        'Cover': [x_max, x_min, y_max, y_min]
                        }

            self.lanelink[str(header[0])] = lane_seg

        with open(file_dir.replace('txt', 'pkl'), 'wb') as f:
            pickle.dump(self.lanelink, f)

        return True

    def load_lanenode(self, file_dir: str) -> bool:

        # check if exists
        if (os.path.exists(file_dir)):
            with open(file_dir, 'rb') as f:
                self.lanenode = pickle.load(f)
            return True

        # make if do not exist
        file = open(file_dir.replace('pkl', 'txt'), 'r')
        Lines = file.readlines()

        self.lanenode = {}
        for line in Lines:
            elements = line.split()
            elements = np.array(elements).astype('float')
            num_elmts = len(elements)

            if ('May_2025' in self.dataset_path):
                num_elmts = num_elmts - 1
                elements = elements[:num_elmts]

            header = elements[:num_elmts - 2].astype('int')
            positions = elements[num_elmts - 2:].reshape(1, 2)

            lane_seg = {'ID': header[0],
                        'NumConLink': header[1],
                        'LinkID': header[2:],
                        'Pts': positions
                        }

            self.lanenode[str(header[0])] = lane_seg

        with open(file_dir.replace('txt', 'pkl'), 'wb') as f:
            pickle.dump(self.lanenode, f)

        return True

    def return_centerlines(self, xy: np.ndarray, x_range: Tuple[int, int], y_range: Tuple[int, int]) -> List:

        w_x_min = xy[0, 0] + x_range[0] - 10
        w_x_max = xy[0, 0] + x_range[1] + 10
        w_y_min = xy[0, 1] + y_range[0] - 10
        w_y_max = xy[0, 1] + y_range[1] + 10
        win_min_max = (w_x_min, w_y_min, w_x_max, w_y_max)

        lane_segments = []
        for idx, key in enumerate(self.lanelink):
            lane_seg = self.lanelink[key]
            lane_min_max = (lane_seg['Cover'][0], lane_seg['Cover'][1], lane_seg['Cover'][2], lane_seg['Cover'][3])
            if (correspondance_check(win_min_max, lane_min_max) == True):
                lane_seg['PrvLinkID'] = self.return_previous_centerline(lane_seg['ID'])
                lane_seg['NxtLinkID'] = self.return_next_centerline(lane_seg['ID'])
                lane_segments.append(lane_seg)
        return lane_segments

    def return_previous_centerline(self, cur_lane_id: int) -> List:
        previous_lane_ids = []
        cur_lane_seg = self.lanelink[str(cur_lane_id)]
        start_node_id = cur_lane_seg['SNodeID']
        lanes_connected_to_the_noode = self.lanenode[str(start_node_id)]['LinkID'].tolist()
        if (len(lanes_connected_to_the_noode) == 0): return previous_lane_ids

        start_pts_cur_lane_seg = cur_lane_seg['Pts'][0]
        for candi_prev_lane_id in lanes_connected_to_the_noode:
            end_pts_prev_lane_seg = self.lanelink[str(candi_prev_lane_id)]['Pts'][-1]
            dist = np.sum(np.abs(start_pts_cur_lane_seg - end_pts_prev_lane_seg))
            if (dist == 0):
                previous_lane_ids.append(candi_prev_lane_id)

        return previous_lane_ids

    def return_next_centerline(self, cur_lane_id: int) -> List:
        next_lane_ids = []
        cur_lane_seg = self.lanelink[str(cur_lane_id)]
        end_node_id = cur_lane_seg['ENodeID']
        lanes_connected_to_the_noode = self.lanenode[str(end_node_id)]['LinkID'].tolist()
        if (len(lanes_connected_to_the_noode) == 0): return next_lane_ids

        end_pts_cur_lane_seg = cur_lane_seg['Pts'][-1]
        for candi_next_lane_id in lanes_connected_to_the_noode:
            start_pts_prev_lane_seg = self.lanelink[str(candi_next_lane_id)]['Pts'][0]
            dist = np.sum(np.abs(end_pts_cur_lane_seg - start_pts_prev_lane_seg))
            if (dist == 0):
                next_lane_ids.append(candi_next_lane_id)

        return next_lane_ids



    def __repr__(self):
        return f"ETRI Map Helper."


def correspondance_check(win_min_max: Tuple, lane_min_max: Tuple) -> bool:

    # four points for window and lane box
    w_x_min, w_y_min, w_x_max, w_y_max = win_min_max
    # l_x_min, l_y_min, l_x_max, l_y_max = lane_min_max
    l_x_max, l_x_min, l_y_max, l_y_min = lane_min_max

    w_TL = (w_x_min, w_y_max)  # l1
    w_BR = (w_x_max, w_y_min)  # r1

    l_TL = (l_x_min, l_y_max)  # l2
    l_BR = (l_x_max, l_y_min)  # r2

    # If one rectangle is on left side of other
    # if (l1.x > r2.x | | l2.x > r1.x)
    if (w_TL[0] > l_BR[0] or l_TL[0] > w_BR[0]):
        return False

    # If one rectangle is above other
    # if (l1.y < r2.y || l2.y < r1.y)
    if (w_TL[1] < l_BR[1] or l_TL[1] < w_BR[1]):
        return False

    return True

def calc_yaw_from_points(vec1: np.ndarray) -> np.ndarray:

    '''
    vec : seq_len x 2
    '''

    seq_len = vec1.shape[0]

    vec1 = vec1.reshape(seq_len, 2)
    x1 = vec1[:, 0]
    y1 = vec1[:, 1]
    heading = np.arctan2(y1, x1)

    return heading
