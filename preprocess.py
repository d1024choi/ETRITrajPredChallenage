# Copyright (c) 2025 Dooseop Choi. All rights reserved.
#
# This source code is licensed under the GPL License found in the
# LICENSE file in the root directory of this source tree.
# For more information, contact d1024.choi@etri.re.kr

from libraries import *
from VossHelper import VossHelper
from map import Map

class DatasetBuilder:

    def __init__(self, args):

        past_horizon_seconds = args.past_horizon_seconds
        future_horizon_seconds = args.future_horizon_seconds
        self.min_num_agents = args.min_num_agents
        self.scene_len = (future_horizon_seconds + past_horizon_seconds) * 10 # 10Hz
        self.obs_len = args.past_horizon_seconds * 10
        self.pred_len = args.future_horizon_seconds * 10
        self.x_range = (-1.0 * args.x_range_abs, args.x_range_abs)
        self.y_range = (-1.0 * args.y_range_abs, args.y_range_abs)

        # help load and convert raw csv files
        self.vh = VossHelper()

        # help handling map data
        self.map = Map(args.dataset_path)


    def return_driving_scene(self, curr_frm_index: int, frm_indices: np.ndarray, data: np.ndarray)-> Tuple[Dict, List]:

        '''
        code 0 : vehicle
        code 1 : truck
        code 2 : pedestrian
        code 3 : cyclist
        '''

        agent = {'num_nodes': None, 'av_index': -1, 'id': None, 'type': None, 'position' : None, 'heading': None,
                 'valid_mask': None, 'predict_mask': None, 'velocity': None, 'category': None, 'wlh': None,
                 'num_valid_nodes': None}


        # target duration
        tar_frm_index_start = frm_indices[curr_frm_index]
        tar_frm_index_end = frm_indices[curr_frm_index] + self.scene_len

        # the number of agents and id
        agent_ids = []
        for tar_frm_index in range(tar_frm_index_start, tar_frm_index_end):
            curr_data = data[data[:, 0] == tar_frm_index] # num_agents x 12
            agent_ids += curr_data[:, 2].tolist()
        agent_ids = np.unique(np.array(agent_ids).astype('int').astype('str')).tolist()
        agent['num_nodes'] = len(agent_ids)
        agent['id'] = agent_ids

        # type and wlh
        agent['type'] = np.zeros(agent['num_nodes']).astype('int')
        agent['wlh'] = np.zeros(shape=(agent['num_nodes'], 3))
        for _, id in enumerate(agent_ids):
            agent_data = data[data[:, 2] == float(id)]
            code = agent_data[0, 1]
            if (code == 0 or code == 1 or code == -1):
                agent['type'][_] = 0 # vehicle
            elif (code == 2):
                agent['type'][_] = 1 # pedestrian
            elif (code == 3):
                agent['type'][_] = 2 # cyclist

            if (id == '-1'):
                agent['wlh'][_, 0] = 2.06
                agent['wlh'][_, 1] = 4.59
                agent['wlh'][_, 2] = 1.54
            else:
                agent['wlh'][_, 0] = agent_data[0, 7]
                agent['wlh'][_, 1] = agent_data[0, 8]
                agent['wlh'][_, 2] = agent_data[0, 9]



        # position, heading, velocity, and valid_mask
        agent['position'] = -1000.0 * np.ones(shape=(agent['num_nodes'], self.scene_len, 3))
        agent['heading'] = -1000.0 * np.ones(shape=(agent['num_nodes'], self.scene_len))
        agent['velocity'] = -1000.0 * np.ones(shape=(agent['num_nodes'], self.scene_len, 3)) # !! UNAVAILABLE !!
        agent['valid_mask'] = np.zeros(shape=(agent['num_nodes'], self.scene_len)).astype('bool')
        tar_frm_indices = [t for t in range(tar_frm_index_start, tar_frm_index_end, 1)]
        for _, tar_frm_index in enumerate(tar_frm_indices):
            curr_data = data[data[:, 0] == tar_frm_index]
            for __, id in enumerate(agent['id']):
                agent_data = curr_data[curr_data[:, 2] == float(id)] # 1 x 12
                if (agent_data.size == 0): continue

                agent['position'][__, _, 0] = agent_data[0, 3]
                agent['position'][__, _, 1] = agent_data[0, 4]
                agent['position'][__, _, 2] = agent_data[0, 5]
                agent['heading'][__, _] = agent_data[0, 6]
                agent['valid_mask'][__, _] = True

        # prediction mask and category
        num_valid_nodes = 0
        agent['predict_mask'] = np.zeros(shape=(agent['num_nodes'], self.scene_len)).astype('bool')
        agent['category'] = np.zeros(agent['num_nodes']).astype('int')
        for _, id in enumerate(agent['id']):
            agent['predict_mask'][_, self.obs_len:] = agent['valid_mask'][_, self.obs_len:]

            if (np.count_nonzero(agent['valid_mask'][_]) == self.scene_len): # complete trajectory
                if (agent['type'][_] == 0):
                    agent['category'][_] = int(2)  # SCORED_TRACK
                    num_valid_nodes += 1
                else:
                    agent['category'][_] = int(1)  # UNSCORED_TRACK
            else:
                agent['category'][_] = int(0)  # TRACK_FRAGMENT
        agent['num_valid_nodes'] = num_valid_nodes

        # map
        xy = agent['position'][0, self.obs_len-1, :2].reshape(1, 2) # current position of ego vehicle
        line_segs = self.map.return_centerlines(xy, self.x_range, self.y_range)

        return agent, line_segs





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default='/home/dooseop/DATASET/nuscenes/')
    parser.add_argument('--version', type=str, default='v1.0-trainval')

    parser.add_argument('--past_horizon_seconds', type=float, default=2)
    parser.add_argument('--future_horizon_seconds', type=float, default=4)
    parser.add_argument('--target_sample_period', type=float, default=5)

    parser.add_argument('--min_num_agents', type=int, default=1)
    parser.add_argument('--obs_len', type=int, default=10)
    parser.add_argument('--pred_len', type=int, default=20)

    parser.add_argument('--preprocess', type=int, default=0)
    parser.add_argument('--num_turn_scene_repeats', type=int, default=0)
    parser.add_argument('--val_ratio', type=float, default=0.05)


    args = parser.parse_args()

    builder = DatasetBuilder(args)

    for i in range(32, 100):
        a = 0
        # print(">> scene_number : %d, angle: %.2f, location: %s" % (i, angle-90, location))

if __name__ == '__main__':
    main()

