# Copyright (c) 2025 Dooseop Choi. All rights reserved.
#
# This source code is licensed under the GPL License found in the
# LICENSE file in the root directory of this source tree.
# For more information, contact d1024.choi@etri.re.kr

from libraries import *

def main():

    # parameter setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./ETRIdataset/May_2025/TEST')
    # parser.add_argument('--dataset_path', type=str, default='./ETRIdataset/May_2025/TRAIN')
    # parser.add_argument('--dataset_path', type=str, default='./ETRIdataset/Jan_2021/')

    # parser.add_argument('--save_path', type=str, default='/home/dooseop/DATASET/ETRI/av2format/train')
    parser.add_argument('--save_path', type=str, default='/home/dooseop/DATASET/ETRI/av2format/test')
    parser.add_argument('--past_horizon_seconds', type=float, default=2, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--future_horizon_seconds', type=float, default=6, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--target_sample_period', type=float, default=10, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--min_num_agents', type=int, default=1)
    parser.add_argument('--pivot', type=int, default=15, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--x_range_abs', type=float, default=150.0, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--y_range_abs', type=float, default=150.0, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--map_size', type=int, default=1024, help='image size for visualization')

    args = parser.parse_args()

    # load dataset builder and visualizer
    from preprocess import DatasetBuilder
    db = DatasetBuilder(args)

    from visualization import Visualizer
    vs = Visualizer(args)

    # get all the raw tracking files
    subfolders = [name for name in os.listdir(args.dataset_path)
                  if os.path.isdir(os.path.join(args.dataset_path, name))]

    # transform each raw tracking file to driving scenes to Argoverse2 driving scenes
    TotalNumScenes = 0
    for idx, subfolder in enumerate(tqdm(subfolders, desc="converting")):

        # read a raw file
        try:
            raw_file_path = os.path.join(args.dataset_path, subfolder + '/label_ori.csv')
            raw_data = db.vh.read_csv_file(raw_file_path)[1:]  # original tracking data
        except:
            raw_file_path = os.path.join(args.dataset_path, subfolder + '/label.csv')
            raw_data = db.vh.read_csv_file(raw_file_path)[1:]  # original tracking data

        frm_indices = np.unique(raw_data[:, 0]).astype('int')
        num_candi_scenes = len(frm_indices) - (db.scene_len + 1)

        for curr_frm_index in range(args.pivot, num_candi_scenes):
            agent, line_segs = db.return_driving_scene(curr_frm_index, frm_indices, raw_data)

            if (agent['num_valid_nodes'] < args.min_num_agents):
                continue

            TotalNumScenes += 1

            scene = {'log_id': subfolder, 'frm_idx': curr_frm_index, 'agent': agent, 'map': line_segs}
            save_path = os.path.join(args.save_path, f'log_{subfolder}_{curr_frm_index:07d}.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(scene, f)

    print(f">> Total Number of Scenes : {TotalNumScenes}")

if __name__ == '__main__':
    main()

