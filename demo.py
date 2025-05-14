# Copyright (c) 2025 Dooseop Choi. All rights reserved.
#
# This source code is licensed under the GPL License found in the
# LICENSE file in the root directory of this source tree.
# For more information, contact d1024.choi@etri.re.kr

from libraries import *

def main():

    # parameter setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='/home/dooseop/DATASET/ETRI/av2format/temp')
    parser.add_argument('--past_horizon_seconds', type=float, default=2, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--future_horizon_seconds', type=float, default=6, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--target_sample_period', type=float, default=10, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--x_range_abs', type=float, default=150.0, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--y_range_abs', type=float, default=150.0, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--map_size', type=int, default=1024, help='image size for visualization')
    args = parser.parse_args()

    # visualization function
    from visualization import Visualizer
    vs = Visualizer(args)

    # all the file names in the dataset
    file_names = [f for f in os.listdir(args.save_path) if f.endswith('.pkl')]

    # visualize each scene
    for idx, file_name in enumerate(file_names):
        with open(os.path.join(args.save_path, file_name), 'rb') as f:
            scene = pickle.load(f)
            vs.view_current_scene(scene['agent'], scene['map'])


if __name__ == '__main__':
    main()

