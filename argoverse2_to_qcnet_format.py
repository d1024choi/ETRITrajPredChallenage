# Copyright (c) 2025 Dooseop Choi. All rights reserved.
#
# This source code is licensed under the GPL License found in the
# LICENSE file in the root directory of this source tree.
# For more information, contact d1024.choi@etri.re.kr

from libraries import *

TO_TENSOR_KEYS = ['type', 'position', 'heading', 'valid_mask', 'predict_mask', 'velocity', 'wlh']

def main():

    # parameter setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, default='/home/dooseop/DATASET/ETRI/av2format/train')
    parser.add_argument('--save_path', type=str, default='/home/dooseop/DATASET/ETRI/av2format/train_qcnet')
    parser.add_argument('--past_horizon_seconds', type=float, default=2, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--future_horizon_seconds', type=float, default=6, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--target_sample_period', type=float, default=10, help='DO NOT CHANGE THIS!!')
    args = parser.parse_args()

    from qcnet_map_preprocess import ProcessMap
    map = ProcessMap()

    # transform each raw tracking file to driving scenes to Argoverse2 driving scenes
    file_names = [f for f in os.listdir(args.source_path) if f.endswith('.pkl')]
    for idx, file_name in enumerate(tqdm(file_names, desc="converting")):

        # read gt data
        with open(os.path.join(args.source_path, file_name), 'rb') as f:
            data = pickle.load(f)

        # from numpy to tensor
        for key, value in data['agent'].items():
            if (key in TO_TENSOR_KEYS):
                data['agent'][key] = torch.from_numpy(value)

        # add map polygon and map position
        qcnet_type_map = map(data['map'])
        data['map_polygon'] = qcnet_type_map['map_polygon']
        data['map_point'] = qcnet_type_map['map_point']
        data[('map_point', 'to', 'map_polygon')] = qcnet_type_map[('map_point', 'to', 'map_polygon')]
        data[('map_polygon', 'to', 'map_polygon')] = qcnet_type_map[('map_polygon', 'to', 'map_polygon')]

        data.pop('map')


        # save data
        file_name_concealed = file_name.replace('.pkl', '_qcnet.pkl')
        with open(os.path.join(args.save_path, file_name_concealed), 'wb') as f:
            pickle.dump(data, f)


if __name__ == '__main__':
    main()

