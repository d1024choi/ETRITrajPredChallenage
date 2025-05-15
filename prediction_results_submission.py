# Copyright (c) 2025 Dooseop Choi. All rights reserved.
#
# This source code is licensed under the GPL License found in the
# LICENSE file in the root directory of this source tree.
# For more information, contact d1024.choi@etri.re.kr

from libraries import *

def fake_prediction_model(data, best_k, pred_len):
    return np.zeros(shape=(data['agent']['num_nodes'], best_k, pred_len, 2))

def main():

    # parameter setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, default='/home/dooseop/DATASET/ETRI/av2format/test_masked')
    parser.add_argument('--save_path', type=str, default='/home/dooseop/DATASET/ETRI/av2format/test_submission')
    parser.add_argument('--past_horizon_seconds', type=float, default=2, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--future_horizon_seconds', type=float, default=6, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--target_sample_period', type=float, default=10, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--best_k', type=int, default=6, help='DO NOT CHANGE THIS!!')
    args = parser.parse_args()

    obs_len = args.past_horizon_seconds * args.target_sample_period
    pred_len = args.future_horizon_seconds * args.target_sample_period

    # transform each raw tracking file to driving scenes to Argoverse2 driving scenes
    file_names = [f for f in os.listdir(args.source_path) if f.endswith('.pkl')]
    for idx, file_name in enumerate(tqdm(file_names, desc="converting")):

        # read gt data
        with open(os.path.join(args.source_path, file_name), 'rb') as f:
            data = pickle.load(f)

        # do prediction
        predictions = fake_prediction_model(data, args.best_k, pred_len)

        # save the prediction result for the submission
        agent = {'num_nodes': data['agent']['num_nodes'],
                 'num_valid_nodes': data['agent']['num_valid_nodes'],
                 'id': data['agent']['id'],
                 'category': data['agent']['category'],
                 'predictions': predictions}
        scene = {'log_id': data['log_id'], 'frm_idx': data['frm_idx'], 'agent': agent}

        # save data
        file_name_submission = file_name.replace('_masked.pkl', '_submission.pkl')
        with open(os.path.join(args.save_path, file_name_submission), 'wb') as f:
            pickle.dump(scene, f)


if __name__ == '__main__':
    main()

