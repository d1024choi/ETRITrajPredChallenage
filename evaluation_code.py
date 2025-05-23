# Copyright (c) 2025 Dooseop Choi. All rights reserved.
#
# This source code is licensed under the GPL License found in the
# LICENSE file in the root directory of this source tree.
# For more information, contact d1024.choi@etri.re.kr

from libraries import *

class EvaluationMetrics:

    def __init__(self, time_horizon: int, num_of_candi: int):

        self.time_horizon = time_horizon
        self.num_of_candi = num_of_candi


    def position_wise_distance(self, gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        '''
        gt : num_agents x seq_len x 2
        pred : num_agents x num_candi x seq_len x 2
        '''

        return (gt[:, None] - pred).pow(2).sqrt().sum(-1) # num_agents x num_candi x seq_len

    def __call__(self, gt: torch.Tensor, pred: torch.Tensor):
        '''
        gt : num_agents x seq_len x 2
        pred : num_agents x num_candi x seq_len x 2
        '''

        if (gt.size(-1) > 2):
            gt = gt[..., :2]

        if (pred.size(-1) > 2):
            pred = pred[..., :2]

        seq_len = gt.size(1)
        num_candi = pred.size(1)

        assert (seq_len == self.time_horizon)
        assert (num_candi >= self.num_of_candi)

        error = self.position_wise_distance(gt, pred) # num_agents x (num_candi + alpha) x seq_len
        error = error[:, :self.num_of_candi] # num_agents x num_candi x seq_len

        minADE = error.mean(-1).min(-1)
        minFDE = error[..., -1].min(-1)

        return minADE[0], minFDE[0]


def main():

    # parameter setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--past_horizon_seconds', type=float, default=2, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--future_horizon_seconds', type=float, default=6, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--target_sample_period', type=float, default=10, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--num_of_candi', type=int, default=6, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--gt_data_path', type=str, default='/home/dooseop/DATASET/ETRI/av2format/test/')
    parser.add_argument('--submit_data_path', type=str, default='/home/dooseop/Pytorch/AIChallenge/QCNet_on_ETRI_Dataset_updated/lightning_logs/version_0/test_results')
    args = parser.parse_args()

    future_time_horizon = int(args.future_horizon_seconds * args.target_sample_period)
    past_time_horizon = int(args.past_horizon_seconds * args.target_sample_period)
    EM = EvaluationMetrics(future_time_horizon, args.num_of_candi)

    file_names = [f for f in os.listdir(args.submit_data_path) if f.endswith('.pkl')]

    total_minADE, total_minFDE = [], []
    for _, file_name in enumerate(tqdm(file_names, desc='under assessment')):


        # prediction submitted
        with open(os.path.join(args.submit_data_path, file_name), 'rb') as file:
            submit_data = pickle.load(file)

        # ground-truth
        file_name = file_name.replace('_submission', '')
        with open(os.path.join(args.gt_data_path, file_name), 'rb') as file:
            gt_data = pickle.load(file)

        # check validity
        assert (submit_data['log_id'] == gt_data['log_id'])
        assert (submit_data['frm_idx'] == gt_data['frm_idx'])
        assert (submit_data['agent']['num_nodes'] == gt_data['agent']['num_nodes'])
        assert (submit_data['agent']['num_valid_nodes'] == gt_data['agent']['num_valid_nodes'])
        assert (submit_data['agent']['id'] == gt_data['agent']['id'])
        assert (np.array_equal(submit_data['agent']['category'], gt_data['agent']['category']))

        valid_node_chk = gt_data['agent']['category'] == 2

        # predicted trajectories
        pred = torch.from_numpy(submit_data['agent']['predictions']) # num_agent x 6 x future_horizon x 2
        pred = pred[valid_node_chk]

        # ground-truth trajectories
        gt = torch.from_numpy(gt_data['agent']['position'][:, past_time_horizon:, :2]) # num_agent x future_horizon x 2
        gt = gt[valid_node_chk]

        minADE, minFDE = EM(gt, pred)
        total_minADE += minADE.tolist()
        total_minFDE += minFDE.tolist()

    print(">> minADE : %.4f, minFDE : %.4f" % (np.mean(total_minADE), np.mean(total_minFDE)))



if __name__ == '__main__':
    main()

