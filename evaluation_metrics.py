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

        minADE6 = error.mean(-1).min(-1)[0]
        minFDE6 = error[..., -1].min(-1)[0]

        minADE1 = error.mean(-1)[:, 0]
        minFDE1 = error[..., -1][:, 0]

        return minADE1, minFDE1, minADE6, minFDE6


def main():

    # parameter setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--past_horizon_seconds', type=float, default=2, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--future_horizon_seconds', type=float, default=6, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--target_sample_period', type=float, default=10, help='DO NOT CHANGE THIS!!')
    parser.add_argument('--num_of_candi', type=int, default=6, help='DO NOT CHANGE THIS!!')
    args = parser.parse_args()

    future_time_horizon = int(args.future_horizon_seconds * args.target_sample_period)
    EM = EvaluationMetrics(future_time_horizon, args.num_of_candi)

    total_minADE1, total_minFDE1 = [], []
    total_minADE6, total_minFDE6 = [], []
    for i in range(100):

        num_agents = random.randint(1, 100)
        gt = torch.randn(size=(num_agents, future_time_horizon, 2))
        pred = torch.randn(size=(num_agents, args.num_of_candi, future_time_horizon, 2))

        minADE1, minFDE1, minADE6, minFDE6 = EM(gt, pred)
        total_minADE1 += minADE1.tolist()
        total_minFDE1 += minFDE1.tolist()
        total_minADE6 += minADE6.tolist()
        total_minFDE6 += minFDE6.tolist()

    print(">> minADE1 : %.4f, minFDE1 : %.4f" % (np.mean(total_minADE1), np.mean(total_minFDE1)))
    print(">> minADE6 : %.4f, minFDE6 : %.4f" % (np.mean(total_minADE6), np.mean(total_minFDE6)))



if __name__ == '__main__':
    main()

