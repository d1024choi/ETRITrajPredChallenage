# Copyright (c) 2025 Dooseop Choi. All rights reserved.
#
# This source code is licensed under the GPL License found in the
# LICENSE file in the root directory of this source tree.
# For more information, contact d1024.choi@etri.re.kr

from libraries import *
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

class Visualizer:
    '''
    ETRIdataset Visualization Tool
    '''

    def __init__(self, args):

        self.args = args

        past_horizon_seconds = args.past_horizon_seconds
        future_horizon_seconds = args.future_horizon_seconds
        self.scene_len = (future_horizon_seconds + past_horizon_seconds) * 10 # 10Hz
        self.obs_len = args.past_horizon_seconds * 10
        self.pred_len = args.future_horizon_seconds * 10

        # An area centered at the self.obs_len-th position of AV covers 2*x_range_abs x 2*y_range_abs m^2
        self.x_range = (-1.0 * args.x_range_abs, args.x_range_abs)
        self.y_range = (-1.0 * args.y_range_abs, args.y_range_abs)

        # The area is displayed on an image of size map_size x map_size pixels
        self.map_size = args.map_size

        axis_range_y = self.y_range[1] - self.y_range[0]
        axis_range_x = self.x_range[1] - self.x_range[0]
        self.scale_y = float(self.map_size - 1) / axis_range_y
        self.scale_x = float(self.map_size - 1) / axis_range_x

        self.dpi = 80
        self.color_centerline = (0, 0, 0)
        self.color_laneside = (0.5, 0.5, 0.5)
        self.color_bbox = (0.5, 0.5, 0.5)
        self.palette = make_palette(self.pred_len)

    def view_current_scene(self, agent: Dict, line_segs: List):

        # draw point cloud topivew
        fig, ax = plt.subplots()
        img = 255 * np.ones(shape=(self.map_size, self.map_size, 3))
        ax.imshow(img.astype('float') / 255.0, extent=[0, self.map_size, 0, self.map_size])

        # center position
        xy = agent['position'][0, self.obs_len-1, :2].reshape(1, 2)

        # lane segments
        lines = []
        for _, line_seg in enumerate(line_segs):
            lines.append(line_seg['Pts'])

        # draw centerlines
        ax = self.draw_centerlines(ax, xy, lines)

        # draw bounding bboxes (objects visible at current time)
        ax = self.draw_bbox(ax, agent)

        # draw observed and future trajectories (objects visible at current time)
        ax = self.draw_observed_trajectory(ax, agent)

        plt.axis([0, self.map_size, 0, self.map_size])
        plt.show()

        # img = self.fig_to_nparray(fig, ax)
        # cv2.imshow("", img.astype('uint8'))
        # cv2.waitKey(0)

    def draw_lines_on_topview(self, ax, line: np.ndarray, color: Tuple):
        col_pels = -(line[:, 1] * self.scale_y).astype(np.int32)
        row_pels = -(line[:, 0] * self.scale_x).astype(np.int32)
        col_pels += int(np.trunc(self.y_range[1] * self.scale_y))
        row_pels += int(np.trunc(self.x_range[1] * self.scale_x))
        ax.plot(col_pels, self.map_size - row_pels, '-', linewidth=1.0, color=color, alpha=1)
        return ax

    def topview_bbox(self, ax, bbox: np.ndarray, id: str, color: Tuple):

        col_pels = -(bbox[:, 1] * self.scale_y).astype(np.int32)
        row_pels = -(bbox[:, 0] * self.scale_x).astype(np.int32)
        col_pels += int(np.trunc(self.y_range[1] * self.scale_y))
        row_pels += int(np.trunc(self.x_range[1] * self.scale_x))
        row_pels = self.map_size - row_pels

        line_col = [col_pels[0], col_pels[1]]
        line_row = [row_pels[0], row_pels[1]]
        ax.plot(line_col, line_row, '-', linewidth=1.0, color=color, alpha=1)

        line_col = [col_pels[1], col_pels[2]]
        line_row = [row_pels[1], row_pels[2]]
        ax.plot(line_col, line_row, '-', linewidth=1.0, color=color, alpha=1)

        line_col = [col_pels[2], col_pels[3]]
        line_row = [row_pels[2], row_pels[3]]
        ax.plot(line_col, line_row, '-', linewidth=1.0, color=color, alpha=1)

        line_col = [col_pels[3], col_pels[0]]
        line_row = [row_pels[3], row_pels[0]]
        ax.plot(line_col, line_row, '-', linewidth=1.0, color=color, alpha=1)

        col_pel = int((col_pels[0] + col_pels[3]) / 2)
        row_pel = int((row_pels[0] + row_pels[3]) / 2)

        line_col = [col_pels[1], col_pel]
        line_row = [row_pels[1], row_pel]
        ax.plot(line_col, line_row, '-', linewidth=1.0, color=color, alpha=1)

        line_col = [col_pels[2], col_pel]
        line_row = [row_pels[2], row_pel]
        ax.plot(line_col, line_row, '-', linewidth=1.0, color=color, alpha=1)

        ax.text(col_pel, row_pel, id)

        return ax

    def topview_trajectory(self, ax, trajectory: np.ndarray, valid: np.ndarray, mode: str):
        '''
        trajectory : (float) seq_len x 2
        valid : (bool) seq_len
        mode : 'past' or 'future'
        '''

        col_pels = -(trajectory[:, 1] * self.scale_y).astype(np.int32)
        row_pels = -(trajectory[:, 0] * self.scale_x).astype(np.int32)
        col_pels += int(np.trunc(self.y_range[1] * self.scale_y))
        row_pels += int(np.trunc(self.x_range[1] * self.scale_x))
        row_pels = self.map_size - row_pels


        if (mode == 'future'):
            for t in range(self.pred_len):
                if (valid[t]):
                    r, g, b = self.palette[t]
                    ax.plot(col_pels[t], row_pels[t], 's', linewidth=1.0, color=(r, g, b), alpha=1.0)
        elif (mode == 'past'):
            for t in range(self.obs_len):
                ax.plot(col_pels[t], row_pels[t], 'o', linewidth=1.0, color=(0.5, 0.5, 0.5), alpha=0.5)

        return ax

    def draw_centerlines(self, ax, xy: np.ndarray, lines: List):
        for _, cur_line in enumerate(lines):
            ax = self.draw_lines_on_topview(ax, cur_line-xy, color=self.color_centerline)
        return ax

    def draw_bbox(self, ax, agent: Dict):
        xy = agent['position'][0, self.obs_len - 1, :2].reshape(1, 2) # AV current position
        num_agents = agent['num_nodes']
        for i in range(num_agents):
            if (agent['valid_mask'][i, self.obs_len-1]):
                bbox = Pose(heading=agent['heading'][i, self.obs_len-1],
                            position=agent['position'][i, self.obs_len-1, :2].reshape(1, 2),
                            wlh=agent['wlh'][i]).bbox # 4 x 2
                ax = self.topview_bbox(ax, bbox - xy, agent['id'][i], self.color_bbox)
        return ax

    def draw_observed_trajectory(self, ax, agent: Dict):
        xy = agent['position'][0, self.obs_len - 1, :2].reshape(1, 2) # AV current position
        num_agents = agent['num_nodes']
        for i in range(num_agents):
            if (agent['valid_mask'][i, self.obs_len-1]):
                trajectory = agent['position'][i, :, :2] # seq_len x 2
                valid = agent['valid_mask'][i] # seq_len
                ax = self.topview_trajectory(ax, trajectory[:self.obs_len] - xy, valid[:self.obs_len], mode='past')
                ax = self.topview_trajectory(ax, trajectory[self.obs_len:] - xy, valid[self.obs_len:], mode='future')
        return ax

    def fig_to_nparray(self, fig, ax):
        fig.set_size_inches(self.map_size / self.dpi, self.map_size / self.dpi)
        ax.set_axis_off()

        fig.canvas.draw()
        render_fig = np.array(fig.canvas.renderer._renderer)

        final_img = np.zeros_like(render_fig[:, :, :3]) # 450, 1600
        final_img[:, :, 2] = render_fig[:, :, 0]
        final_img[:, :, 1] = render_fig[:, :, 1]
        final_img[:, :, 0] = render_fig[:, :, 2]

        plt.close()
        return final_img

def make_palette(pred_len):
    red = np.array([1, 0, 0])
    orange = np.array([1, 0.5, 0])
    yellow = np.array([1, 1.0, 0])
    green = np.array([0, 1.0, 0])
    blue = np.array([0, 0, 1])
    colors = [red, orange, yellow, green, blue]

    palette = []
    for t in range(pred_len):

        cur_pos = 4.0 * float(t) / float(pred_len - 1)  # pred_len -> 0 ~ 4
        prev_pos = int(cur_pos)
        next_pos = int(cur_pos) + 1

        if (next_pos > 4):
            next_pos = 4

        prev_color = colors[prev_pos]
        next_color = colors[next_pos]

        prev_w = float(next_pos) - cur_pos
        next_w = 1 - prev_w

        cur_color = prev_w * prev_color + next_w * next_color
        palette.append(cur_color)

    return palette

def correspondance_check(win_min_max, lane_min_max):

    # four points for window and lane box
    w_x_min, w_y_min, w_x_max, w_y_max = win_min_max
    l_x_min, l_y_min, l_x_max, l_y_max = lane_min_max

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

def make_rot_matrix_from_yaw(yaw):

    '''
    yaw in radian
    '''

    m_cos = np.cos(yaw)
    m_sin = np.sin(yaw)
    m_R = [[m_cos, -1 * m_sin], [m_sin, m_cos]]

    return np.array(m_R)


def rotation_matrix(heading):

    m_cos = np.cos(heading)
    m_sin = np.sin(heading)
    m_R = np.array([m_cos, -1 * m_sin, m_sin, m_cos]).reshape(2, 2)
    return m_R