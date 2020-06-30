# from arguments import get_args
import csv
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np


def plot_array(stats_summary1, stats_summary2=None):

    assert stats_summary1.shape[0] == 10, 'dimension is not 10!'
    if stats_summary2 is not None: assert stats_summary2.shape[0] == 10, 'dimension is not 10!'
    save_dir_spread = ['./figs/c_reward.png', './figs/room_explored.png', './figs/dummy_agents.png',
                       './figs/max_belief.png', './figs/min_belief.png']
    title_str = ['cumulative reward', '# of cells explored', '# of dummy agents found',
                       'max red belief', 'min red belief']
    legend_position = ['upper center', 'lower right', 'lower right', 'lower right', 'upper right']
    length = stats_summary1.shape[1]

    num_fig = stats_summary1.shape[0] // 2
    num_fig = 5
    # fig_list = [plt.subplots() for _ in range(num_fig)]
    for i in range(num_fig):
        current_mean = stats_summary1[2 * i,:]
        current_var = stats_summary1[2 * i + 1, ]
        step_index = np.linspace(0, length-1, length, endpoint=True)
        # fig1 = plt.gcf()
        plt.close(fig=None)
        fig, ax = plt.subplots()
        # fig, ax = fig_list[i]

        var_ratio = 5e-3 if i == 0 else 1
        ax.plot(step_index, current_mean, '-b', label='learned policy')
        ax.fill_between(step_index, current_mean - var_ratio*current_var,
                        current_mean + var_ratio*current_var, facecolor="blue", alpha=0.3)

        if stats_summary2 is not None:
            current_mean2 = stats_summary2[2 * i, :]
            current_var2 = stats_summary2[2 * i + 1, ]
            ax.plot(step_index, current_mean2, '-r', label='handcraft policy')
            ax.fill_between(step_index, current_mean2 - var_ratio*current_var2,
                            current_mean2 + var_ratio*current_var2, facecolor="red", alpha=0.3)




        # ax.plot(data_spread_sparse[0][1:,1], data_spread_sparse[0][1:,2], '-r', label='sparse-Att')
        font_size = 18
        plt.xlabel('Step', fontsize=font_size)
        plt.ylabel(title_str[i], fontsize=font_size)
        # plt.rcParams.update({'font.size': 22})
        # leg = ax.legend()
        ax.legend(loc=legend_position[i], fontsize=font_size)
        # plt.xlim((0, len_sparse*10*num_process))
        # plt.xlim((0, 65*10**3))
        # plt.ylim(((-0.5, 0)))
        plt.savefig(save_dir_spread[i], dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1)

        # plt.show()
        # plt.close(fig=None)
        print()


if __name__ == '__main__':

    # f1 = open('trained.txt', 'r+')
    # f2 = open("handcraft.txt", 'r+')
    # trained_policy_data = f1.read()
    # handcraft_policy_data = f2.read()
    # f1.close()
    # f2.close()
    handcraft_policy_data = None
    trained_policy_data = np.genfromtxt("./text/trained.txt", dtype='str')
    trained_policy_data = trained_policy_data.astype(np.float)

    handcraft_policy_data = np.genfromtxt("./text/handcraft.txt", dtype='str')
    handcraft_policy_data = handcraft_policy_data.astype(np.float)

    plot_array(trained_policy_data, handcraft_policy_data)