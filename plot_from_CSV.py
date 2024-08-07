import csv

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re

color_list = {"3": "#56B4E9", "4": "#2878b5", "5": "#009E73", "8": "#f05260"}


def plot_trajectory_history_result():
    filetype = '.csv'
    file_list = []
    legend_list = []
    method_list = []

    for _, _, files in os.walk(f'results/csv'):
        for f in files:
            if filetype in f:
                file_list.append(f)
    file_list.sort(reverse=False)

    fig = plt.figure(figsize=(4,3))

    for i, csv_file in enumerate(file_list):
        print(csv_file, end='\t')
        if 'rl' in csv_file:
            method_name = 'ViPER'
            linestyle = 'solid'
        elif 'durham' in csv_file:
            method_name = 'Durham'
            linestyle = 'dashed'
        else:
            raise ValueError('Unknown method')
        match = re.search(r'n=(\d+)', csv_file)
        if match:
            method_name += f' (n={match.group(1)})'
        else:
            raise ValueError('Unknown method')
        print(method_name)
        method_list.append(method_name)
        color = color_list[match.group(1)]

        csv_file = 'results/csv/'+ csv_file
        trajectory_history = pd.read_csv(csv_file)
        trajectory_history = trajectory_history.sort_values('length')
        trajectory_mean = trajectory_history.rolling(400, on='length', min_periods=20, center=True).mean()
        trajectory_std = trajectory_history.rolling(400, on='length', min_periods=20, center=True).std()

        trajectory_mean = trajectory_mean[trajectory_mean['length'] <= 500]
        trajectory_std = trajectory_std[trajectory_std['length'] <= 500]

        line = plt.plot(trajectory_mean.length, trajectory_mean.safe, color=color, linewidth=1.5, zorder=i, linestyle=linestyle)
        plt.fill_between(trajectory_mean.length, trajectory_mean.safe - trajectory_std.safe,
                         trajectory_mean.safe + trajectory_std.safe, color=color, alpha=0.15)
        # line = plt.plot(trajectory_mean.length, trajectory_mean.explore, color=color, linewidth=1.5, zorder=10 - i,
        #                 linestyle=linestyle)
        # plt.fill_between(trajectory_mean.length, trajectory_mean.explore - trajectory_std.explore,
        #                  trajectory_mean.explore + trajectory_std.explore, color=color, alpha=0.15)
        legend_list.append(line[0])
        plt.xticks(size=10)
        plt.xlabel('Length', fontdict={'size': 12})
        plt.xlim(0, 500)
        # plt.tick_params(bottom=False, left=False, axis='both', pad=0.1)
        plt.ylabel('Cleared Rate', fontdict={'size': 12})
        plt.yticks(size=10)
        plt.ylim(0, 1)


    plt.legend(legend_list, method_list, labelspacing=0.1, borderaxespad=0.1, handlelength=1.1, prop={'size': 11}, frameon=False)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'results/cleared_rate_length.pdf')


def plot_trajectory_robutst_result():
    filetype = '.csv'
    file_list = []
    legend_list = []
    method_list = []

    for _, _, files in os.walk(f'results/csv'):
        for f in files:
            if filetype in f:
                file_list.append(f)
    file_list.sort(reverse=False)

    fig = plt.figure(figsize=(4, 3))

    for i, csv_file in enumerate(file_list):
        print(csv_file, end='\t')
        if 'robust' in csv_file:
            if '43' in csv_file:
                method_name = 'n=4>3'
                color = '#14517c'
            elif '54' in csv_file:
                method_name = 'n=5>4'
                color = '#2878b5'
            linestyle = 'solid'
        elif 'rl' in csv_file:
            if '3' in csv_file:
                method_name = 'n=3'
                color = '#05b9e2'
            elif '4' in csv_file:
                method_name = 'n=4'
                color = '#32b897'
            elif '5' in csv_file:
                method_name = 'n=5'
                color = '#54b345'
            linestyle = 'dashed'
        else:
            raise ValueError('Unknown method')
        print(method_name)
        method_list.append(method_name)

        csv_file = 'results/csv/' + csv_file
        trajectory_history = pd.read_csv(csv_file)
        trajectory_history = trajectory_history.sort_values('length')
        trajectory_mean = trajectory_history.rolling(400, on='length', min_periods=20, center=True).mean()
        trajectory_std = trajectory_history.rolling(400, on='length', min_periods=20, center=True).std()

        trajectory_mean = trajectory_mean[trajectory_mean['length'] <= 500]
        trajectory_std = trajectory_std[trajectory_std['length'] <= 500]

        line = plt.plot(trajectory_mean.length, trajectory_mean.safe, color=color, linewidth=1.5, zorder=10-i,
                        linestyle=linestyle)
        if 'robust' in csv_file:
            plt.fill_between(trajectory_mean.length, trajectory_mean.safe - trajectory_std.safe,
                             trajectory_mean.safe + trajectory_std.safe, color=color, alpha=0.15)
        # line = plt.plot(trajectory_mean.length, trajectory_mean.explore, color=color, linewidth=1.5, zorder=10 - i,
        #                 linestyle=linestyle)
        # plt.fill_between(trajectory_mean.length, trajectory_mean.explore - trajectory_std.explore,
        #                  trajectory_mean.explore + trajectory_std.explore, color=color, alpha=0.15)
        legend_list.append(line[0])
        plt.xticks(size=10)
        plt.xlabel('Length', fontdict={'size': 12})
        plt.xlim(0, 500)
        # plt.tick_params(bottom=False, left=False, axis='both', pad=0.1)
        plt.ylabel('Cleared Rate', fontdict={'size': 12})
        plt.yticks(size=10)
        plt.ylim(0, 1)

    plt.legend(legend_list, method_list, labelspacing=0.1, borderaxespad=0.1, handlelength=1.1, prop={'size': 11},
               frameon=False)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'results/cleared_rate_comp.pdf')


def plot_training_curve():
    filetype = '.csv'
    file_list = []
    legend_list = []
    method_list = []

    for _, _, files in os.walk(f'results/ablation'):
        for f in files:
            if filetype in f:
                file_list.append(f)
    file_list.sort(reverse=False)

    fig = plt.figure(figsize=(4, 3))


    for cnt, tag in enumerate(['Max']):
        linestyle = 'solid'
        for i, csv_file in enumerate(file_list):
            if tag in csv_file:
                if 'explore' in csv_file:
                    method_name = 'ViPER'
                    color = '#14517c'
                elif 'noMAAC' in csv_file:
                    method_name = 'ViPER w/o MAAC'
                    color = '#009E73'
                elif 'noGT-' in csv_file:
                    method_name = 'ViPER w/o GT'
                    color = '#f05260'
                elif 'noGTMAAC' in csv_file:
                    method_name = 'ViPER w/o MAAC & GT'
                    color = '#ff8000'
                else:
                    raise ValueError('Unknown method')
                print(csv_file)
                method_list.append(method_name)
            else:
                continue

            csv_file = 'results/ablation/' + csv_file
            trajectory_history = pd.read_csv(csv_file)
            trajectory_history = trajectory_history.sort_values('Step')
            trajectory_mean = trajectory_history.rolling(200, on='Step', min_periods=20, center=True).mean()
            trajectory_std = trajectory_history.rolling(200, on='Step', min_periods=20, center=True).std()

            trajectory_mean = trajectory_mean[trajectory_mean['Step'] <= 40000]
            trajectory_std = trajectory_std[trajectory_std['Step'] <= 40000]

            line = plt.plot(trajectory_mean.Step, trajectory_mean.Value, color=color, linewidth=1.5, zorder=10 - i,
                            linestyle=linestyle)

            plt.fill_between(trajectory_mean.Step, trajectory_mean.Value - 0.2*trajectory_std.Value,
                             trajectory_mean.Value + 0.2*trajectory_std.Value, color=color, alpha=0.15)
            # line = plt.plot(trajectory_mean.length, trajectory_mean.explore, color=color, linewidth=1.5, zorder=10 - i,
            #                 linestyle=linestyle)
            # plt.fill_between(trajectory_mean.length, trajectory_mean.explore - trajectory_std.explore,
            #                  trajectory_mean.explore + trajectory_std.explore, color=color, alpha=0.15)
            legend_list.append(line[0])
            # plt.xticks(size=10)
            plt.xticks([0, 10000, 20000, 30000, 40000],
                       ["0", "10k", "20k", "30k", "40k"], size=10)
            plt.xlabel('Training Episode', fontdict={'size': 12})
            plt.xlim(0, 40000)
            # plt.tick_params(bottom=False, left=False, axis='both', pad=0.1)
            plt.ylabel('Averaged Length', fontdict={'size': 12})
            plt.yticks(size=10)
            # plt.ylim(0, 1)
        print(method_list)
        plt.legend(legend_list, method_list, labelspacing=0.1, borderaxespad=0.1, handlelength=1.1, prop={'size': 11},
                   frameon=False)
        plt.tight_layout()
        plt.show()
        # plt.savefig(f'results/ablation_curve_length.pdf')



if __name__ == '__main__':
   # plot_trajectory_history_result()
   # plot_trajectory_robutst_result()
   plot_training_curve()
