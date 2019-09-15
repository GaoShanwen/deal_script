# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import collections
from matplotlib.patches import Polygon

plt.switch_backend('agg')

def plot_pic(attributes, vis_data, vis_plot_alpha, vis_thin_alpha):
    x = np.array(attributes)
    key = list(vis_data.keys())
    y = np.array(vis_data[key[0]])

    # fig, axes = plt.subplots()
    plt.subplot(311)
    # 绘制曲线
    plt.plot(x, y, 'r', linewidth=2)
    ax1 = plt.gca()

    if 'size' in key[0]:
        # ax1.set_title(key[0].split('_')[0] + ' short size:')
        xticks = [int(attributes[0]), int(attributes[len(attributes) - 1])]
        xticks += list(range(int(attributes[0]+100)//200*200 + 200, int(attributes[len(attributes) - 1]-100), 200))
    else:
        # ax1.set_title(key[0].split('_')[0] + ' ratio:')
        xticks = [float('%.02f' % attributes[0])]
        xticks +=list(range(int(attributes[0])+1 , int(attributes[len(attributes) - 1])+1, 1))
        xticks +=[float('%.02f' % (attributes[len(attributes) - 1]))]
    # 坐标轴设置
    # axes.set_xticks(xticks)
    plt.xticks(rotation=45)
    # dim = (xticks[5]-xticks[0])//5
    # ax1.xaxis.set_ticks(np.arange(xticks[0], xticks[5] +dim, dim))
    plt1_Y_min_value, plt1_Y_max_value = 0, 1#900, 1400
    # axes.set_yticks([])
    ax1.yaxis.set_ticks(np.arange(plt1_Y_min_value, plt1_Y_max_value +0.1, 0.1))
    # plt.ylim(ymax=plt1_Y_max_value, ymin=plt1_Y_min_value)
    plt.grid(b=True)#, axis='y'
    if 'size' in key[0]:
        plt.figtext(0.9, 0.05, '$X:pixel$')
    else:
        plt.figtext(0.9, 0.05, '$X:w/h$')
    plt.figtext(0.1, 0.9, '$Y$')
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    x = np.array(attributes)
    key = list(vis_plot_alpha.keys())
    y = np.array(vis_plot_alpha[key[0]])

    plt.subplot(312)
    # 绘制曲线
    plt.plot(x, y, 'r', linewidth=2)
    ax1 = plt.gca()

    if 'size' in key[0]:
        # ax1.set_title(key[0].split('_')[0] + ' short size:')
        xticks = [int(attributes[0]), int(attributes[len(attributes) - 1])]
        xticks += list(range(int(attributes[0]+100)//200*200 + 200, int(attributes[len(attributes) - 1]-100), 200))
        # xticks+=[800]
    else:
        # ax1.set_title(key[0].split('_')[0] + ' ratio:')
        xticks = [float('%.02f' % attributes[0])]
        xticks +=list(range(int(attributes[0])+1 , int(attributes[len(attributes) - 1])+1, 1))
        xticks +=[float('%.02f' % (attributes[len(attributes) - 1]))]

    # 坐标轴设置
    # axes.set_xticks(xticks)
    plt.xticks(rotation=45)
    # dim = (xticks[5]-xticks[0])//5
    # ax1.xaxis.set_ticks(np.arange(xticks[0], xticks[5] +dim, dim))
    plt1_Y_min_value, plt1_Y_max_value = 0, 1#900, 1400
    # axes.set_yticks([])
    ax1.yaxis.set_ticks(np.arange(plt1_Y_min_value, plt1_Y_max_value +0.1, 0.1))
    # plt.ylim(ymax=plt1_Y_max_value, ymin=plt1_Y_min_value)
    plt.grid(b=True)#, axis='y'
    if 'size' in key[0]:
        plt.figtext(0.9, 0.05, '$X:pixel$')
    else:
        plt.figtext(0.9, 0.05, '$X:w/h$')
    plt.figtext(0.1, 0.9, '$Y$')
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    x = np.array(attributes)
    key = list(vis_thin_alpha.keys())
    y = np.array(vis_thin_alpha[key[0]])

    plt.subplot(313)
    # 绘制曲线
    plt.plot(x, y, 'r', linewidth=2)
    ax1 = plt.gca()

    if 'size' in key[0]:
        # ax1.set_title(key[0].split('_')[0] + ' short size:')
        xticks = [int(attributes[0]), int(attributes[len(attributes) - 1])]
        xticks += list(
            range(int(attributes[0] + 100) // 200 * 200 + 200, int(attributes[len(attributes) - 1] - 100), 200))
        # xticks+=[800]
    else:
        # ax1.set_title(key[0].split('_')[0] + ' ratio:')
        xticks = [float('%.02f' % attributes[0])]
        xticks += list(range(int(attributes[0]) + 1, int(attributes[len(attributes) - 1]) + 1, 1))
        xticks += [float('%.02f' % (attributes[len(attributes) - 1]))]

    # 坐标轴设置
    # axes.set_xticks(xticks)
    plt.xticks(rotation=45)
    # dim = (xticks[5]-xticks[0])//5
    # ax1.xaxis.set_ticks(np.arange(xticks[0], xticks[5] +dim, dim))
    plt1_Y_min_value, plt1_Y_max_value = 0, 1  # 900, 1400
    # axes.set_yticks([])
    ax1.yaxis.set_ticks(np.arange(plt1_Y_min_value, plt1_Y_max_value + 0.1, 0.1))
    # plt.ylim(ymax=plt1_Y_max_value, ymin=plt1_Y_min_value)
    plt.grid(b=True)  # , axis='y'
    if 'size' in key[0]:
        plt.figtext(0.9, 0.05, '$X:pixel$')
    else:
        plt.figtext(0.9, 0.05, '$X:w/h$')
    plt.figtext(0.1, 0.9, '$Y$')
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    plt.savefig("{}_p.png".format(key[0]), bbox_inches=0, dpi=300)


def readfile(filename):
    attributes = []
    vis_data = {filename.split('/')[-1].split('.txt')[0]: []}
    vis_plot_alpha = {filename.split('/')[-1].split('.txt')[0]: []}
    vis_thin_alpha = {filename.split('/')[-1].split('.txt')[0]: []}
    # cut_num = 1
    # if 'test' in filename:
    #     cut_num =1
    # if 'train' in filename:
    #     cut_num =12
    with open(filename, 'r') as read_f:
        lines = read_f.readlines()
        splitlines = [x.strip().split(',') for x in lines]
        for iter, splitline in enumerate(splitlines):
            if len(splitline)<2:
                continue
            attributes.append(float(splitline[0]))
            # if float(splitline[1])>0.98:
            #     cut_num -= 1
            #     if cut_num==0:
            #         vis_data[filename.split('/')[-1].split('.txt')[0]].append(1.0)
            #         vis_plot_alpha[filename.split('/')[-1].split('.txt')[0]].append(0.0)
            #         vis_thin_alpha[filename.split('/')[-1].split('.txt')[0]].append(0.0)
            #         break
            vis_data[filename.split('/')[-1].split('.txt')[0]].append(float(splitline[1]))
            vis_plot_alpha[filename.split('/')[-1].split('.txt')[0]].append(float(splitline[2]))
            vis_thin_alpha[filename.split('/')[-1].split('.txt')[0]].append(float(splitline[3]))
    return attributes, vis_data, vis_plot_alpha, vis_thin_alpha


if __name__ == '__main__':
    # val_path = './result/statistics/val_hbox_ratios_log.txt'
    # attributes, vis_data, vis_plot_alpha, vis_thin_alpha = readfile(val_path)
    # plot_pic(attributes, vis_data, vis_plot_alpha, vis_thin_alpha)
    val_path = './result/statistics/val_pbox_ratios_log.txt'
    attributes, vis_data, vis_plot_alpha, vis_thin_alpha = readfile(val_path)
    plot_pic(attributes, vis_data, vis_plot_alpha, vis_thin_alpha)

