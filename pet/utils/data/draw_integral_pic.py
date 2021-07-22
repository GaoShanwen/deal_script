import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt


cate_map = {
    "owner": [-1,], 
    "order": [14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48], 
    "other": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,],
    "all": []
    }


keys = ["owner", "order", "other", "all"]
colors = ['blue', 'green', 'orange', 'red']


class StatisticDistance(object):
    def __init__(self, src_path, dst_dir):
        self.src_path = src_path
        prefix = src_path.split('_')[-1].replace('.txt', '')

        for cate in keys:
            self.dst_path = os.path.join(dst_dir, f'{prefix}_{cate}.txt')
            self.get_result(cate)
            self.save_result()
        
    def statistics_instance(self, obj):
        value = round(math.log(obj, 10), 4)
        if value not in self.distance_dict.keys():
            self.distance_dict[value] = 0
        self.distance_dict[value] += 1

    def get_result(self, cate):
        self.inst_count = 0
        self.distance_dict = {}
        with open(self.src_path, 'r') as load_f:
            _ = load_f.readline()
            pred_list = [
                float(line.strip().split(',')[-1]) for line in load_f.readlines()
                if int(line.strip().split('_')[0]) in cate_map[cate] or cate == "all"
                ]
        load_f.close()
        
        self.inst_count = len(pred_list)
        for pred in pred_list:
            self.statistics_instance(pred)

    def save_result(self):
        key_sort = list(self.distance_dict.keys())
        key_sort.sort()
        with open(self.dst_path, "w") as save_f:
            line = 'instance_number: {} distance_num: {} '.format(self.inst_count, len(self.distance_dict.keys()))
            save_f.writelines(line + '\n')
            inter_num = 0
            for key in key_sort:
                inter_num += self.distance_dict[key]
                rbox_info ='{}'.format(self.distance_dict[key]/len(self.distance_dict.keys()))
                line = "{},{},{}".format(float('%.06f' % key), (inter_num/self.inst_count), rbox_info)
                save_f.writelines(line + '\n')
        save_f.close()


# class DrawStatisticResult(object):
#     def __init__(self, file_path):
#         self.attributes = []
#         self.vis_data = {file_path.split('/')[-1].split('.txt')[0]: []}
#         self.readfile(file_path)
#         self.plot_pic()

#     def readfile(self, filename):
#         with open(filename, 'r') as read_f:
#             lines = read_f.readlines()
#             splitlines = [x.strip().split(',') for x in lines]
#             for iter, splitline in enumerate(splitlines):
#                 if len(splitline) < 2:
#                     continue
#                 if 'size' in filename:
#                     self.attributes.append(float(splitline[0]))
#                 else:
#                     self.attributes.append(float(splitline[0]))
#                 self.vis_data[filename.split('/')[-1].split('.txt')[0]].append(float(splitline[1]))

#     def plot_pic(self):
#         x = np.array(self.attributes)
#         key = list(self.vis_data.keys())
#         y = np.array(self.vis_data[key[0]])

#         fig, axes = plt.subplots()
#         # 绘制曲线
#         plt.plot(x, y, 'r', linewidth=2)
#         ax1 = plt.gca()

#         ax1.set_title(key[0].split('_')[0])
#         xticks = [float('%.04f' % self.attributes[0])]
#         xticks += list(
#             np.arange(self.attributes[0], 
#             self.attributes[len(self.attributes) - 1], 0.2)
#             )  # +1
#         xticks += [float('%.04f' % (self.attributes[len(self.attributes) - 1]))]

#         # 坐标轴设置
#         axes.set_xticks(xticks)
#         # print(type(x), np.power(10, x))
#         plt.xticks(xticks, np.around(np.power(10,xticks), decimals=3))
#         plt.xticks(rotation=45)
#         # dim = (xticks[5]-xticks[0])//5
#         # ax1.xaxis.set_ticks(np.arange(xticks[0], xticks[5] +dim, dim))
#         plt1_Y_min_value, plt1_Y_max_value = 0, 1  # 900, 1400
#         axes.set_yticks([])
#         ax1.yaxis.set_ticks(np.arange(plt1_Y_min_value, plt1_Y_max_value + 0.1, 0.1))
#         # plt.ylim(ymax=plt1_Y_max_value, ymin=plt1_Y_min_value)
#         plt.grid(b=True)  # , axis='y'
#         if 'size' in key[0]:
#             plt.figtext(0.9, 0.05, '$X:pixel$')
#         else:
#             plt.figtext(0.9, 0.05, '$X:w/h$')
#         plt.figtext(0.1, 0.9, '$Y$')
#         plt.xticks(fontsize=5)
#         plt.yticks(fontsize=5)

#         # it's with question, you can show and save it in plt.
#         plt.savefig("{}.png".format(key[0]), dpi=300, bbox_inches='tight')# bbox_inches=0,

class DrawStatisticResult(object):
    def __init__(self, file_dir):
        self.attributes = {}
        self.vis_data = {}
        for file_name in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file_name)
            self.attributes[file_name.split('_')[-1].split('.txt')[0]] = []
            self.vis_data[file_name.split('_')[-1].split('.txt')[0]] = []
            self.readfile(file_path)
        self.plot_pic(file_name.split('_')[-2])

    def readfile(self, filename):
        with open(filename, 'r') as read_f:
            _ = read_f.readline()
            lines = read_f.readlines()
            splitlines = [x.strip().split(',') for x in lines]
            cate = filename.split('_')[-1].split('.txt')[0]
            for iter, splitline in enumerate(splitlines):
                self.attributes[cate].append(float(splitline[0]))
                self.vis_data[cate].append(float(splitline[1]))
    
    def plot_pic(self, name):
        fig, axes = plt.subplots()
        for iter, key in enumerate(keys):
            x = np.array(self.attributes[key])
            y = np.array(self.vis_data[key])
            # 绘制曲线
            plt.plot(x, y, colors[iter], linewidth=2, label=key)

        ax1 = plt.gca()

        # ax1.set_title(key[0].split('_')[0])
        xticks = [float('%.04f' % self.attributes["all"][0])]
        xticks += list(
            np.arange(self.attributes["all"][0],
            self.attributes["all"][len(self.attributes["all"])-1], 0.2)
            )
        xticks += [float('%.04f' % (self.attributes["all"][len(self.attributes["all"]) - 1]))]

        # 坐标轴设置
        # axes.set_xticks(xticks)
        # # print(type(x), np.power(10, x))
        plt.xticks(xticks, np.around(np.power(10, xticks), decimals=4))
        plt.xticks(rotation=45)
        # dim = (xticks[5]-xticks[0])//5
        # ax1.xaxis.set_ticks(np.arange(xticks[0], xticks[5] +dim, dim))
        plt1_y_min_value, plt1_y_max_value = 0, 1
        axes.set_yticks([])
        ax1.yaxis.set_ticks(np.arange(plt1_y_min_value, plt1_y_max_value + 0.1, 0.1))
        # plt.ylim(ymax=plt1_Y_max_value, ymin=plt1_Y_min_value)
        plt.grid(linestyle='--', b=True)#, axis='y'
        if 'size' in key[0]:
            plt.figtext(0.9, 0.05, '$X:pixel$')
        else:
            plt.figtext(0.9, 0.05, '$X:distance$')
        plt.figtext(0.1, 0.9, '$Y$')
        plt.title(f'{name}')#.format()
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend(loc='upper left')  # , ncol=3

        # it's with question, you can show and save it in plt.
        plt.savefig(f"ratio_{name}.png", dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep Sort")
    parser.add_argument("-s", "--src_path", type=str, default='', help="source txt path")
    parser.add_argument("-d", "--dst_path", type=str, default='', help="object txt path")
    args = parser.parse_args()
    StatisticDistance(args.src_path, args.dst_path)
    DrawStatisticResult(args.dst_path)
