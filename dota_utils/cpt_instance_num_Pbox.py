import os
import sys
sys.path.append('./')
from DOTA_devkit.dota_utils import wordname_15
from DOTA_devkit import dota_utils as util
import collections
import math
import numpy as np


def dots4ToRec8(bbox):
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3],
    return xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax


def multi_mkdir(path):
    if not os.path.isdir(path):
        multi_mkdir(os.path.split(path)[0])
    else:
        return
    os.mkdir(path)


def get_rbox(poly):
    # print(poly)
    # gt_ind = gt_line.split(' ')
    pt1 = (int(float(poly[0])), int(float(poly[1])))
    pt2 = (int(float(poly[2])), int(float(poly[3])))
    pt3 = (int(float(poly[4])), int(float(poly[5])))
    # pt4 = (int(float(gt_ind[6])), int(float(gt_ind[7])))

    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

    angle = 0

    if edge1 > edge2:
        width = edge1
        height = edge2
        if pt1[0] - pt2[0] != 0:
            angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
        else:
            angle = 90.0
    elif edge2 >= edge1:
        width = edge2
        height = edge1
        # print pt2[0], pt3[0]
        if pt2[0] - pt3[0] != 0:
            angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
        else:
            angle = 90.0
    if angle < -45.0:
        angle = angle + 180

    x_ctr = float(pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
    y_ctr = float(pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2
    return [x_ctr, y_ctr, width, height, angle]


def rbox2poly(rbox):#, height, width):
    cx, cy, w, h, angle = rbox
    lt = [cx - w / 2, cy - h / 2, 1]
    rt = [cx + w / 2, cy - h / 2, 1]
    lb = [cx - w / 2, cy + h / 2, 1]
    rb = [cx + w / 2, cy + h / 2, 1]

    pts = []

    # angle = angle * 0.45

    pts.append(lt)
    pts.append(rt)
    pts.append(rb)
    pts.append(lb)

    angle = -angle #90

    # if angle != 0:
    cos_cita = np.cos(np.pi / 180 * angle)
    sin_cita = np.sin(np.pi / 180 * angle)

    # else :
    #	cos_cita = 1
    #	sin_cita = 0

    M0 = np.array([[1, 0, 0], [0, 1, 0], [-cx, -cy, 1]])
    M1 = np.array([[cos_cita, sin_cita, 0], [-sin_cita, cos_cita, 0], [0, 0, 1]])
    M2 = np.array([[1, 0, 0], [0, 1, 0], [cx, cy, 1]])
    rotation_matrix = M0.dot(M1).dot(M2)

    rotated_pts = np.dot(np.array(pts), rotation_matrix)

    # rotated_pts = over_bound_handle(rotated_pts, height, width)

    return rotated_pts #return_bboxes


def get_pbox(rbox):#, img_height, img_width):
    # print(rbox)
    poly = rbox2poly(rbox)#, img_height, img_width)
    # xmin, ymin, xmax, ymax = min(poly[0::2]), min(poly[1::2]), \
    #                          max(poly[0::2]), max(poly[1::2])
    xmin, xmax, ymin, ymax = min(poly[0][0], min(poly[1][0], min(poly[2][0], poly[3][0]))), \
                             max(poly[0][0], max(poly[1][0], max(poly[2][0], poly[3][0]))), \
                             min(poly[0][1], min(poly[1][1], min(poly[2][1], poly[3][1]))), \
                             max(poly[0][1], max(poly[1][1], max(poly[2][1], poly[3][1])))
    high ,width = ymax - ymin, xmax - xmin
    pt1, pt2, pt3, pt4 = poly[0:5]
    # print(pt1, pt2, pt3, pt4)

    x_list = [pt1[0], pt2[0], pt3[0], pt4[0]]
    y_list = [pt1[1], pt2[1], pt3[1], pt4[1]]
    # arr_x = np.array(x_list)
    # arr_y = np.array(y_list)
    # left_plot_y = min(arr_y[np.where(arr_x <= (xmin + width*0.01))])
    # left_plot_x = max(arr_x[np.where(arr_y <= (ymin + high*0.01))])
    # # print(arr_y[np.where(arr_x == xmin)], arr_x[np.where(arr_y == ymin)])
    left_plot_y = y_list[x_list.index(xmin)]
    left_plot_x = x_list[y_list.index(ymin)]
    alpha = (left_plot_y - ymin) / high
    beta = (left_plot_x - xmin) / width
    if max(alpha, beta) < 0.5 or min(alpha, beta) > 0.5:
        thin_flag = True
        if max(alpha, beta) < 0.01 or min(alpha, beta)>0.99:
            thin_flag = False
            # print(rbox)
    else:
        thin_flag = False
    # print([xmin, ymin, width, high, alpha, thin_flag])
    return [xmin, ymin, width, high, alpha, thin_flag]


def pbox2poly(pbox):
    xmin, ymin, width, high, alpha, thin_flag = pbox[0:6]
    distance_square1 = width * width + high * high * (1 - 2 * alpha) * (1 - 2 * alpha)
    # distance_square2 = high * high + width * width * (1 - 2 * beta) * (1 - 2 * beta)
    # print(xmin, ymin, width, high, alpha, thin_flag)
    # print((distance_square1 - high * high) / (width * width))
    # [133, 1852, 153, 1842, 173, 1882, 154, 1892]
    # [1.530e+02 1.892e+03 1.000e+00] [1.330e+02 1.852e+03 1.000e+00] [1.530e+02 1.842e+03 1.000e+00] [1.730e+02 1.882e+03 1.000e+00]
    # 133.00000028328895 1841.9999999999998 39.99999943342232 50.0 0.1999999924456324 False
    beta_get1 = (1 - math.sqrt(max(distance_square1 - high * high, 0) / (width * width))) / 2
    beta_get2 = (1 + math.sqrt(max(distance_square1 - high * high, 0) / (width * width))) / 2
    if (thin_flag and alpha<0.5) or (not thin_flag and alpha>0.5):
        beta_get = min(beta_get1, beta_get2)
    else:
        beta_get = max(beta_get1, beta_get2)
    # print(alpha, beta, beta_get, distance_square1, distance_square2)
    poly8 = [[xmin, ymin + alpha * high], \
            [xmin + beta_get * width, ymin], \
            [xmin + width, ymin + (1 - alpha) * high], \
            [xmin + (1 - beta_get) * width, ymin + high]]
    # print(beta_get)
    return np.array(poly8)


def poly2pbox(poly):
    # poly = rbox2poly(rbox)#, img_height, img_width)
    poly[6], poly[7] = poly[0] + poly[4] -poly[2], poly[1] + poly[5] -poly[3]
    xmin, ymin, xmax, ymax = min(poly[0::2]), min(poly[1::2]), \
                             max(poly[0::2]), max(poly[1::2])
    # xmin, xmax, ymin, ymax = min(poly[0][0], min(poly[1][0], min(poly[2][0], poly[3][0]))), \
    #                          max(poly[0][0], max(poly[1][0], max(poly[2][0], poly[3][0]))), \
    #                          min(poly[0][1], min(poly[1][1], min(poly[2][1], poly[3][1]))), \
    #                          max(poly[0][1], max(poly[1][1], max(poly[2][1], poly[3][1])))
    high ,width = ymax - ymin, xmax - xmin
    # pt1, pt2, pt3, pt4 = poly[0:5]
    pt1, pt2, pt3, pt4 = poly[0:2], poly[2:4], poly[4:6], poly[6:],
    # print(pt1, pt2, pt3, pt4)

    x_list = [pt1[0], pt2[0], pt3[0], pt4[0]]
    y_list = [pt1[1], pt2[1], pt3[1], pt4[1]]
    arr_x = np.array(x_list)
    arr_y = np.array(y_list)
    left_plot_y = min(arr_y[np.where(arr_x == xmin)])
    left_plot_x = max(arr_x[np.where(arr_y == ymin)])
    # print(arr_y[np.where(arr_x == xmin)], arr_x[np.where(arr_y == ymin)])
    # left_plot_y = y_list[x_list.index(xmin)]
    # left_plot_x = x_list[y_list.index(ymin)]
    alpha = (left_plot_y - ymin) / high
    beta = (left_plot_x - xmin) / width
    if max(alpha, beta) < 0.5 or min(alpha, beta) > 0.5:
        thin_flag = True
    else:
        thin_flag = False
    distance_square1 = width * width + high * high * (1 - 2 * alpha) * (1 - 2 * alpha)
    # distance_square2 = high * high + width * width * (1 - 2 * beta) * (1 - 2 * beta)
    beta_get1 = (1 - math.sqrt(max(distance_square1 - high * high, 0) / (width * width))) / 2
    beta_get2 = (1 + math.sqrt(max(distance_square1 - high * high, 0) / (width * width))) / 2
    if (thin_flag and alpha<0.5) or (not thin_flag and alpha>0.5):
        beta_get = min(beta_get1, beta_get2)
    else:
        beta_get = max(beta_get1, beta_get2)
    # # beta_get = 0
    # print(alpha, beta, beta_get, distance_square1, distance_square2)
    area = math.sqrt(high * high * alpha * alpha + width * width * beta_get * beta_get) * \
           math.sqrt(high * high * (1 - alpha) * (1 - alpha) + width * width * (1 - beta_get) * (1 - beta_get))
    return [xmin, ymin, width, high, alpha, thin_flag], area


def DOTA2COCO(srcpath, dest_dir, category_list):
    multi_mkdir(dest_dir)
    labelparent = os.path.join(srcpath, 'labelTxt-v1.0')

    inst_count = 0
    image_id = 0
    categories_dict = {}
    data_dict = {}
    for category in category_list:
        if category == '__background__':
            continue
        categories_dict[category] = collections.OrderedDict({'T': 0, 'S': 0, 'M': 0, 'L': 0, 'total': 0})
        data_dict[category] = []
    categories_dict['all'] = collections.OrderedDict({'T': 0, 'S': 0, 'M': 0, 'L': 0, 'total': 0})
    # with open(destfile, 'w') as f_out:
    filenames = util.GetFileFromThisRootDir(labelparent)
    for file in filenames:
        image_id = image_id + 1
        objects = util.parse_dota_poly2(file)
        if not len(objects):
            continue
        basename = util.custombasename(file)
        # data_dict[basename] = []
        for obj in objects:
            inst_count = inst_count + 1
            single_obj = {}
            single_obj['filename'] = basename
            # obj['poly'] = [1062.0, 1886.0, 1062.0, 1826.0, 1120.0, 1826.0000000000002, 1120.0, 1886.0000000000002]
            single_obj['bbox'] = pbox2poly(get_pbox(get_rbox(obj['poly'])))
            # pbox, obj['area'] = poly2pbox(obj['poly'])
            # single_obj['bbox'] = pbox2poly(pbox)
            if obj['area']  <16*16:
                categories_dict[obj['name']]['T'] += 1
            elif obj['area']<32*32:
                categories_dict[obj['name']]['S'] += 1
            elif obj['area']<96*96:
                categories_dict[obj['name']]['M'] += 1
            else:
                categories_dict[obj['name']]['L'] += 1
            # xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
            #                          max(obj['poly'][0::2]), max(obj['poly'][1::2])
            # single_obj['bbox'] = dots4ToRec8([xmin, ymin, xmax, ymax])
            # print(basename)
            # single_obj['bbox'] = pbox2poly(get_pbox(get_rbox(obj['poly'])))
            data_dict[obj['name']].append(single_obj)
            # data_dict['annotations']
            # single_obj['id'] = inst_count
            # break

        # if image_id>=2:
        # break
    # print(data_dict)
    print()
    if 'val' in dest_dir:
        print('| val/name:'.ljust(24), '| T:'.ljust(24), '| S:'.ljust(24), '| M:'.ljust(24),
              '| L:'.ljust(24), '| Total |'.ljust(26))
    else:
        print('| train/name:'.ljust(24), '| T:'.ljust(24), '| S:'.ljust(24), '| M:'.ljust(24),
              '| L:'.ljust(24), '| Total |'.ljust(26))
    print('| - | - | - | - | - | - |')
    for category in category_list:
        if category == '__background__':
            continue
        categories_dict[category]['total'] = categories_dict[category]['T'] + categories_dict[category]['S'] \
                                             + categories_dict[category]['M'] + categories_dict[category]['L']
        txt_file_path = os.path.join(dest_dir, category + '.txt')
        with open(txt_file_path, "w") as save_f:
            # print(category, len(data_dict[category]))
            for category_ins in data_dict[category]:
                # line = '{}'.format(category_ins['bbox']).replace(',', '')
                line = '{} {} {}'.format(category_ins['filename'], 1.0, category_ins['bbox'].tolist())
                line = line.replace('(', '').replace(',', '').replace(')', '') \
                    .replace('[', '').replace(']', '')
                save_f.writelines(line + '\n')
        save_f.close()
        print('| {}'.format(category).ljust(24),
              '| {}'.format(100* categories_dict[category]['T']/categories_dict[category]['total']).ljust(24),
              '| {}'.format(100* categories_dict[category]['S'] / categories_dict[category]['total']).ljust(24),
              '| {}'.format(100* categories_dict[category]['M'] / categories_dict[category]['total']).ljust(24),
              '| {}'.format(100* categories_dict[category]['L'] / categories_dict[category]['total']).ljust(24),
              '| {} |'.format(categories_dict[category]['total']))
            # print(line)
            # break
        categories_dict['all']['T'] += categories_dict[category]['T']
        categories_dict['all']['S'] += categories_dict[category]['S']
        categories_dict['all']['M'] += categories_dict[category]['M']
        categories_dict['all']['L'] += categories_dict[category]['L']
        categories_dict['all']['total'] += categories_dict[category]['total']
        # break
    print('| {}'.format('all').ljust(24),
          '| {}'.format(100 * categories_dict['all']['T'] / categories_dict['all']['total']).ljust(24),
          '| {}'.format(100 * categories_dict['all']['S'] / categories_dict['all']['total']).ljust(24),
          '| {}'.format(100 * categories_dict['all']['M'] / categories_dict['all']['total']).ljust(24),
          '| {}'.format(100 * categories_dict['all']['L'] / categories_dict['all']['total']).ljust(24),
          '| {} |'.format(categories_dict['all']['total']))
    print()
    print('img:{}, ins:{}'.format(image_id, inst_count))
    # print(categories_dict)
    # json.dump(data_dict, f_out)


if __name__ == '__main__':
    DOTA2COCO(r'./data/DOTA-v1/data_ori/train', r'./result/cpt_Pbox_train_label', wordname_15)
    DOTA2COCO(r'./data/DOTA-v1/data_ori/val', r'./result/cpt_Pbox_val_label', wordname_15)

