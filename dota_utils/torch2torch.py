import torch
import os
import sys
# sys.path.append('../cls')
# from collections import OrderedDict
import numpy as np

if __name__ == '__main__':
    load_dir = 'Pet-dev/ckpts/rcnn/DOTA/e2e_rrpn_FPN_X-101_C4_ref'
    model_path = os.path.join(load_dir, 'model_final.pth')
    # net, loss, optimizer= get_model('test', 'BCELoss')# test train

    torch_dict = torch.load(model_path)['model']

    dict_name = list(torch_dict)
    for i, k in enumerate(dict_name):
        print(k)
