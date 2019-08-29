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

    torch_dict = torch.load(model_path)['model']##.state_dict() ['state_dict']
    # print('model layer1:', torch_dict['forw0.0.weight'].shape)# layer0conv_W,forw0.0.weight
    # print('torch:',  torch_dict['forw1.0.normal1.gama'][0,:,0,0,0])
    # print('torch:',  torch_dict['forw0.0.bias'])

    dict_name = list(torch_dict)
    for i, k in enumerate(dict_name):
        print(k)
