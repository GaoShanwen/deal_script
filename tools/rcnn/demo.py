import os
import torch
import argparse

import _init_paths
from pet.rcnn.core.config import cfg, merge_cfg_from_file, merge_cfg_from_list
from pet.rcnn.modeling.rbox_model_builder import Generalized_RRCNN
from pet.utils.checkpointer import CheckPointer
from pet.utils.checkpointer import get_weights, load_weights
from pet.utils.net import convert_bn2affine_model
from pet.utils.data.dataset_catalog import contains, get_im_dir
from tools.rcnn.predictor import inference
from pet.utils.misc import mkdir_p

# Parse arguments
parser = argparse.ArgumentParser(description='Pet Model Testing')
parser.add_argument('--cfg', dest='cfg_file',
                    help='optional config file',
                    default='./cfgs/rcnn/mscoco/e2e_faster_rcnn_R-50-FPN_1x.yaml', type=str)
parser.add_argument('--gpu_id', type=str, default='0,1,2,3,4,5,6,7', help='gpu id for evaluation')
parser.add_argument('--src_folder', type=str, default='', help='optional image dir',)
parser.add_argument('--range', help='start (inclusive) and end (exclusive) indices', type=int, nargs=2)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('opts', help='See pet/rcnn/core/config.py for all options',
                    default=None,
                    nargs=argparse.REMAINDER)


def initialize_model_from_cfg():
    """Initialize a model from the global cfg. Loads test-time weights and
    set to evaluation mode.
    """
    # model = Generalized_RCNN()
    model = Generalized_RRCNN()
    # if cfg.MODEL.BATCH_NORM == 'freeze':
    #     model = convert_bn2affine_model(model)

    # # Load trained model
    cfg.TEST.WEIGHTS = get_weights(cfg.CKPT, cfg.TEST.WEIGHTS)
    # load_weights(model, cfg.TEST.WEIGHTS)
    # checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
    # _ = checkpointer.load(cfg.MODEL.WEIGHT)
    checkpointer = CheckPointer(cfg.CKPT, weights_path=cfg.TEST.WEIGHTS, #auto_resume=cfg.TRAIN.AUTO_RESUME,
                                local_rank=args.local_rank)

    # Load model or random-initialization
    model = checkpointer.load_model(model, convert_conv1=cfg.MODEL.CONV1_RGB2BGR)
    model.eval()
    model.to(torch.device(cfg.DEVICE))

    return model


def get_file_paths_recursive(dataset_list=None, file_ext=None):
    """ Get the absolute path of all files in given folder recursively
    :param dataset_list:
    :param file_ext:
    :return:
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    for dataset_name in dataset_list:
        assert contains(dataset_name), 'Unknown dataset name: {}'.format(dataset_name)
        assert os.path.exists(get_im_dir(dataset_name)), 'Im dir \'{}\' not found'.format(get_im_dir(dataset_name))
        # logging_rank('Creating: {}'.format(dataset_name), local_rank=local_rank)

    file_list = []
    for dataset_name in dataset_list:
        folder = get_im_dir(dataset_name)
        if folder is None:
            return file_list

        for dir_path, dir_names, file_names in os.walk(folder):
            for file_name in file_names:
                if file_ext is None:
                    file_list.append(os.path.join(dir_path, file_name))
                    continue
                if file_name.endswith(file_ext):
                    file_list.append(os.path.join(dir_path, file_name))
    return file_list


if __name__ == "__main__":
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)

    if not os.path.isdir(os.path.join(cfg.CKPT, 'test')):
        mkdir_p(os.path.join(cfg.CKPT, 'test'))
    if cfg.VIS.ENABLED:
        if not os.path.exists(os.path.join(cfg.CKPT, 'vis')):
            mkdir_p(os.path.join(cfg.CKPT, 'vis'))

    det_net = initialize_model_from_cfg()
    file_paths = get_file_paths_recursive(cfg.TEST.DATASETS, '.png')
    des_txt_folder = os.path.join(cfg.CKPT, 'test')
    des_img_folder = os.path.join(cfg.CKPT, 'vis')
    print('************* META INFO ***************')
    print('config_file:', args.cfg_file)
    print('weights_dir:', cfg.CKPT)
    print('test_images:{} / num:{}'.format(cfg.TEST.DATASETS, len(file_paths)))
    print('result_text:', des_txt_folder)
    print('result_image:', des_img_folder)
    print('***************************************')

    threshold = {'__background__': 0,
            'plane': 0.3, 'baseball-diamond': 0.3, 'bridge': 0.0001, 'ground-track-field': 0.3, 'small-vehicle': 0.2,
            'large-vehicle': 0.1, 'ship': 0.05, 'tennis-court': 0.3, 'basketball-court': 0.3, 'storage-tank': 0.2,
            'soccer-ball-field': 0.3, 'roundabout': 0.1, 'harbor': 0.0001, 'swimming-pool': 0.1, 'helicopter': 0.2}
    inference(det_net, file_paths, des_txt_folder, des_img_folder, threshold)
