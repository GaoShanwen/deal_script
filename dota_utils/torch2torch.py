import torch
import os
# from collections import OrderedDict
import numpy as np

class ModelMapping(object):
    def __init__(
        self,
        key_model_path,
        value_model_path,
        final_model_path,
        debug_flag=False,
        mapping_dict={},
    ):
        self.mapping_dict = mapping_dict
        self.final_model_path = final_model_path
        self.stati_model_dict = {}
        self.last_dict = {}

        self.key_model_dict = torch.load(key_model_path)['model']
        self.value_model_dict = torch.load(value_model_path)['state_dicts']

        print('-'*10, 'object model', '-'*10)
        self.shape_print(self.key_model_dict)
        print('-'*10, 'loaded model', '-'*10)
        self.shape_print(self.value_model_dict)

        self.stati_model(self.key_model_dict)
        self.weight_mapping()
        self.check_error()
        self.save_model()

    def stati_model(self, torch_dict):
        for k in torch_dict.keys():
            cur_key = np.array(torch_dict[k].cpu()).shape
            if cur_key not in self.stati_model_dict:
                self.stati_model_dict[cur_key] = 1
            else:
                self.stati_model_dict[cur_key] += 1

    def shape_print(self, model_dict):
        for key in model_dict.keys():
            print(str(key).ljust(60), np.array(model_dict[key].cpu()).shape)#

    def weight_mapping(self):
        for cur_read_key in self.key_model_dict.keys():
            cur_read_shape = np.array(self.key_model_dict[cur_read_key].cpu()).shape
            if self.stati_model_dict[cur_read_shape] == 1:
                for k in self.value_model_dict.keys():
                    cur_save_shape = np.array(self.value_model_dict[k].cpu()).shape
                    if cur_read_shape == cur_save_shape:
                        print(k.ljust(60), cur_read_key)
                        self.last_dict[cur_read_key] = self.value_model_dict[k]
                        break
                if cur_read_key not in self.last_dict:
                    print(str(cur_read_key).ljust(60))
                    self.last_dict[cur_read_key] = self.value_model_dict[cur_read_key]
            else:
                cur_read_key_new = cur_read_key
                for replace_key in self.mapping_dict.keys():
                    cur_read_key_new = cur_read_key_new.replace(replace_key,  self.mapping_dict[replace_key])
                if cur_read_key_new not in self.value_model_dict.keys():
                    # print(cur_read_key.ljust(60), cur_read_key_new)
                    print(cur_read_key)
                else:
                    self.last_dict[cur_read_key] = self.value_model_dict[cur_read_key_new]

    def check_error(self):
        print(f'val model dict num: {len(self.value_model_dict.keys())}\n', 
              f'key model dict num: {len(self.key_model_dict.keys())}\n', 
              f'seccess dict num: {len(self.last_dict.keys())}')
        for key in self.key_model_dict.keys():
            if key not in self.last_dict.keys():
                self.last_dict[key] = self.key_model_dict[key]

                print(key)

    def save_model(self):
        final_model = {"model": self.last_dict}
        torch.save(final_model, self.final_model_path)


if __name__ == '__main__':
    model_path1 = './project/fast-reid/projects/attribute_recognition/logs/pa100k/strong_baseline/model_final.pth'
    model_path2 = './project/PAR_sb/exp_result/PA100k/img_model/ckpt_max.pth'
    final_model_path = './project/fast-reid/projects/attribute_recognition/logs/pa100k/strong_baseline/model_final1.pth'
    final_model_path = "model_final_.pth"
    replace_dict = {"backbone": "module.backbone",
                    "heads.classifier": "module.classifier.logits.0",
                    "heads.bottleneck.0": "module.classifier.logits.1",
                    }
    ModelMapping(model_path1, model_path2, final_model_path, replace_dict)

    model_path1 = "./project/aifwcls/projects/mobilenet-v1_imagenet/output/model_final.pth"
    model_path2 = "./.torch/fvcore_cache/models/mobilenet_v2-b0353104.pth"
