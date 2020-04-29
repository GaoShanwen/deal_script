import os
import numpy as np
import cv2
import json
import sys
import torch
from PIL import Image
from tqdm import tqdm
import copy
import shapely.geometry as shgeo

sys.path.append('./')

from pet.utils.data.structures.bounding_box import BoxList
from pet.utils.data.structures.mask import Mask
from pet.utils.data.structures.parsing_poly import ParsingPoly
from pet.utils.data.structures.boxlist_ops import remove_boxes_by_center, remove_boxes_by_overlap


class SplitBase(object):
    def __init__(self, ori_img_dir, ori_json_path, out_img_dir, out_json_path,
                 gap=200, sub_size=800, thresh=0.7, ext='.png', rate=1):
        self.ori_img_dir = ori_img_dir
        self.ori_json_path = ori_json_path
        self.out_img_dir = out_img_dir
        self.out_json_path = out_json_path

        self.sub_size = sub_size
        self.slide = self.sub_size - gap
        self.thresh = thresh
        self.rate = rate
        self.ext = ext
        self.padding = True
        self.pad_pixel = (0, 0, 0)

        with open(self.ori_json_path, 'r') as load_f:
            load_dict = json.load(load_f)
            self.annotations = load_dict["annotations"]
            self.images = load_dict["images"]
            self.categories = load_dict["categories"]

        self.result_images = []
        self.result_annos = []
        self.image_id = 1
        self.ins_id = 1

        self.save_json()

    def parse_anns(self, image_size, anns):
        boxes = [ann['bbox'] for ann in anns]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        bbox = BoxList(boxes, image_size, mode="xywh").convert("xyxy")

        classes = [ann['category_id'] for ann in anns]
        classes = torch.tensor(classes)
        bbox.add_field("labels", classes)

        masks = [ann['segmentation'] for ann in anns]
        masks = Mask(masks, image_size, mode='poly')
        bbox.add_field("masks", masks)

        parsing = [ann['parsing'] for ann in anns]
        parsing = ParsingPoly(parsing, image_size)
        bbox.add_field("parsing", parsing)
        return bbox

    def load_anns(self, bbox, sub_img_name):
        labels = bbox.get_field("labels").tolist()
        masks = bbox.get_field("masks").instances.polygons
        parsings = bbox.get_field("parsing").parsing#.instances.polygons
        masks = [mask.polygons for mask in masks]
        new_masks = []
        for mask in masks:
            mask = [m.tolist() for m in mask]
            new_masks.append(mask)

        new_parsings = []
        for parsing in parsings:
            new_parts = []
            for part in parsing:
                new_part = []
                if len(part):
                    new_part = [p for p in part]#.tolist()
                new_parts.append(new_part)
            new_parsings.append(new_parts)

        bboxes = bbox.convert("xywh").bbox.tolist()
        n = len(labels)
        if not n:
            return None

        for i in range(n):
            label = labels[i]
            bbox = bboxes[i]
            mask = new_masks[i]
            parsing = new_parsings[i]
            mask_area = 0
            if len(mask) >= 1:
                Mask = [np.array(m).reshape(-1, 2).tolist() for m in mask]
                for m in Mask:
                    mask_area += shgeo.polygon.Polygon(m).area
            else:
                Mask = np.array(mask).reshape(-1, 2)
                mask_area = shgeo.polygon.Polygon(Mask).area
            annotation = {'segmentation': mask, 'parsing': parsing, 'area': round(mask_area), 'iscrowd': 0,
                          'image_id': self.image_id, 'bbox': bbox,
                          'category_id': label, 'id': self.ins_id}
            self.result_annos.append(annotation)
            self.ins_id += 1
        img_info = {'file_name': sub_img_name, 'width': self.sub_size, 'id': self.image_id, 'height': self.sub_size}
        self.result_images.append(img_info)
        self.image_id += 1

    def image_crop_with_padding(self, img, crop_region, save_path):
        left, up, right, down = crop_region
        sub_img = copy.deepcopy(img[up: (up + self.sub_size), left: (left + self.sub_size)])
        h, w, c = np.shape(sub_img)
        if self.padding:
            out_img = np.zeros((self.sub_size, self.sub_size, 3))
            out_img[0:h, 0:w, :] = sub_img
            cv2.imwrite(save_path, out_img)
        else:
            cv2.imwrite(save_path, out_img)

    def targets_crop(self, targets, crop_region, crop_shape):
        set_left, set_up, right, bottom = crop_region
        crop_targets = targets.crop(crop_region)
        targets = targets.move((set_left, set_up))
        targets = remove_boxes_by_overlap(targets, crop_targets, self.thresh)
        targets = targets.set_size(crop_shape)
        return targets

    def img_deal(self, img_ins):
        index = img_ins["id"]
        weight, height = img_ins["width"], img_ins["height"]
        img_size = (weight, height)
        img_annos = [anno for anno in self.annotations if anno["image_id"] == index]

        name = img_ins["file_name"][:-4]
        image = cv2.imread(os.path.join(self.ori_img_dir, name+self.ext))

        out_base_name = name + '__' + str(self.rate) + '__'
        left, up = 0, 0
        while left < weight:
            if (left + self.sub_size) >= weight:
                left = max(weight - self.sub_size, 0)
            up = 0
            while up < height:
                if (up + self.sub_size) >= height:
                    up = max(height - self.sub_size, 0)
                right = min(left + self.sub_size, weight - 1)
                down = min(up + self.sub_size, height - 1)

                sub_img_name = out_base_name + str(left) + '___' + str(up)+self.ext
                crop_region, crop_shape = (left, up, right, down), (self.sub_size, self.sub_size)
                save_path = os.path.join(self.out_img_dir, sub_img_name)
                self.image_crop_with_padding(image, crop_region, save_path)

                bbox = self.parse_anns(img_size, img_annos)
                bbox = self.targets_crop(bbox, crop_region, crop_shape)
                self.load_anns(bbox, sub_img_name)
                if (up + self.sub_size) >= height:
                    break
                else:
                    up = up + self.slide
            if (left + self.sub_size) >= weight:
                break
            else:
                left = left + self.slide

    def save_json(self):
        for img_info in tqdm(self.images):
            self.img_deal(img_info)
        dataset = {'images': self.result_images, 'categories': self.categories, 'annotations': self.result_annos}
        json.dump(dataset, open(self.out_json_path, 'w'))


if __name__ == '__main__':
    task_name = 'val'
    ori_img_dir = 'data/IPD-RSMO/data_ori/{}/images'.format(task_name)
    ori_json_path = 'data/IPD-RSMO/annotations/IPD_RSMO_{}_with_small.json'.format(task_name)
    out_img_dir = "data/IPD-RSMO/data_sub800_200/{}".format(task_name)
    out_json_path = 'data/IPD-RSMO/annotations/IPD_RSMO_{}_sub800_200_ws.json'.format(task_name)
    split = SplitBase(ori_img_dir, ori_json_path, out_img_dir, out_json_path)
