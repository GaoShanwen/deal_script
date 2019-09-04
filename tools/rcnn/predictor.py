# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import cv2
import torch
from torchvision import transforms as T
# from pet.utils.data import transforms as T
from pet.utils.data.structures.image_list import to_image_list
from PIL import Image, ImageDraw

from pet.rcnn.core.config import cfg
import numpy as np
import time
from tqdm import tqdm

# from maskrcnn_benchmark.modeling.detector import build_detection_model
# from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
# from maskrcnn_benchmark.structures.image_list import to_image_list
# from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
# from maskrcnn_benchmark import layers as L
# from maskrcnn_benchmark.utils import cv2_util
CATEGORIES = [
        '__background__',
        'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
        'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
        'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter'
    ]

class RRPNDemo(object):
    # COCO categories for pretty print


    def __init__(
        self,
        # cfg,
        model,
        confidence_threshold={},#0.7,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=224,
    ):
        # self.cfg = cfg.clone()
        self.model = model#build_detection_model(cfg)
        # self.model.eval()
        self.device = torch.device(cfg.DEVICE)
        # self.model.to(self.device)
        self.min_image_size = min_image_size

        # save_dir = cfg.OUTPUT_DIR
        # checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        # _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        # mask_threshold = -1 if show_mask_heatmaps else 0.5
        # self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.categories = CATEGORIES
        # self.show_mask_heatmaps = show_mask_heatmaps
        # self.masks_per_dim = masks_per_dim

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        # cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        # print(cfg.PIXEL_MEANS, cfg.PIXEL_STDS)
        normalize_transform = T.Normalize(
            # mean=cfg.RRPN_RCNN.INPUT.PIXEL_MEAN, std=cfg.RRPN_RCNN.INPUT.PIXEL_STD
            mean=cfg.PIXEL_MEANS, std=cfg.PIXEL_STDS#
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    # def build_transform(self, is_train=False):
    #     # if is_train:
    #     #     min_size = cfg.RRPN_RCNN.INPUT.MIN_SIZE_TRAIN
    #     #     max_size = cfg.RRPN_RCNN.INPUT.MAX_SIZE_TRAIN
    #     #     flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
    #     # else:
    #     #     min_size = cfg.RRPN_RCNN.INPUT.MIN_SIZE_TEST
    #     #     max_size = cfg.RRPN_RCNN.INPUT.MAX_SIZE_TEST
    #     #     flip_prob = 0
    #
    #     to_bgr255 = cfg.RRPN_RCNN.INPUT.TO_BGR255
    #     normalize_transform = T.Normalize(
    #         mean=cfg.RRPN_RCNN.INPUT.PIXEL_MEAN, std=cfg.RRPN_RCNN.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    #     )
    #     transform = T.Compose(
    #         [
    #             # T.Resize(min_size, max_size),
    #             # T.RandomHorizontalFlip(flip_prob),
    #             T.ToTensor(),
    #             normalize_transform,
    #         ]
    #     )
    #     return transform

    def run_on_opencv_image(self, image):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)

        result = image.copy()
        # if self.show_mask_heatmaps:
        #     return self.create_mask_montage(result, top_predictions)
        result = self.overlay_boxes(result, top_predictions)
        # if self.cfg.MODEL.MASK_ON:
        #     result = self.overlay_mask(result, top_predictions)
        result,  scores, labels= self.overlay_class_names(result, top_predictions)

        return result, top_predictions, scores, labels

    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        # print(image, type(image), image.numpy().shape)
        # cv2.imwrite('show1.png', image.numpy())
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, cfg.TEST.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        # print(np.array(image_list[:]).shape)
        # print(image_list)
        # image_list[0].save('demo_1.png')
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))
        # print('from compute_prediction:', prediction)
        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        # changed in 9.3/2019
        labels = predictions.get_field("labels")
        all_keep = torch.empty([1])
        for index, threshold_key in enumerate(CATEGORIES):
            if threshold_key == '__background__':
                continue
            category_keep = torch.nonzero(labels == index).squeeze(1)
            score_keep = torch.nonzero(scores[category_keep] >= self.confidence_threshold[threshold_key]).squeeze(1)
            if index == 1:
                all_keep = category_keep[score_keep]
            all_keep = torch.cat((all_keep, category_keep[score_keep]), 0)
        predictions = predictions[all_keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()
        '''
        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )
        '''
        return image

    def overlay_mask(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")

        colors = self.compute_colors_for_labels(labels).tolist()

        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None]
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.drawContours(image, contours, -1, color, 3)

        composite = image

        return composite

    def create_mask_montage(self, image, predictions):
        """
        Create a montage showing the probability heatmaps for each one one of the
        detected objects

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask`.
        """
        masks = predictions.get_field("mask")
        masks_per_dim = self.masks_per_dim
        masks = L.interpolate(
            masks.float(), scale_factor=1 / masks_per_dim
        ).byte()
        height, width = masks.shape[-2:]
        max_masks = masks_per_dim ** 2
        masks = masks[:max_masks]
        # handle case where we have less detections than max_masks
        if len(masks) < max_masks:
            masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
            masks_padded[: len(masks)] = masks
            masks = masks_padded
        masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
        result = torch.zeros(
            (masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
        )
        for y in range(masks_per_dim):
            start_y = y * height
            end_y = (y + 1) * height
            for x in range(masks_per_dim):
                start_x = x * width
                end_x = (x + 1) * width
                result[start_y:end_y, start_x:end_x] = masks[y, x]
        return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.categories[i] for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(label, score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
            )

        return image,  scores, labels

def over_bound_handle(pts, img_height, img_width):

    pts[np.where(pts < 0)] = 1

    pts[np.where(pts[:,0] > img_width), 0] = img_width-1
    pts[np.where(pts[:,1] > img_height), 1] = img_height-1

    return pts

def write_result_ICDAR_RRPN2polys(im_file, dets, threshold, result_dir, height, width, scores, labels):
    file_spl = im_file.split('/')
    file_name = file_spl[len(file_spl) - 1]
    file_name_arr = file_name.split(".")

    file_name_str = file_name_arr[0]

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    result = os.path.join(result_dir, "res_" + file_name_str + ".txt")

    return_bboxes = []

    if not os.path.isfile(result):
        os.mknod(result)
    result_file = open(result, "w")

    result_str = ""

    for idx in range(len(dets)):
        cx, cy, w, h, angle = dets[idx][0:5]
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

        angle = 90 - angle

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

        # print im
        # print im.shape
        #			im = im.transpose(2,0,1)

        # det_str = str(int(rotated_pts[0][0])) + "," + str(int(rotated_pts[0][1])) + "," + \
        #          str(int(rotated_pts[1][0])) + "," + str(int(rotated_pts[1][1])) + "," + \
        #          str(int(rotated_pts[2][0])) + "," + str(int(rotated_pts[2][1])) + "," + \
        #          str(int(rotated_pts[3][0])) + "," + str(int(rotated_pts[3][1])) + "\r\n"

        # rotated_pts = rotated_pts[:,0:2]


        # if (dets[idx][5] > threshold):
        rotated_pts = over_bound_handle(rotated_pts, height, width)
        det_str = str(int(rotated_pts[0][0])) + "," + str(int(rotated_pts[0][1])) + "," + \
                  str(int(rotated_pts[1][0])) + "," + str(int(rotated_pts[1][1])) + "," + \
                  str(int(rotated_pts[2][0])) + "," + str(int(rotated_pts[2][1])) + "," + \
                  str(int(rotated_pts[3][0])) + "," + str(int(rotated_pts[3][1])) + "," + \
                  str(labels[idx])+',' + str(scores[idx])+"\r\n"
        # det_str = str(cx)+','+str(cy)+','+str(w)+','+str(h)+','+str(angle)+','+str(labels[idx])+',' + str(scores[idx])+"\r\n"

        result_str = result_str + det_str
        return_bboxes.append(dets[idx])

        # print rotated_pts.shape

    result_file.write(result_str)
    result_file.close()

    return return_bboxes

def vis_image(img, boxes, cls_prob=None, mode=1, font_file='./fonts/ARIAL.TTF'):
    # img = cv2.imread(image_path)
    # cv2.setMouseCallback("image", trigger)
    # font = ImageFont.truetype(font_file, 32)
    font = cv2.FONT_HERSHEY_SIMPLEX
    draw = ImageDraw.Draw(img)

    for idx in range(len(boxes)):
        cx, cy, w, h, angle = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]
        # need a box score larger than thresh

        lt = [cx - w / 2, cy - h / 2, 1]
        rt = [cx + w / 2, cy - h / 2, 1]
        lb = [cx - w / 2, cy + h / 2, 1]
        rb = [cx + w / 2, cy + h / 2, 1]

        pts = []

        pts.append(lt)
        pts.append(rt)
        pts.append(rb)
        pts.append(lb)

        angle = -angle

        # angle = 90 - angle

        cos_cita = np.cos(np.pi / 180 * angle)
        sin_cita = np.sin(np.pi / 180 * angle)

        M0 = np.array([[1, 0, 0], [0, 1, 0], [-cx, -cy, 1]])
        M1 = np.array([[cos_cita, sin_cita, 0], [-sin_cita, cos_cita, 0], [0, 0, 1]])
        M2 = np.array([[1, 0, 0], [0, 1, 0], [cx, cy, 1]])
        rotation_matrix = M0.dot(M1).dot(M2)

        rotated_pts = np.dot(np.array(pts), rotation_matrix)

        # rotated_pts[rotated_pts <= 0] = 1
        # rotated_pts[rotated_pts > img.shape[1]] = img.shape[1] - 1

        if mode == 1:
            draw.line((int(rotated_pts[0, 0]), int(rotated_pts[0, 1]), int(rotated_pts[1, 0]), int(rotated_pts[1, 1])), fill=(0, 255, 0))
            draw.line((int(rotated_pts[1, 0]), int(rotated_pts[1, 1]), int(rotated_pts[2, 0]), int(rotated_pts[2, 1])),
                      fill=(0, 255, 0))
            draw.line((int(rotated_pts[2, 0]), int(rotated_pts[2, 1]), int(rotated_pts[3, 0]), int(rotated_pts[3, 1])),
                      fill=(0, 255, 0))
            draw.line((int(rotated_pts[3, 0]), int(rotated_pts[3, 1]), int(rotated_pts[0, 0]), int(rotated_pts[0, 1])),
                      fill=(0, 255, 0))

        elif mode == 0:
            draw.line((int(rotated_pts[0, 0]), int(rotated_pts[0, 1]), int(rotated_pts[1, 0]), int(rotated_pts[1, 1])),
                      fill=(0, 0, 255))
            draw.line((int(rotated_pts[1, 0]), int(rotated_pts[1, 1]), int(rotated_pts[2, 0]), int(rotated_pts[2, 1])),
                      fill=(0, 0, 255))
            draw.line((int(rotated_pts[2, 0]), int(rotated_pts[2, 1]), int(rotated_pts[3, 0]), int(rotated_pts[3, 1])),
                      fill=(0, 0, 255))
            draw.line((int(rotated_pts[3, 0]), int(rotated_pts[3, 1]), int(rotated_pts[0, 0]), int(rotated_pts[0, 1])),
                      fill=(0, 0, 255))

        elif mode == 2:
            draw.line((int(rotated_pts[0, 0]), int(rotated_pts[0, 1]), int(rotated_pts[1, 0]), int(rotated_pts[1, 1])),
                      fill=(255, 255, 0))
            draw.line((int(rotated_pts[1, 0]), int(rotated_pts[1, 1]), int(rotated_pts[2, 0]), int(rotated_pts[2, 1])),
                      fill=(255, 255, 0))
            draw.line((int(rotated_pts[2, 0]), int(rotated_pts[2, 1]), int(rotated_pts[3, 0]), int(rotated_pts[3, 1])),
                      fill=(255, 255, 0))
            draw.line((int(rotated_pts[3, 0]), int(rotated_pts[3, 1]), int(rotated_pts[0, 0]), int(rotated_pts[0, 1])),
                      fill=(255, 255, 0))

        if not cls_prob is None:
            score = cls_prob[idx]
            if idx == 0:
                draw.text((int(rotated_pts[0, 0]+30), int(rotated_pts[0, 1]+30)), str(idx), fill=(255, 255, 255, 128), font=font)

    # cv2.imshow("image", cv2.resize(img, (1024, 768)))
    # cv2.wait Key(0)
    del draw
    return img

def inference(det_net, file_paths, des_txt_folder, des_img_folder='', threshold={}):
    dota_demo = RRPNDemo(
        # cfg,
        det_net,
        min_image_size=800,
        confidence_threshold=threshold,#0.30,#
    )
    vis = False #True #
    file_paths.sort()
    # num_images = len(file_paths)
    # print(len(file_paths))
    # cnt = 0
    for file_path in tqdm(file_paths):#cnt, enumerate
        # cnt += 1
        # print('name', file_path)
        # if cnt<= 16620:
        #     continue
        # impath = os.path.join(image_dir, image)
        # file_path = 'data/DOTA/data_800/val/images/P0007__1__0___1337.png'
        # print(file_path)
        image = cv2.imread(file_path)
        # print('img2:', image.shape)
        # tic = time.time()
        predictions, bounding_boxes, scores, labels = dota_demo.run_on_opencv_image(image)
        # toc = time.time()

        # print('time cost:', str(toc - tic)[:6], '|', str(cnt) + '/' + str(num_images))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes_np = bounding_boxes.bbox.data.cpu().numpy()
        bboxes_np[:, 2:4] /= cfg.RRPN_RCNN.GT_BOX_MARGIN

        width, height = bounding_boxes.size

        if vis:
            pil_image = vis_image(Image.fromarray(image), bboxes_np)
            img_file_path = os.path.join(des_img_folder, r'demo_{}.png'.format(file_path.split('/')[-1][:-4]))
            # print(img_file_path)
            pil_image.save(img_file_path)
            # cv2.imwrite(img_file_path, pil_image)
        write_result_ICDAR_RRPN2polys(file_path[:-4], bboxes_np, threshold=0.7, result_dir=des_txt_folder, height=height,
                                      width=width, scores=scores, labels=labels)
        # if cnt>=20:
        #     break
