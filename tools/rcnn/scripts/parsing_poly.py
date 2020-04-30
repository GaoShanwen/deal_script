import cv2
import os
import copy
import numpy as np

import torch

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class ParsingPoly(object):
    """
    This class handles parsing for all objects in the image
    """

    def __init__(self, parsing, size, mode=None):
        # if isinstance(parsing, torch.Tensor):
        #     # The raw data representation is passed as argument
        #     parsing = parsing.clone()
        # elif isinstance(parsing, (list, tuple)):
        #     parsing = torch.as_tensor(parsing)

        # if len(parsing.shape) == 2:
        #     # if only a single instance mask is passed
        #     parsing = parsing[None]
        #
        # assert len(parsing.shape) == 3
        # assert parsing.shape[1] == size[1], "%s != %s" % (parsing.shape[1], size[1])
        # assert parsing.shape[2] == size[0], "%s != %s" % (parsing.shape[2], size[0])

        self.parsing = parsing
        self.size = size
        self.mode = mode
        self.extra_fields = {}

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError("Only FLIP_LEFT_RIGHT implemented")

        dim = 1 if method == FLIP_TOP_BOTTOM else 2
        flipped_parsing = self.parsing.flip(dim)

        flipped_parsing = flipped_parsing.numpy()
        for l_r in FLIP_MAP:
            left = np.where(flipped_parsing == l_r[0])
            right = np.where(flipped_parsing == l_r[1])
            flipped_parsing[left] = l_r[1]
            flipped_parsing[right] = l_r[0]
        flipped_parsing = torch.from_numpy(flipped_parsing)

        return Parsing(flipped_parsing, self.size)

    # def move(self, gap):
    #     c, h, w = self.parsing.shape
    #     old_up, old_left, old_bottom, old_right = max(gap[1], 0), max(gap[0], 0), h, w
    #
    #     new_up, new_left = max(0 - gap[1], 0), max(0 - gap[0], 0)
    #     new_bottom, new_right = h + new_up - old_up, w + new_left - old_left
    #     new_shape = (c, h + new_up, w + new_left)
    #
    #     moved_parsing = torch.zeros(new_shape, dtype=torch.uint8)
    #     moved_parsing[:, new_up:new_bottom, new_left:new_right] =
    # self.parsing[:, old_up:old_bottom, old_left:old_right]
    #
    #     moved_size = new_shape[2], new_shape[1]
    #     return Parsing(moved_parsing, moved_size)

    def move(self, gap):
        assert isinstance(gap, (list, tuple, torch.Tensor)), str(type(gap))

        # gap is assumed to be xy.
        current_width, current_height = self.size
        gap_x, gap_y = map(float, gap)

        w, h = current_width - gap_x, current_height - gap_y

        # moved_parsings = []
        # for poly in self.parsings:
        #     p = poly.clone()
        #     p[0::2] = p[0::2] - gap_x
        #     p[1::2] = p[1::2] - gap_y
        #     moved_parsings.append(p)

        moved_parsings = []
        for parsing in self.parsing:
            cropped_parts = []
            for part in parsing:
                cropped_part = []
                if len(part):
                    for poly in part:
                        p = np.array(poly)
                        p[0::2] = p[0::2] - gap_x  # .clamp(min=0, max=w)
                        p[1::2] = p[1::2] - gap_y  # .clamp(min=0, max=h)
                        p = p.tolist()
                        cropped_part.append(p)
                cropped_parts.append(cropped_part)
            moved_parsings.append(cropped_parts)

        return ParsingPoly(moved_parsings, size=(w, h))

    # def crop(self, box):
    #     assert isinstance(box, (list, tuple, torch.Tensor)), str(type(box))
    #     # box is assumed to be xyxy
    #     current_width, current_height = self.size
    #     xmin, ymin, xmax, ymax = [round(float(b)) for b in box]
    #
    #     assert xmin <= xmax and ymin <= ymax, str(box)
    #     xmin = min(max(xmin, 0), current_width - 1)
    #     ymin = min(max(ymin, 0), current_height - 1)
    #
    #     xmax = min(max(xmax, 0), current_width)
    #     ymax = min(max(ymax, 0), current_height)
    #
    #     xmax = max(xmax, xmin + 1)
    #     ymax = max(ymax, ymin + 1)
    #
    #     width, height = xmax - xmin, ymax - ymin
    #     cropped_parsing = self.parsing[:, ymin:ymax, xmin:xmax]
    #     cropped_size = width, height
    #     return Parsing(cropped_parsing, cropped_size)

    def crop(self, box):
        assert isinstance(box, (list, tuple, torch.Tensor)), str(type(box))

        # box is assumed to be xyxy
        current_width, current_height = self.size
        xmin, ymin, xmax, ymax = map(float, box)

        # assert xmin <= xmax and ymin <= ymax, str(box)
        xmin = min(max(xmin, 0), current_width - 1)
        ymin = min(max(ymin, 0), current_height - 1)

        xmax = min(max(xmax, 0), current_width)
        ymax = min(max(ymax, 0), current_height)

        xmax = max(xmax, xmin + 1)
        ymax = max(ymax, ymin + 1)

        w, h = xmax - xmin, ymax - ymin

        # cropped_parsings = []
        # for poly in self.parsing:
        #     p = poly.clone()
        #     p[0::2] = p[0::2] - xmin  # .clamp(min=0, max=w)
        #     p[1::2] = p[1::2] - ymin  # .clamp(min=0, max=h)
        #     cropped_parsings.append(p)

        cropped_parsings = []
        for parsing in self.parsing:
            cropped_parts = []
            for part in parsing:
                cropped_part = []
                if len(part):
                    for poly in part:
                        # p = poly.clone()
                        p = np.array(poly)
                        p[0::2] = p[0::2] - xmin  # .clamp(min=0, max=w)
                        p[1::2] = p[1::2] - ymin  # .clamp(min=0, max=h)
                        p = p.tolist()
                        cropped_part.append(p)
                cropped_parts.append(cropped_part)
            cropped_parsings.append(cropped_parts)

        return ParsingPoly(cropped_parsings, size=(w, h))

    # def set_size(self, size):
    #     c, h, w = self.parsing.shape
    #     new_shape = (c, size[1], size[0])
    #
    #     new_parsing = torch.zeros(new_shape, dtype=torch.uint8)
    #     new_parsing[:, :min(h, size[1]), :min(w, size[0])] = self.parsing[:, :min(h, size[1]), :min(w, size[0])]
    #
    #     self.parsing = new_parsing
    #     return Parsing(self.parsing, size)

    def set_size(self, size):
        return ParsingPoly(self.parsing, size=size)

    def resize(self, size):
        try:
            iter(size)
        except TypeError:
            assert isinstance(size, (int, float))
            size = size, size
        width, height = map(int, size)

        assert width > 0
        assert height > 0

        # Height comes first here!
        resized_parsing = torch.nn.functional.interpolate(
            self.parsing[None].float(),
            size=(height, width),
            mode="nearest",
        )[0].type_as(self.parsing)

        resized_size = width, height
        return Parsing(resized_parsing, resized_size)

    def to(self, *args, **kwargs):
        return self

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def __len__(self):
        return len(self.parsing)

    def __getitem__(self, index):
        selected_parsings = []
        for i in index:
            selected_parsings.append(self.parsing[i])
        # print(type(self.parsing), len(self.parsing), index)
        # parsing = torch.as_tensor(self.parsing)[index]#.clone()
        # parsing = self.parsing[index]#.clone()
        return ParsingPoly(selected_parsings, self.size)

    def __iter__(self):
        self.iter_idx = 0
        return self

    def __next__(self):
        if self.iter_idx < self.__len__():
            next_parsing = self.__getitem__(self.iter_idx)
            self.iter_idx += 1
            return next_parsing
        raise StopIteration()

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_parsing={}, ".format(len(self.parsing))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        return s



