import os
import xml.dom.minidom

from tqdm import tqdm
import cv2
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate features or SQL queries")
    parser.add_argument("--obj-root", type=str, default="")
    parser.add_argument("--src-root", type=str, default="")
    parser.add_argument("--set-cats", type=str, nargs='*', default=None)
    return parser.parse_args()


class XMLWriter(object):
    def __init__(self, cats: list, src_root: str, obj_root: str, anno_type: str='yolo'):
        self.src_root = src_root
        self.obj_root = obj_root
        self.cats = cats
        self.doc = None
        self.data = {}
        self.anno_type = anno_type

    def create_element_for_xml(self, obj_name, obj_value, node):
        node_element = self.doc.createElement(obj_name)
        node_element.appendChild(self.doc.createTextNode(obj_value))
        node.appendChild(node_element)


    def run_write(self, base_name: str, annos: list):
        """
        Converts annotations(YOLO format) to VOC format(Detect Task Type)
        """
        self.doc = xml.dom.minidom.Document()
        root = self.doc.createElement('annotation')
        self.doc.appendChild(root)

        filename = base_name + ".jpg"
        path = os.path.join(self.src_root, filename)
        filename = filename if os.path.exists(path) else base_name + ".png"
        path = os.path.join(self.src_root, filename)

        self.create_element_for_xml("folder", self.src_root, root)
        self.create_element_for_xml("filename", filename, root)
        self.create_element_for_xml("path", path, root)

        sourcename = self.doc.createElement('source')
        self.create_element_for_xml("database", "Unknown", sourcename)
        root.appendChild(sourcename)

        height, width, channel = cv2.imread(path).shape
        nodesize = self.doc.createElement('size')

        self.create_element_for_xml("width", str(width), nodesize)
        self.create_element_for_xml("height", str(height), nodesize)
        self.create_element_for_xml("depth", str(channel), nodesize)
        root.appendChild(nodesize)

        self.create_element_for_xml("segmented", "0", root)
        self.create_element_for_xml("shape_type", "POLYGON", root)
        
        for anno in annos:
            nodeobject = self.doc.createElement('object')
            if self.anno_type == "yolo":
                cat_id, x, y, w, h = map(eval, anno)
                x, y, w, h = width * x, height * y, height * w, height * h
                x1, y1, x2, y2 = list(map(str, map(round, [x-w/2, y-h/2, x+w/2, y+h/2])))
                cat = self.cats[cat_id]
            elif self.anno_type == "coco":
                cat, x1, y1, x2, y2 = anno
            else:
                raise ValueError("Unsupported annotation type.")
            if cat not in self.data:
                self.data.update({cat: 0})
            self.data[cat] += 1
            self.create_element_for_xml("name", cat, nodeobject)
            self.create_element_for_xml("pose", "Unspecified", nodeobject)
            self.create_element_for_xml("truncated", "0", nodeobject)
            self.create_element_for_xml("difficult", "0", nodeobject)

            nodebbox = self.doc.createElement('bndbox')
            self.create_element_for_xml("xmin", str(x1), nodebbox)
            self.create_element_for_xml("ymin", str(y1), nodebbox)
            self.create_element_for_xml("xmax", str(x2), nodebbox)
            self.create_element_for_xml("ymax", str(y2), nodebbox)
            nodeobject.appendChild(nodebbox)
            root.appendChild(nodeobject)

        fp = open(os.path.join(self.obj_root, base_name + '.xml'), 'w', encoding='utf-8')
        self.doc.writexml(fp, indent='  ', newl='\n', addindent='  ')
        fp.close()


def analyze_cats(cats):
    if len(cats) != 1:
        return cats
    cats = cats[0]
    if not (cats.endswith(".txt") or cats.endswith(".names")):
        raise ValueError("Cats should be a TXT file or a list of names.")
    with open(cats, "r") as f:
        return [cat.strip() for cat in f.readlines()]


class Txt2xml(object):
    def __init__(self, cfg, **kwargs):
        self.src_root = cfg.src_root
        self.obj_root = cfg.obj_root or cfg.src_root
        anno_type = kwargs.get("anno_type", "yolo")
        self.cats = analyze_cats(cfg.set_cats)
        print(self.cats)
        
        self.writer = XMLWriter(self.cats, self.src_root, self.obj_root, anno_type)
        self.convert2xml()
        print(f"Convert complete, Every object number: {self.writer.data}")
        try:
            print(f"Convert complete, Total object number: {sum(self.writer.data)}")
        except Exception as e:
            print(f"Convert complete, Total object number: {sum(self.writer.data.values())}")

    def convert2xml(self):
        filelist = [filename for filename in os.listdir(self.src_root) if filename.endswith(".txt")]
        for file_name in tqdm(filelist):
            base_name = os.path.splitext(file_name)[0]
            with open(os.path.join(self.src_root, base_name + ".txt")) as f:
                annos = [line.strip().split(' ') for line in f.readlines()]
            self.writer.run_write(base_name, annos)


if __name__ == '__main__':
    args = parse_args()
    Txt2xml(args)