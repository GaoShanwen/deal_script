import os
import json
from tqdm import tqdm
from pycocotools.coco import COCO
from txt2xml import parse_args, Txt2xml
from tkinter import _flatten


class Json2xml(Txt2xml):
    def __init__(self, cfg, **kwargs):
        cfg.set_cats = []
        super().__init__(cfg, anno_type="coco")

    def convert2xml(self):
        filelist = [filename for filename in os.listdir(self.src_root) if filename.endswith(".json")]
        for file_name in tqdm(filelist):
            base_name = os.path.splitext(file_name)[0]
            with open(os.path.join(self.src_root, file_name)) as f:
                data = json.load(f)  # 解析 JSON 文件
            if data["version"] == "5.6.1":
                annos = [[d["label"]]+list(_flatten(d["points"])) for d in data["shapes"] if d["label"] not in ["Person"]]
            else:
                annos = [[d["label"]]+list(_flatten(d["points"][::2])) for d in data["shapes"] if d["label"] not in ["Person"]]
            self.writer.run_write(base_name, annos)


if __name__ == '__main__':
    args = parse_args()
    Json2xml(args)
