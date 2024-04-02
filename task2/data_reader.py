from pathlib import Path
import cv2
import numpy as np
import lmdb


GB = 2**30
class LMDBReader:
    def __init__(self, path: Path):
        self.path = path
        self.env = None
        self.namelist_ = []
    
    def open(self):
        self.env = lmdb.open(self.path, 
                             map_size=GB * 16,
                             lock=False, 
                             subdir=False, 
                             readonly=True)
        self.namelist_ = []
        with self.env.begin(buffers=True) as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                key = bytes(key).decode('utf-8')
                self.namelist_.append(key)
    
    def namelist(self):
        return self.namelist_
    
    def decode_image(self, name):
        key = name.encode('utf-8')
        with self.env.begin() as txn:
            sample = txn.get(key)
        buf = np.frombuffer(sample, dtype='uint8')
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    
    def close(self):
        self.env.close()
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", 
             "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


def parse_gtruth(raw: str) -> str:
    # 0_0_22_27_27_33_16
    ids = [int(part) for part in raw.split('_')]
    province = provinces[ids[0]]
    first = alphabets[ids[1]]
    rest = [ads[x] for x in ids[2:]]
    return ''.join([province, first] + rest)


def parse_coords(raw: str) -> str:
    # 580&528_404&533_404&465_580&460
    parts = raw.split('_')
    coords = [tuple(map(int, part.split('&'))) for part in parts]
    return coords


def parse_meta(path: str):
    if path[-3:] == 'jpg':
        path = path[:-4]
    parts = path.split('-')
    idx = parts[0]
    coords = parse_coords(parts[3])
    gtruth = parse_gtruth(parts[4])
    return idx, coords, gtruth


class CCPDHelper:
    def __init__(self, reader: LMDBReader, namelist=None):
        self.reader = reader
        self.namelist = namelist if namelist is not None else reader.namelist()

    def item(self, idx: int):
        """
        returns (img, gtruth)
          # img: [H, W, C], in RGB space
          # gtruth: str
        """
        assert idx >= 0 and idx < self.size(), "Bad index"
        name = self.namelist[idx]
        meta = name.split('/')[1]
        _, _, gtruth = parse_meta(meta)
        img = self.reader.decode_image(name)
        return img, gtruth

    def size(self) -> int:
        return len(self.namelist)
