import pickle
import os
import lzma
import argparse
import logging
import pickle
import numpy as np

from typing import List, Union, Tuple
from abc import ABC, abstractmethod

from yolox.tracker.byte_tracker import BYTETracker

LOG_LEVEL = os.environ.get('PY_LOG_LEVEL', 'WARNING').upper()
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s | %(message)s",
    level=LOG_LEVEL,
    datefmt="%Y-%m-%d %H:%M:%S.fff")

class BBoxDetector(ABC):

    @abstractmethod
    def next(self) -> Union[np.ndarray, None]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class BBoxDetectionResult(BBoxDetector):

    data_: List[np.ndarray]
    cur_idx_: int

    def __init__(self, path: str) -> None:
        self.cur_idx_ = 0
        logging.info("loading pickle file at \"{fpath}\"".format(fpath=path))
        if path.lower().endswith(".lzma"):
            with lzma.open(path) as f:
                self.data_ = pickle.loads(f.read())
        elif path.lower().endswith(".pkl"):
            self.data_ = pickle.load(path)
        else:
            comps: List[str] = path.split(".")
            if len(comps) <= 1:
                raise ValueError(f"empty file extension")
            else:
                raise ValueError(f"unrecognized file extension \"{comps[-1]}\"")
        logging.info("done loading pickle file")

    def next(self) -> Union[np.ndarray, None]:
        if self.cur_idx_ >= len(self.data_):
            return None
        self.cur_idx_ += 1
        return self.data_[self.cur_idx_ - 1]

    def reset(self) -> None:
        self.cur_idx_ = 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("ByteTrack")
    parser.add_argument(
        "--detres", type=str, help="path to detection result in pickle or lzma format"
    )
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument(
        "--compress", action="store_true", default=True, help="whether or not to compress the result"
    )
    parser.add_argument(
        "--out", type=str, help="path to output file"
    )
    return parser

def save_data(path: str, content, should_compress: bool):
    data: bytes = pickle.dumps(content)
    base_names: List[str] = os.path.basename(path).split(".")
    if len(base_names) > 1:
        base_names = base_names[:-1]
    base_names.append("pkl")
    if should_compress:
        lzc = lzma.LZMACompressor()
        out1: bytes = lzc.compress(data)
        out2: bytes = lzc.flush()
        data = b"".join([out1, out2])
        base_names.append("lzma")
    out_basename = ".".join(base_names)
    res_path = os.path.join(os.path.dirname(path), out_basename)
    logging.info(f"saving tracking result to \"{res_path}\"")
    with open(res_path, "wb") as outfile:
        outfile.write(data)
    logging.info("done saving tracking result")

if __name__ == "__main__":
    parser: argparse.ArgumentParser = build_parser()
    args = parser.parse_args()

    tracker: BYTETracker = BYTETracker(args)

    frame_idx: int = 0
    detections: BBoxDetector = BBoxDetectionResult(args.detres)
    cur: Union[np.ndarray, None] = detections.next()
    res: List[np.ndarray] = list() # List[List[track_id,x,y,width,height]]

    while cur is not None:
        online_targets = tracker.update(cur)
        frame_res: np.ndarray = np.zeros((0, 5), dtype=np.float32)
        for t in online_targets:
            ltwh = t.tlwh
            frame_res = np.append(
                frame_res,
                [[t.track_id, ltwh[0], ltwh[1], ltwh[2], ltwh[3]]],
                axis=0
            )
        res.append(frame_res)
        cur = detections.next()
        frame_idx += 1
    
    save_data(args.out, res, args.compress)
