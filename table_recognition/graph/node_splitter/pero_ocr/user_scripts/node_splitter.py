import cv2
import json
import numpy as np
from scipy.special import log_softmax

from table_recognition.graph.node_splitter.pero_ocr.pero_ocr.document_ocr.layout import PageLayout
from table_recognition.graph.node_splitter.pero_ocr.pero_ocr.ocr_engine.pytorch_ocr_engine import PytorchEngineLineOCR
from table_recognition.graph.node_splitter.pero_ocr.pero_ocr.force_alignment import align_text
from table_recognition.graph.node_splitter.pero_ocr.pero_ocr.document_ocr.crop_engine import EngineLineCropper


class LabelEncoder:
    def __init__(self, char_to_idx):
        self.char_to_idx = char_to_idx

    def encode(self, label: str):
        return np.array([self.char_to_idx[c] for c in label])


class NodeSplitter(object):
    def __init__(self, img_path, ocr_output_path):
        self.img_path = img_path
        self.ocr_output_path = ocr_output_path

        self.out_nodes = []

    def split_node(self):
        image = cv2.imread(self.img_path)
        page_layout = PageLayout(self.ocr_output_path)

        # TODO - remove absolute path
        with open("/home/lpiwowar/master-thesis/weights/kurrent/ocr_engine.json", "r") as f:
            ocr_config = json.load(f)
        line_cropper = EngineLineCropper(line_height=ocr_config["line_px_height"], poly=2,
                                         scale=ocr_config["line_vertical_scale"])
        
        # TODO - remove absolute path
        ocr_engine = PytorchEngineLineOCR("/home/lpiwowar/master-thesis/weights/kurrent/ocr_engine.json")
        char_to_idx = {c: i for i, c in enumerate(ocr_config["characters"])}
        label_encoder = LabelEncoder(char_to_idx)

        for line in page_layout.lines_iterator():
            line_crop = self.line_cropper.crop(image, line.baseline, line.heights)
            line_batched = line_crop[np.newaxis]

            _, logits, _ = ocr_engine.process_lines(line_batched, sparse_logits=False, tight_crop_logits=True)
            logits = logits[0]
            log_probs = log_softmax(logits, axis=1)
            labels = label_encoder.encode(line.transcription)
            char_positions = align_text(-log_probs, labels, log_probs.shape[1] - 1)
            char_positions = char_positions * ocr_engine.net_subsampling

            if True:
                for i, p in enumerate(char_positions):
                    color = [0, 0, 255] if line.transcription[i] != " " else [0, 255, 0]
                    line_crop[:, p, :] = color
                cv2.imshow("Character positions", line_crop)
                cv2.waitKey(0)
