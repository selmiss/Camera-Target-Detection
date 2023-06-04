import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.torch_utils import select_device, TracedModel
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import cv2
OBJ_LIST = [ 'car', 'bus', 'truck']
DETECTOR_PATH = './weights/yolov7.pt'


def plot_bboxes(image, bboxes, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.003 * (image.shape[0] + image.shape[1]) / 2)  # line/font thickness
    list_pts = []
    point_radius = 4

    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        color = (0, 255, 0)
        # if y2 < 240:
        #     continue
        c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        cv2.rectangle(image, c1, c2, color, 1)  # filled
        # cv2.putText(image, 'ID-{}'.format(pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
        #             [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return image


class baseDet(object):
    def __init__(self):
        self.img_size = 640
        self.threshold = 0.3
        self.stride = 1

    def build_config(self):
        self.frameCounter = 0

    def feedCap(self, im):
        retDict = {
            'frame': None,
            'list_of_ids': None,
            'obj_bboxes': []
        }
        self.frameCounter += 1
        with torch.no_grad():
            im, obj_bboxes = self.detect(im)
        retDict['frame'] = im
        retDict['obj_bboxes'] = obj_bboxes

        return retDict

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def detect(self):
        raise EOFError("Undefined model type.")


class Detector(baseDet):
    def __init__(self):
        super(Detector, self).__init__()
        self.init_model()
        self.build_config()

    def init_model(self):
        self.weights = DETECTOR_PATH
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        if self.device.type != 'cpu':
            model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(model.parameters())))
        # model.half()
        model = TracedModel(model, self.device, 640)
        self.m = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names

    def preprocess(self, img):
        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        # img = img.astype(np.float32)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 图像归一化
        # img = img.half()  # 半精度
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img0, img

    def detect(self, im):
        im0, img = self.preprocess(im)
        pred = self.m(img, augment=True)[0]
        # pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.45, classes=[2, 5, 7], agnostic=True)
        pred_boxes = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))
            im = plot_bboxes(im, pred_boxes)
        return im, pred_boxes

