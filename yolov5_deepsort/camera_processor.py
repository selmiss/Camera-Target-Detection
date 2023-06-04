import sys
sys.path.append('./')
import os.path as osp
from objdetector import Detector
import cv2
root_dir = osp.abspath(osp.dirname(__file__))
cases_dir = osp.join(root_dir, 'configs')


class CameraProcessor:
    def __init__(self, url=''):

        # Get a video stream from video address
        self.url = url
        self.vid_cap = cv2.VideoCapture(self.url)
        self.width = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.origin_fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
        print(self.url, self.origin_fps)
        # Object detection area
        self.volume_area = [0, 0, self.width, self.height]
        print(f'width: {self.width} height: {self.height} fps: {self.origin_fps}')

        # Detection status
        self.processing = False

        # Set the result of detection
        self.max_level = 0

    def process(self):
        frame_idx = 1
        det = Detector()
        t = int(1000 / self.origin_fps)
        # t = 2000

        volume_info = [0]
        while self.processing:
            ret_val, im0 = self.vid_cap.read()
            if not ret_val:
                break

            result = det.feedCap(im0)

            # Get object boxes and frame
            obj_bboxes = result['obj_bboxes']
            frame = result['frame']
            print('boxes:', len(obj_bboxes))
            # Gets the number of detection boxes per frame
            car_num = self.count_volume(obj_bboxes)
            volume_info.append(volume_info[frame_idx - 1] + car_num)
            print(frame_idx, car_num)
            # cv2.rectangle(frame, (0, int(self.height // 2.5)), (self.width, self.height), (0, 255, 0), 1)
            if frame_idx > 2 * self.origin_fps:
                average_volume = (volume_info[frame_idx] - volume_info[frame_idx - 2*self.origin_fps]) / (self.origin_fps * 2)
                print(average_volume)

            # frame = imutils.resize(frame, height=500)
            cv2.imshow("Cars detection", frame)
            cv2.waitKey(t)
            frame_idx += 1

            if frame_idx == (40 * self.origin_fps):
                self.stop_process()

    def start_process(self):
        if self.processing:
            print('[WARN] Camera  is already running')
            return
        self.processing = True
        print(f'[INFO] Camera  started')
        self.process()

    def stop_process(self):
        self.processing = False
        print('[INFO] Camera  ended')

    # Calculate IOU
    def IOU(self, box1, box2):
        """
        :box1:[x1,y1,x2,y2]# (x1,y1) means left-top，(x2,y2) means right-bottom
        :box2:[x1,y1,x2,y2]
        :return: iou_ratio Intersection ratio
        """
        width1 = abs(box1[2] - box1[0])
        height1 = abs(box1[1] - box1[3])
        width2 = abs(box2[2] - box2[0])
        height2 = abs(box2[1] - box2[3])
        xmax = max(box1[0], box1[2], box2[0], box2[2])
        ymax = max(box1[1], box1[3], box2[1], box2[3])
        xmin = min(box1[0], box1[2], box2[0], box2[2])
        ymin = min(box1[1], box1[3], box2[1], box2[3])
        W = xmin + width1 + width2 - xmax
        H = ymin + height1 + height2 - ymax
        '''
        There is no intersection when H and W are both less than
         or equal to 0, and there is intersection in other cases.
        '''
        if W <= 0 or H <= 0:
            iou_ratio = 0
        else:
            iou_area = W * H  # Calculate the area of the intersection
            box1_area = width1 * height1
            box2_area = width2 * height2
            iou_ratio = iou_area / (box1_area + box2_area - iou_area)  # Calculate the area of the union
        return iou_ratio

    # Count the number of vehicles in the area
    def count_volume(self, obj_bboxes):
        num = 0
        for box in obj_bboxes:
            x1, y1, x2, y2, lbl, conf = box
            bbox = [x1, y1, x2, y2]
            if self.IOU(bbox, self.volume_area) > 0:
                num += 1
        return num


if __name__ == '__main__':
    camera = CameraProcessor('./testdata/test_vedio1.mp4')
    camera.start_process()
