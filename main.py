from yolov5_deepsort.camera_processor import CameraProcessor

deploy = False


def main_app():
    cameras = []
    for camera in cameras:
        processor = CameraProcessor(camera)
        processor.process()


if __name__ == '__main__':
    main_app()
