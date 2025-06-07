import numpy as np

from frankateach.constants import CAM_FPS
from frankateach.utils import FrequencyTimer, notify_component_start
from frankateach.network import ZMQCameraPublisher

import cv2
import time


class FishEyeCamera:
    def __init__(
        self,
        host,
        port,
        cam_id,
        cam_config,
    ):
        # Disabling scientific notations
        np.set_printoptions(suppress=True)
        self.cam_id = cam_id
        self.cam_config = cam_config
        self._cam_serial_num = cam_config.cam_serial_num

        # Different publishers to avoid overload
        self.rgb_publisher = ZMQCameraPublisher(host, port)
        self.timer = FrequencyTimer(CAM_FPS)  # 30 fps

        # Starting the Fisheye pipeline
        self._start_fisheye()

    def _start_fisheye(self):
        self.cap = cv2.VideoCapture(self._cam_serial_num)
        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Check if the camera is opened successfully, wait until it is
        while not self.cap.isOpened():
            self.cap.isOpened()

    def get_rgb_depth_images(self):
        frame = None
        while frame is None:
            _, frame = self.cap.read()
        timestamp = time.time()
        return frame, timestamp

    @staticmethod
    def rescale_image(image, rescale_factor):
        width, height = (
            int(image.shape[1] / rescale_factor),
            int(image.shape[0] / rescale_factor),
        )
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    def stream(self):
        notify_component_start("fisheye camera")
        print(f"Started the pipeline for FishEye camera: {self.cam_id}!")

        try:
            while True:
                self.timer.start_loop()
                color_image, timestamp = self.get_rgb_depth_images()

                # Publishing the rgb images
                self.rgb_publisher.pub_rgb_image(color_image, timestamp)

                self.timer.end_loop()
                if cv2.waitKey(1) == ord("q"):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.cap.release()
            print("Shutting down pipeline for camera {}.".format(self.cam_id))
