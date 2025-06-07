import numpy as np

from frankateach.network import ZMQCameraPublisher
from frankateach.utils import FrequencyTimer, notify_component_start
from frankateach.constants import CAM_FPS, DEPTH_PORT_OFFSET

import pyrealsense2 as rs
import time


class RealsenseCamera:
    def __init__(
        self,
        host,
        port,
        cam_id,
        cam_config,
    ):
        self.cam_id = cam_id
        self.cam_config = cam_config
        self._cam_serial_num = cam_config.cam_serial_num
        self._depth = cam_config.depth

        # Different publishers to avoid overload
        self.rgb_publisher = ZMQCameraPublisher(host, port)

        if self._depth:
            self.depth_publisher = ZMQCameraPublisher(
                host, port=port + DEPTH_PORT_OFFSET
            )

        self.timer = FrequencyTimer(CAM_FPS)

        # Starting the realsense pipeline
        self._start_realsense(self._cam_serial_num)

    def _start_realsense(self, cam_serial_num):
        config = rs.config()
        self.pipeline = rs.pipeline()
        config.enable_device(cam_serial_num)

        # Enabling camera streams
        config.enable_stream(
            rs.stream.color,
            self.cam_config.width,
            self.cam_config.height,
            rs.format.bgr8,
            self.cam_config.fps,
        )
        if self._depth:
            config.enable_stream(
                rs.stream.depth,
                self.cam_config.width,
                self.cam_config.height,
                rs.format.z16,
                self.cam_config.fps,
            )

        # Starting the pipeline
        cfg = self.pipeline.start(config)
        device = cfg.get_device()

        time.sleep(1)

        if self._depth:
            # Setting the depth mode to high accuracy mode
            depth_sensor = device.first_depth_sensor()
            depth_sensor.set_option(
                rs.option.visual_preset, self.cam_config.processing_preset
            )

        self.realsense = self.pipeline

        # Obtaining the color intrinsics matrix for aligning the color and depth images
        profile = self.pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intrinsics = color_profile.get_intrinsics()
        self.intrinsics_matrix = np.array(
            [
                [intrinsics.fx, 0, intrinsics.ppx],
                [0, intrinsics.fy, intrinsics.ppy],
                [0, 0, 1],
            ]
        )

        # Align function - aligns other frames with the color frame
        self.align = rs.align(rs.stream.color)

    def get_rgb_depth_images(self):
        frames = None

        while frames is None:
            # Obtaining and aligning the frames
            frames = self.realsense.wait_for_frames()
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            if self._depth:
                depth_frame = aligned_frames.get_depth_frame()
                depth_image = np.asanyarray(depth_frame.get_data())
            else:
                depth_image = None

        return color_image, depth_image, frames.get_timestamp()

    def stream(self):
        # Starting the realsense stream
        notify_component_start("realsense")
        print(f"Started the Realsense pipeline for camera: {self._cam_serial_num}!")

        try:
            while True:
                self.timer.start_loop()
                color_image, depth_image, timestamp = self.get_rgb_depth_images()

                # color_image = rotate_image(color_image, self.cam_configs.rotation_angle)
                # depth_image = rotate_image(depth_image, self.cam_configs.rotation_angle)

                self.rgb_publisher.pub_rgb_image(color_image, timestamp)
                if self._depth:
                    self.depth_publisher.pub_depth_image(depth_image, timestamp)

                self.timer.end_loop()
        except KeyboardInterrupt:
            pass
        finally:
            print("Shutting down realsense pipeline for camera {}.".format(self.cam_id))
            self.rgb_publisher.stop()
            if self._depth:
                self.depth_publisher.stop()
            self.pipeline.stop()
