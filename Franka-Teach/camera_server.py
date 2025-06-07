from collections import defaultdict
from frankateach.camera_server import CameraServer
from frankateach.constants import HOST, CAM_PORT
import hydra
import argparse


@hydra.main(version_base="1.2", config_path="configs", config_name="camera")
def main(cfg):
    cam_configs = defaultdict(list)
    for camera in cfg.cam_info:
        cam_config = argparse.Namespace(**camera, **cfg.cam_config[camera.type])
        cam_configs[camera.type].append(cam_config)

    camera_server = CameraServer(
        host=HOST,
        cam_port=CAM_PORT,
        cam_configs=cam_configs,
    )

    camera_server._init_camera_threads()


if __name__ == "__main__":
    main()
