import zmq
import base64
import numpy as np
import pickle
import cv2

class VideoStreamer(object):
    def __init__(self, host, cam_port):
        self._init_socket(host, cam_port)

    def _init_socket(self, host, port):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect('tcp://{}:{}'.format(host, port))
        self.socket.setsockopt(zmq.SUBSCRIBE, b"rgb_image")

    def _get_image(self):
        raw_data = self.socket.recv()
        data = raw_data.lstrip(b"rgb_image ")
        data = pickle.loads(data)
        encoded_data = np.fromstring(base64.b64decode(data['rgb_image']), np.uint8)
        return encoded_data.tobytes()

    def yield_frames(self):
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + self._get_image() + b'\r\n')  # concat frame one by one and show result


class MonitoringApplication(object):
    def __init__(self, configs):
        # Loading the network configurations
        self.host_address = configs.host_address
        self.port_offset = configs.cam_port_offset
        self.num_cams = configs.num_cams

        # Initializing the streamers        
        self._init_cam_streamers()

    def _init_cam_streamers(self):
        self.cam_streamers = []
        for idx in range(self.num_cams):
            self.cam_streamers.append(
                VideoStreamer(
                    host = self.host_address,
                    cam_port = self.port_offset + idx + 1
                )
            )

    def get_cam_streamer(self, id):
        return self.cam_streamers[id - 1]