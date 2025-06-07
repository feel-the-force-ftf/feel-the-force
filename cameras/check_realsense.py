import pyrealsense2 as rs
import numpy as np
import cv2

def check_realsense_cameras():
    # First, get list of all connected RealSense devices
    ctx = rs.context()
    devices = ctx.query_devices()
    
    print(f"\nFound {len(devices)} RealSense devices:\n")
    
    for i, dev in enumerate(devices):
        print(f"Device {i+1}:")
        print(f"    Serial Number: {dev.get_info(rs.camera_info.serial_number)}")
        print(f"    Name: {dev.get_info(rs.camera_info.name)}")
        print(f"    Firmware Version: {dev.get_info(rs.camera_info.firmware_version)}")
        print()

    # Try to start each camera
    for i, dev in enumerate(devices):
        try:
            # Configure camera stream
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(dev.get_info(rs.camera_info.serial_number))
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            # Start streaming
            print(f"Testing Device {i+1}...")
            pipeline.start(config)
            
            # Get a few frames
            for _ in range(5):
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    print(f"    Failed to get frames from Device {i+1}")
                    continue
                
                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Show images
                cv2.imshow(f'Color Camera {i+1}', color_image)
                cv2.imshow(f'Depth Camera {i+1}', depth_image)
                cv2.waitKey(1)
            
            print(f"    Device {i+1} working correctly")
            pipeline.stop()
            
        except Exception as e:
            print(f"Error with Device {i+1}: {str(e)}")
        
        print()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    check_realsense_cameras()