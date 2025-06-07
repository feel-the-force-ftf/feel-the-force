import pyrealsense2 as rs

def get_camera_intrinsics():
    # Create a context object to manage RealSense devices
    ctx = rs.context()
    
    # Get all connected devices
    for i, device in enumerate(ctx.query_devices()):
        print(f"\nDevice {i}: {device.get_info(rs.camera_info.serial_number)}")
        
        # Create a pipeline for this device
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(device.get_info(rs.camera_info.serial_number))
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start the pipeline
        pipeline_profile = pipeline.start(config)
        
        # Get the intrinsics
        color_stream = pipeline_profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        print(f"fx: {intrinsics.fx}")
        print(f"fy: {intrinsics.fy}")
        print(f"ppx (cx): {intrinsics.ppx}")
        print(f"ppy (cy): {intrinsics.ppy}")
        print(f"distortion coefficients: {intrinsics.coeffs}")
        
        # Stop the pipeline
        pipeline.stop()

if __name__ == "__main__":
    get_camera_intrinsics()