import pyrealsense2 as rs

devices = rs.context().query_devices()
print(f"Found {len(devices)} RealSense device(s)")
for dev in devices:
    print(dev.get_info(rs.camera_info.name),
          "| serial", dev.get_info(rs.camera_info.serial_number),
          "| USB", dev.get_info(rs.camera_info.usb_type_descriptor))

if len(devices) > 0:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    try:
        frame = pipeline.wait_for_frames().get_color_frame()
        print(f"Got a {frame.get_width()}x{frame.get_height()} color frame")
    finally:
        pipeline.stop()
