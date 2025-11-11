import os, sys
this_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(this_file), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import time
from multiprocessing import shared_memory
from televuer import TeleVuerWrapper
import logging_mp
import rclpy
from std_msgs.msg import Float32, Float32MultiArray, Float64MultiArray, Bool
logger_mp = logging_mp.get_logger(__name__, level=logging_mp.INFO)


def run_test_tv_wrapper():
    ros_node = None
    image_shape = (480, 640 * 2, 3)
    image_shm = shared_memory.SharedMemory(create=True, size=np.prod(image_shape) * np.uint8().itemsize)
    image_array = np.ndarray(image_shape, dtype=np.uint8, buffer=image_shm.buf)

    # from image_server.image_client import ImageClient
    # import threading
    # image_client = ImageClient(tv_img_shape = image_shape, tv_img_shm_name = image_shm.name, image_show=True, server_address="127.0.0.1")
    # image_receive_thread = threading.Thread(target = image_client.receive_process, daemon = True)
    # image_receive_thread.daemon = True
    # image_receive_thread.start()
    
    use_hand_track=False
    tv_wrapper = TeleVuerWrapper(binocular=True, use_hand_tracking=use_hand_track, img_shape=image_shape, img_shm_name=image_shm.name, 
                                   return_state_data=True, return_hand_rot_data = True)
    # ROS2 init & publishers
    rclpy.init(args=None)
    ros_node = rclpy.create_node('xr_tele_state_pub')
    pub_head = ros_node.create_publisher(Float64MultiArray, 'xr/head_pose_mat', 10)
    pub_left = ros_node.create_publisher(Float64MultiArray, 'xr/left_arm_pose_mat', 10)
    pub_right = ros_node.create_publisher(Float64MultiArray, 'xr/right_arm_pose_mat', 10)
    if use_hand_track:
        pub_lh_pos = ros_node.create_publisher(Float32MultiArray, 'xr/left_hand_pos', 10)
        pub_rh_pos = ros_node.create_publisher(Float32MultiArray, 'xr/right_hand_pos', 10)
        pub_lh_rot = ros_node.create_publisher(Float32MultiArray, 'xr/left_hand_rot', 10)
        pub_rh_rot = ros_node.create_publisher(Float32MultiArray, 'xr/right_hand_rot', 10)
        pub_l_pinch = ros_node.create_publisher(Float32, 'xr/left_pinch_value', 10)
        pub_r_pinch = ros_node.create_publisher(Float32, 'xr/right_pinch_value', 10)
        pub_l_pinch_state = ros_node.create_publisher(Bool, 'xr/left_pinch_state', 10)
        pub_r_pinch_state = ros_node.create_publisher(Bool, 'xr/right_pinch_state', 10)
        pub_l_sq_state = ros_node.create_publisher(Bool, 'xr/left_squeeze_state', 10)
        pub_r_sq_state = ros_node.create_publisher(Bool, 'xr/right_squeeze_state', 10)
        pub_l_sq_value = ros_node.create_publisher(Float32, 'xr/left_squeeze_value', 10)
        pub_r_sq_value = ros_node.create_publisher(Float32, 'xr/right_squeeze_value', 10)
    else:
        pub_l_trig = ros_node.create_publisher(Float32, 'xr/left_trigger_value', 10)
        pub_r_trig = ros_node.create_publisher(Float32, 'xr/right_trigger_value', 10)
        pub_l_sq = ros_node.create_publisher(Float32, 'xr/left_squeeze_ctrl_value', 10)
        pub_r_sq = ros_node.create_publisher(Float32, 'xr/right_squeeze_ctrl_value', 10)
        pub_l_thumb = ros_node.create_publisher(Float32MultiArray, 'xr/left_thumbstick_value', 10)
        pub_r_thumb = ros_node.create_publisher(Float32MultiArray, 'xr/right_thumbstick_value', 10)
        pub_l_a = ros_node.create_publisher(Bool, 'xr/left_aButton', 10)
        pub_l_b = ros_node.create_publisher(Bool, 'xr/left_bButton', 10)
        pub_r_a = ros_node.create_publisher(Bool, 'xr/right_aButton', 10)
        pub_r_b = ros_node.create_publisher(Bool, 'xr/right_bButton', 10)
    try:
        input("Press Enter to start tv_wrapper test...")
        running = True
        while running:
            start_time = time.time()
            teleData = tv_wrapper.get_motion_state_data()

            logger_mp.info("=== TeleData Snapshot ===")
            logger_mp.info(f"[Head Rotation Matrix]:\n{teleData.head_pose}")
            logger_mp.info(f"[Left EE Pose]:\n{teleData.left_arm_pose}")
            logger_mp.info(f"[Right EE Pose]:\n{teleData.right_arm_pose}")
            # publish core pose matrices (row-major flatten)
            head_msg = Float64MultiArray()
            head_msg.data = teleData.head_pose.astype(np.float64).reshape(-1).tolist()
            pub_head.publish(head_msg)
            left_msg = Float64MultiArray()
            left_msg.data = teleData.left_arm_pose.astype(np.float64).reshape(-1).tolist()
            pub_left.publish(left_msg)
            right_msg = Float64MultiArray()
            right_msg.data = teleData.right_arm_pose.astype(np.float64).reshape(-1).tolist()
            pub_right.publish(right_msg)

            if use_hand_track: # hand
                logger_mp.info(f"[Left Hand Position] shape {teleData.left_hand_pos.shape}:\n{teleData.left_hand_pos}")
                logger_mp.info(f"[Right Hand Position] shape {teleData.right_hand_pos.shape}:\n{teleData.right_hand_pos}")
                # publish hand data
                pos_l_msg = Float32MultiArray()
                pos_l_msg.data = teleData.left_hand_pos.astype(np.float32).reshape(-1).tolist()
                pub_lh_pos.publish(pos_l_msg)
                pos_r_msg = Float32MultiArray()
                pos_r_msg.data = teleData.right_hand_pos.astype(np.float32).reshape(-1).tolist()
                pub_rh_pos.publish(pos_r_msg)
                
                if teleData.left_hand_rot is not None:
                    logger_mp.info(f"[Left Hand Rotation] shape {teleData.left_hand_rot.shape}:\n{teleData.left_hand_rot}")
                    rot_l_msg = Float32MultiArray()
                    rot_l_msg.data = teleData.left_hand_rot.astype(np.float32).reshape(-1).tolist()  # 25*9
                    pub_lh_rot.publish(rot_l_msg)
                if teleData.right_hand_rot is not None:
                    logger_mp.info(f"[Right Hand Rotation] shape {teleData.right_hand_rot.shape}:\n{teleData.right_hand_rot}")
                    rot_r_msg = Float32MultiArray()
                    rot_r_msg.data = teleData.right_hand_rot.astype(np.float32).reshape(-1).tolist()  # 25*9
                    pub_rh_rot.publish(rot_r_msg)
                
                if teleData.left_pinch_value is not None:
                    logger_mp.info(f"[Left Pinch Value]: {teleData.left_pinch_value:.2f}")
                    m = Float32(); m.data = float(teleData.left_pinch_value); pub_l_pinch.publish(m)
                if teleData.right_pinch_value is not None:
                    logger_mp.info(f"[Right Pinch Value]: {teleData.right_pinch_value:.2f}")
                    m = Float32(); m.data = float(teleData.right_pinch_value); pub_r_pinch.publish(m)
                
                if teleData.tele_state:
                    state = teleData.tele_state
                    logger_mp.info("[Hand State]:")
                    logger_mp.info(f"  Left Pinch state: {state.left_pinch_state}")
                    logger_mp.info(f"  Left Squeeze: {state.left_squeeze_state} ({state.left_squeeze_value:.2f})")
                    logger_mp.info(f"  Right Pinch state: {state.right_pinch_state}")
                    logger_mp.info(f"  Right Squeeze: {state.right_squeeze_state} ({state.right_squeeze_value:.2f})")
                    b = Bool(); b.data = bool(state.left_pinch_state); pub_l_pinch_state.publish(b)
                    b = Bool(); b.data = bool(state.right_pinch_state); pub_r_pinch_state.publish(b)
                    b = Bool(); b.data = bool(state.left_squeeze_state); pub_l_sq_state.publish(b)
                    b = Bool(); b.data = bool(state.right_squeeze_state); pub_r_sq_state.publish(b)
                    m = Float32(); m.data = float(state.left_squeeze_value); pub_l_sq_value.publish(m)
                    m = Float32(); m.data = float(state.right_squeeze_value); pub_r_sq_value.publish(m)
            else: # controller
                logger_mp.info(f"[Left Trigger Value]: {teleData.left_trigger_value:.2f}")
                logger_mp.info(f"[Right Trigger Value]: {teleData.right_trigger_value:.2f}")
                # publish controller analogs
                if teleData.left_trigger_value is not None:
                    m = Float32(); m.data = float(teleData.left_trigger_value); pub_l_trig.publish(m)
                if teleData.right_trigger_value is not None:
                    m = Float32(); m.data = float(teleData.right_trigger_value); pub_r_trig.publish(m)
                
                if teleData.tele_state:
                    state = teleData.tele_state
                    logger_mp.info("[Controller State]:")
                    logger_mp.info(f"  Left Trigger: {state.left_trigger_state}")
                    logger_mp.info(f"  Left Squeeze: {state.left_squeeze_ctrl_state} ({state.left_squeeze_ctrl_value:.2f})")
                    logger_mp.info(f"  Left Thumbstick: {state.left_thumbstick_state} ({state.left_thumbstick_value})")
                    logger_mp.info(f"  Left A/B Buttons: A={state.left_aButton}, B={state.left_bButton}")
                    logger_mp.info(f"  Right Trigger: {state.right_trigger_state}")
                    logger_mp.info(f"  Right Squeeze: {state.right_squeeze_ctrl_state} ({state.right_squeeze_ctrl_value:.2f})")
                    logger_mp.info(f"  Right Thumbstick: {state.right_thumbstick_state} ({state.right_thumbstick_value})")
                    logger_mp.info(f"  Right A/B Buttons: A={state.right_aButton}, B={state.right_bButton}")
                    # publish controller states
                    m = Float32(); m.data = float(state.left_squeeze_ctrl_value); pub_l_sq.publish(m)
                    m = Float32(); m.data = float(state.right_squeeze_ctrl_value); pub_r_sq.publish(m)
                    t = Float32MultiArray(); t.data = state.left_thumbstick_value.astype(np.float32).tolist(); pub_l_thumb.publish(t)
                    t = Float32MultiArray(); t.data = state.right_thumbstick_value.astype(np.float32).tolist(); pub_r_thumb.publish(t)
                    b = Bool(); b.data = bool(state.left_aButton); pub_l_a.publish(b)
                    b = Bool(); b.data = bool(state.left_bButton); pub_l_b.publish(b)
                    b = Bool(); b.data = bool(state.right_aButton); pub_r_a.publish(b)
                    b = Bool(); b.data = bool(state.right_bButton); pub_r_b.publish(b)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, 0.033 - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        running = False
        logger_mp.warning("KeyboardInterrupt, exiting program...")
    finally:
        image_shm.unlink()
        image_shm.close()
        if ros_node is not None:
            ros_node.destroy_node()
            rclpy.shutdown()
        logger_mp.warning("Finally, exiting program...")
        exit(0)

if __name__ == '__main__':
    run_test_tv_wrapper()