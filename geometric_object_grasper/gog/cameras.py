import numpy as np
import pybullet as p


"""Camera configs."""


class RealSense:
    """Default configuration with 2 RealSense RGB-D cameras."""

    image_size = (480, 640)
    intrinsics = (450., 0, 320., 0, 450., 240., 0, 0, 1)
    z_near = -0.01
    z_far = 10

    # Set default camera poses. (w.r.t. workspace center)

    # UP
    front_position = np.array([1, 0.0, 1.5])
    front_target_pos = np.array([0.0, 0.0, 0.3])
    front_up_vector = np.array([0.0, 0.0, 1.0])

    top_position = np.array([0.0, 0.0, 1.5])
    top_target_pos = np.array([0.0, 0.0, 0.3])
    top_up_vector = np.array([0.0, -1.0, 0.0])

    back_position = np.array([-1, 0.0, 1.5])
    back_target_pos = np.array([0.0, 0.0, 0.3])
    back_up_vector = np.array([0.0, 0.0, 1.0])

    side_1_position = np.array([0.0, 1, 1.5])
    side_1_target_pos = np.array([0.0, 0.0, 0.3])
    side_1_up_vector = np.array([0.0, 0.0, 1.0])

    side_2_position = np.array([0.0, -1, 1.5])
    side_2_target_pos = np.array([0.0, 0.0, 0.3])
    side_2_up_vector = np.array([0.0, 0.0, 1.0])

    # MIDDLE
    front_position_middle = np.array([1.5, 0.0, 0.3])
    front_target_pos_middle = np.array([0.0, 0.0, 0.3])
    front_up_vector_middle = np.array([0.0, 0.0, 1.0])

    back_position_middle = np.array([-1.5, 0.0, 0.3])
    back_target_pos_middle = np.array([0.0, 0.0, 0.3])
    back_up_vector_middle = np.array([0.0, 0.0, 1.0])

    side_1_position_middle = np.array([0.0, 1.5, 0.3])
    side_1_target_pos_middle = np.array([0.0, 0.0, 0.3])
    side_1_up_vector_middle = np.array([0.0, 0.0, 1.0])

    side_2_position_middle = np.array([0.0, -1.5, 0.3])
    side_2_target_pos_middle = np.array([0.0, 0.0, 0.3])
    side_2_up_vector_middle = np.array([0.0, 0.0, 1.0])


    # DOWN
    front_position_down = np.array([1, 0.0, -1])
    front_target_pos_down = np.array([-1.0, 0.0, 0.3])
    front_up_vector_down = np.array([0.0, 0.0, 1.0])

    bottom_position_down = np.array([0.0, 0.0, -1])
    bottom_target_pos_down = np.array([0.0, 0.0, 0.3])
    bottom_up_vector_down = np.array([0.0, -1.0, 0.0])

    back_position_down = np.array([-1, 0.0, -1])
    back_target_pos_down = np.array([1.0, 0.0, 0.3])
    back_up_vector_down = np.array([0.0, 0.0, 1.0])

    side_1_position_down = np.array([0.0, 1, -1])
    side_1_target_pos_down = np.array([0.0, -1.5, 0.3])
    side_1_up_vector_down = np.array([0.0, 0.0, 1.0])

    side_2_position_down = np.array([0.0, -1, -1])
    side_2_target_pos_down = np.array([0.0, 1.5, 0.3])
    side_2_up_vector_down = np.array([0.0, 0.0, 1.0])

    # Default camera configs.
    CONFIG = [
    {
      'image_size': image_size,
      'intrinsics': intrinsics,
      'pos': front_position,
      'target_pos': front_target_pos,
      'up_vector': front_up_vector,
      'zrange': (z_near, z_far)},
    {
     'image_size': image_size,
     'intrinsics': intrinsics,
     'pos': top_position,
     'target_pos': top_target_pos,
     'up_vector': top_up_vector,
     'zrange': (z_near, z_far)},
    {
     'image_size': image_size,
     'intrinsics': intrinsics,
     'pos': back_position,
     'target_pos': back_target_pos,
     'up_vector': back_up_vector,
     'zrange': (z_near, z_far)},
    {
    'image_size': image_size,
    'intrinsics': intrinsics,
    'pos': side_1_position,
    'target_pos': side_1_target_pos,
    'up_vector': side_1_up_vector,
    'zrange': (z_near, z_far)},
    {
     'image_size': image_size,
     'intrinsics': intrinsics,
     'pos': side_2_position,
     'target_pos': side_2_target_pos,
     'up_vector': side_2_up_vector,
     'zrange': (z_near, z_far)},
    {
      'image_size': image_size,
      'intrinsics': intrinsics,
      'pos': front_position_down,
      'target_pos': front_target_pos_down,
      'up_vector': front_up_vector_down,
      'zrange': (z_near, z_far)},
    {
     'image_size': image_size,
     'intrinsics': intrinsics,
     'pos': bottom_position_down,
     'target_pos': bottom_target_pos_down,
     'up_vector': bottom_up_vector_down,
     'zrange': (z_near, z_far)},
    {
     'image_size': image_size,
     'intrinsics': intrinsics,
     'pos': back_position_down,
     'target_pos': back_target_pos_down,
     'up_vector': back_up_vector_down,
     'zrange': (z_near, z_far)},
    {
    'image_size': image_size,
    'intrinsics': intrinsics,
    'pos': side_1_position_down,
    'target_pos': side_1_target_pos_down,
    'up_vector': side_1_up_vector_down,
    'zrange': (z_near, z_far)},
    {
     'image_size': image_size,
     'intrinsics': intrinsics,
     'pos': side_2_position_down,
     'target_pos': side_2_target_pos_down,
     'up_vector': side_2_up_vector_down,
     'zrange': (z_near, z_far)},
    {
     'image_size': image_size,
     'intrinsics': intrinsics,
     'pos': front_position_middle,
     'target_pos': front_target_pos_middle,
     'up_vector': front_up_vector_middle,
     'zrange': (z_near, z_far)},
    {
     'image_size': image_size,
     'intrinsics': intrinsics,
     'pos': back_position_middle,
     'target_pos': back_target_pos_middle,
     'up_vector': back_up_vector_middle,
     'zrange': (z_near, z_far)},
    {
    'image_size': image_size,
    'intrinsics': intrinsics,
    'pos': side_1_position_middle,
    'target_pos': side_1_target_pos_middle,
    'up_vector': side_1_up_vector_middle,
    'zrange': (z_near, z_far)},
    {
     'image_size': image_size,
     'intrinsics': intrinsics,
     'pos': side_2_position_middle,
     'target_pos': side_2_target_pos_middle,
     'up_vector': side_2_up_vector_middle,
     'zrange': (z_near, z_far)}
    ]