import argparse
import os

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools

from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils

tf.enable_eager_execution()

def parse_range_image_and_camera_projection(frame):
  """Parse range images and camera projections given a frame.

  Args:
     frame: open dataset frame proto

  Returns:
     range_images: A dict of {laser_name,
       [range_image_first_return, range_image_second_return]}.
     camera_projections: A dict of {laser_name,
       [camera_projection_from_first_return,
        camera_projection_from_second_return]}.
    range_image_top_pose: range image pixel pose for top lidar.
  """
  range_images = {}
  camera_projections = {}
  range_image_top_pose = None
  for laser in frame.lasers:
    if len(laser.ri_return1.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
      range_image_str_tensor = tf.io.decode_compressed(
          laser.ri_return1.range_image_compressed, 'ZLIB')
      ri = dataset_pb2.MatrixFloat()
      ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
      range_images[laser.name] = [ri]

      if laser.name == dataset_pb2.LaserName.TOP:
        range_image_top_pose_str_tensor = tf.io.decode_compressed(
            laser.ri_return1.range_image_pose_compressed, 'ZLIB')
        range_image_top_pose = dataset_pb2.MatrixFloat()
        range_image_top_pose.ParseFromString(
            bytearray(range_image_top_pose_str_tensor.numpy()))

      camera_projection_str_tensor = tf.io.decode_compressed(
          laser.ri_return1.camera_projection_compressed, 'ZLIB')
      cp = dataset_pb2.MatrixInt32()
      cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
      camera_projections[laser.name] = [cp]
    if len(laser.ri_return2.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
      range_image_str_tensor = tf.io.decode_compressed(
          laser.ri_return2.range_image_compressed, 'ZLIB')
      ri = dataset_pb2.MatrixFloat()
      ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
      range_images[laser.name].append(ri)

      camera_projection_str_tensor = tf.io.decode_compressed(
          laser.ri_return2.camera_projection_compressed, 'ZLIB')
      cp = dataset_pb2.MatrixInt32()
      cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
      camera_projections[laser.name].append(cp)
  return range_images, camera_projections, range_image_top_pose


def convert_range_image_to_cartesian(frame,
                                     range_images,
                                     range_image_top_pose,
                                     ri_index=0,
                                     keep_polar_features=False):
  """Convert range images from polar coordinates to Cartesian coordinates.

  Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return,
       range_image_second_return]}.
    range_image_top_pose: range image pixel pose for top lidar.
    ri_index: 0 for the first return, 1 for the second return.
    keep_polar_features: If true, keep the features from the polar range image
      (i.e. range, intensity, and elongation) as the first features in the
      output range image.

  Returns:
    dict of {laser_name, (H, W, D)} range images in Cartesian coordinates. D
      will be 3 if keep_polar_features is False (x, y, z) and 6 if
      keep_polar_features is True (range, intensity, elongation, x, y, z).
  """
  cartesian_range_images = {}
  frame_pose = tf.convert_to_tensor(
      value=np.reshape(np.array(frame.pose.transform), [4, 4]))

  # [H, W, 6]
  range_image_top_pose_tensor = tf.reshape(
      tf.convert_to_tensor(value=range_image_top_pose.data),
      range_image_top_pose.shape.dims)
  # [H, W, 3, 3]
  range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
      range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
      range_image_top_pose_tensor[..., 2])
  range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
  range_image_top_pose_tensor = transform_utils.get_transform(
      range_image_top_pose_tensor_rotation,
      range_image_top_pose_tensor_translation)

  for c in frame.context.laser_calibrations:
    range_image = range_images[c.name][ri_index]
    if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
      beam_inclinations = range_image_utils.compute_inclination(
          tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
          height=range_image.shape.dims[0])
    else:
      beam_inclinations = tf.constant(c.beam_inclinations)

    beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
    extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

    range_image_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
    pixel_pose_local = None
    frame_pose_local = None
    if c.name == dataset_pb2.LaserName.TOP:
      pixel_pose_local = range_image_top_pose_tensor
      pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
      frame_pose_local = tf.expand_dims(frame_pose, axis=0)
    range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
        tf.expand_dims(range_image_tensor[..., 0], axis=0),
        tf.expand_dims(extrinsic, axis=0),
        tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
        pixel_pose=pixel_pose_local,
        frame_pose=frame_pose_local)

    range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)

    if keep_polar_features:
      # If we want to keep the polar coordinate features of range, intensity,
      # and elongation, concatenate them to be the initial dimensions of the
      # returned Cartesian range image.
      range_image_cartesian = tf.concat(
          [range_image_tensor[..., 0:3], range_image_cartesian], axis=-1)

    cartesian_range_images[c.name] = range_image_cartesian

  return cartesian_range_images


def convert_range_image_to_point_cloud(frame,
                                       range_images,
                                       camera_projections,
                                       range_image_top_pose,
                                       ri_index=0,
                                       keep_polar_features=False):
  """Convert range images to point cloud.

  Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return,
      range_image_second_return]}.
    camera_projections: A dict of {laser_name,
      [camera_projection_from_first_return,
      camera_projection_from_second_return]}.
    range_image_top_pose: range image pixel pose for top lidar.
    ri_index: 0 for the first return, 1 for the second return.
    keep_polar_features: If true, keep the features from the polar range image
      (i.e. range, intensity, and elongation) as the first features in the
      output range image.

  Returns:
    points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
      (NOTE: Will be {[N, 6]} if keep_polar_features is true.
    cp_points: {[N, 6]} list of camera projections of length 5
      (number of lidars).
  """
  calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
  points = []
  cp_points = []

  cartesian_range_images = convert_range_image_to_cartesian(
      frame, range_images, range_image_top_pose, ri_index, keep_polar_features)

  for c in calibrations:
    range_image = range_images[c.name][ri_index]
    range_image_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
    range_image_mask = range_image_tensor[..., 0] > 0

    range_image_cartesian = cartesian_range_images[c.name]
    points_tensor = tf.gather_nd(range_image_cartesian,
                                 tf.compat.v1.where(range_image_mask))

    cp = camera_projections[c.name][ri_index]
    cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
    cp_points_tensor = tf.gather_nd(cp_tensor,
                                    tf.compat.v1.where(range_image_mask))
    points.append(points_tensor.numpy())
    cp_points.append(cp_points_tensor.numpy())

  return points, cp_points


def convert_frame_to_dict(frame):
  """Convert the frame proto into a dict of numpy arrays.

  The keys, shapes, and data types are:
    POSE: 4x4 float32 array
    TIMESTAMP: int64 scalar

    For each lidar:
      <LIDAR_NAME>_BEAM_INCLINATION: H float32 array
      <LIDAR_NAME>_LIDAR_EXTRINSIC: 4x4 float32 array
      <LIDAR_NAME>_RANGE_IMAGE_FIRST_RETURN: HxWx6 float32 array
      <LIDAR_NAME>_RANGE_IMAGE_SECOND_RETURN: HxWx6 float32 array
      <LIDAR_NAME>_CAM_PROJ_FIRST_RETURN: HxWx6 int64 array
      <LIDAR_NAME>_CAM_PROJ_SECOND_RETURN: HxWx6 float32 array
      (top lidar only) TOP_RANGE_IMAGE_POSE: HxWx6 float32 array

    For each camera:
      <CAMERA_NAME>_IMAGE: HxWx3 uint8 array
      <CAMERA_NAME>_INTRINSIC: 9 float32 array
      <CAMERA_NAME>_EXTRINSIC: 4x4 float32 array
      <CAMERA_NAME>_WIDTH: int64 scalar
      <CAMERA_NAME>_HEIGHT: int64 scalar
      <CAMERA_NAME>_SDC_VELOCITY: 6 float32 array
      <CAMERA_NAME>_POSE: 4x4 float32 array
      <CAMERA_NAME>_POSE_TIMESTAMP: float32 scalar
      <CAMERA_NAME>_ROLLING_SHUTTER_DURATION: float32 scalar
      <CAMERA_NAME>_ROLLING_SHUTTER_DIRECTION: int64 scalar

  NOTE: This function only works in eager mode for now.

  See the LaserName.Name and CameraName.Name enums in dataset.proto for the
  valid lidar and camera name strings that will be present in the returned
  dictionaries.

  Args:
    frame: open dataset frame

  Returns:
    Dict from string field name to numpy ndarray.
  """
  range_images, camera_projection_protos, range_image_top_pose = (
      parse_range_image_and_camera_projection(frame))
  first_return_cartesian_range_images = convert_range_image_to_cartesian(
      frame, range_images, range_image_top_pose, ri_index=0,
      keep_polar_features=True)
  second_return_cartesian_range_images = convert_range_image_to_cartesian(
      frame, range_images, range_image_top_pose, ri_index=1,
      keep_polar_features=True)

  data_dict = {}

  # Save the beam inclinations, extrinsic matrices, first/second return range
  # images, and first/second return camera projections for each lidar.
  for c in frame.context.laser_calibrations:
    laser_name_str = dataset_pb2.LaserName.Name.Name(c.name)

    beam_inclination_key = f'{laser_name_str}_BEAM_INCLINATION'
    if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
      data_dict[beam_inclination_key] = range_image_utils.compute_inclination(
          tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
          height=range_images[c.name][0].shape.dims[0]).numpy()
    else:
      data_dict[beam_inclination_key] = np.array(
          c.beam_inclinations, np.float32)

    data_dict[f'{laser_name_str}_LIDAR_EXTRINSIC'] = np.reshape(
        np.array(c.extrinsic.transform, np.float32), [4, 4])

    data_dict[f'{laser_name_str}_RANGE_IMAGE_FIRST_RETURN'] = (
        first_return_cartesian_range_images[c.name].numpy())
    data_dict[f'{laser_name_str}_RANGE_IMAGE_SECOND_RETURN'] = (
        second_return_cartesian_range_images[c.name].numpy())

    first_return_cp = camera_projection_protos[c.name][0]
    data_dict[f'{laser_name_str}_CAM_PROJ_FIRST_RETURN'] = np.reshape(
        np.array(first_return_cp.data), first_return_cp.shape.dims)

    second_return_cp = camera_projection_protos[c.name][1]
    data_dict[f'{laser_name_str}_CAM_PROJ_SECOND_RETURN'] = np.reshape(
        np.array(second_return_cp.data), second_return_cp.shape.dims)

  # Save the H x W x 3 RGB image for each camera, extracted from JPEG.
  for im in frame.images:
    cam_name_str = dataset_pb2.CameraName.Name.Name(im.name)
    data_dict[f'{cam_name_str}_IMAGE'] = tf.io.decode_jpeg(im.image).numpy()
    data_dict[f'{cam_name_str}_SDC_VELOCITY'] = np.array([
        im.velocity.v_x, im.velocity.v_y, im.velocity.v_z, im.velocity.w_x,
        im.velocity.w_y, im.velocity.w_z
    ], np.float32)
    data_dict[f'{cam_name_str}_POSE'] = np.reshape(
        np.array(im.pose.transform, np.float32), (4, 4))
    data_dict[f'{cam_name_str}_POSE_TIMESTAMP'] = np.array(
        im.pose_timestamp, np.float32)
    data_dict[f'{cam_name_str}_ROLLING_SHUTTER_DURATION'] = np.array(im.shutter)

  # Save the intrinsics, 4x4 extrinsic matrix, width, and height of each camera.
  for c in frame.context.camera_calibrations:
    cam_name_str = dataset_pb2.CameraName.Name.Name(c.name)
    data_dict[f'{cam_name_str}_INTRINSIC'] = np.array(c.intrinsic, np.float32)
    data_dict[f'{cam_name_str}_EXTRINSIC'] = np.reshape(
        np.array(c.extrinsic.transform, np.float32), [4, 4])
    data_dict[f'{cam_name_str}_WIDTH'] = np.array(c.width)
    data_dict[f'{cam_name_str}_HEIGHT'] = np.array(c.height)
    data_dict[f'{cam_name_str}_ROLLING_SHUTTER_DURATION'] = np.array(
        c.rolling_shutter_direction)

  # Save the range image pixel pose for the top lidar.
  data_dict['TOP_RANGE_IMAGE_POSE'] = np.reshape(
      np.array(range_image_top_pose.data, np.float32),
      range_image_top_pose.shape.dims)

  data_dict['POSE'] = np.reshape(
      np.array(frame.pose.transform, np.float32), (4, 4))
  data_dict['TIMESTAMP'] = np.array(frame.timestamp_micros)

  return data_dict

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_dir', type=str, required=True)
  parser.add_argument('--output_dir', type=str, required=True)
  args = parser.parse_args()

  dataset = tf.data.TFRecordDataset(args.dataset_dir, compression_type='')
  ouput_dir = args.output_dir

  for i,data in enumerate(dataset):
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    result = convert_frame_to_dict(frame)
    context_dir = ouput_dir + frame.context.name
    time_dir = context_dir + "/" + str(frame.timestamp_micros)
    if not os.path.exists(context_dir):
      os.mkdir(context_dir)
    if not os.path.exists(time_dir):
      os.mkdir(time_dir)
    for key in result:
      np.save(time_dir +"/"+ key + ".npy", result[key])

