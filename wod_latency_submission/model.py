# Copyright 2021 The Waymo Open Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Module to load and run a SavedModel."""
from typing import Dict
import importlib_resources
import numpy as np
import tensorflow as tf
import os

class Model(object):
  """A simple wrapper around a SavedModel."""

  def __init__(self):
    """Create an empty object with no underlying model yet."""
    self._model = None
    self._model_fn = None

  def initialize(self, model_dir: str):
    """Load the model from the SavedModel at model_dir.

    Args:
      model_dir: string directory of the SavedModel.
    """
    self._model = tf.saved_model.load(model_dir)


DATA_FIELDS = ['FRONT_IMAGE']
model = Model()


def initialize_model():
 saved_model_path = os.path.dirname(__file__) + "/saved_model"
 model.initialize(saved_model_path)
  # Run the model once on dummy input to warm it up.
 #run_model(np.zeros((64, 2650, 6)))


def run_model(FRONT_IMAGE):
  """Run the model on the 6-dimensional range image.

  Args:
    TOP_RANGE_IMAGE_FIRST_RETURN: H x W x 6 numpy ndarray

  Returns:
    Dict from string to numpy ndarray.
  """
  # Convert the numpy array to the desired formats (including converting to TF
  # Tensor.)

  input_tensor = tf.convert_to_tensor(np.expand_dims(FRONT_IMAGE, 0))
  detections = model._model(input_tensor)
  
  pred_boxes = detections['detection_boxes'][0].numpy() 
  pred_class = detections['detection_classes'][0].numpy().astype(np.int32)
  pred_score = detections['detection_scores'][0].numpy() 

  # Return the Tensors converted into numpy arrays.
  return {
      # Take the first example to go from 1 x N (x 7) to N (x 7).
      'boxes': pred_boxes,
      'scores': pred_score,
      # Add a "classes" field that is always CAR.
      'classes': pred_class,
  }


