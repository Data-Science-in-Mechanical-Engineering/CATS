# Copyright 2023 Big Vision Authors.
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

"""AutoAugment and RandAugment policies for enhanced image preprocessing.

AutoAugment Reference: https://arxiv.org/abs/1805.09501
RandAugment Reference: https://arxiv.org/abs/1909.13719

This code is forked from
https://github.com/tensorflow/tpu/blob/11d0db15cf1c3667f6e36fecffa111399e008acd/models/official/efficientnet/autoaugment.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dataclasses
import inspect
import math
import tensorflow as tf

"""
Method from http://arxiv.org/pdf/1909.13719, different to the paper the magnitude is between 0 and 1
"""

def identity(magnitude):
    """Identity function."""
    return lambda x, training: x


def brightness(magnitude):
    return tf.keras.layers.RandomBrightness((-magnitude, magnitude))

def contrast(magnitude):
    return tf.keras.layers.RandomContrast((-magnitude, magnitude))

def flip(magnitude):
    return tf.keras.layers.RandomFlip()

def rotate(magnitude):
    return tf.keras.layers.RandomRotation(factor=magnitude)

def translate(magnitude):
    return tf.keras.layers.RandomTranslation(magnitude, magnitude)

def zoom(magnitude):
    return tf.keras.layers.RandomZoom(magnitude)

def init_randaugment(magnitude=0.5):
    """Initialize the RandAugment policy.
    Returns:
        A list of functions that represent the RandAugment policy.
    """
    return [
        identity(magnitude),
        brightness(magnitude),
        contrast(magnitude),
        flip(magnitude),
        rotate(magnitude),
        translate(magnitude),
        zoom(magnitude),
    ]


def distort_image_with_randaugment(image, num_layers, functions):
    """Applies the RandAugment policy to `image`.
    RandAugment is from the paper https://arxiv.org/abs/1909.13719,
    Args:
    image: `Tensor` of shape [height, width, 3] representing an image.
    num_layers: Integer, the number of augmentation transformations to apply
        sequentially to an image. Represented as (N) in the paper. Usually best
        values will be in the range [1, 3].
    magnitude: Integer, shared magnitude across all augmentation operations.
        Represented as (M) in the paper. Usually best values are in the range
        [5, 30].
    Returns:
    The augmented version of `image`.
    """
    replace_value = [128] * 3

    for layer_num in range(num_layers):
        op_to_select = tf.random.uniform(
            [], maxval=len(functions), dtype=tf.int32)
        with tf.name_scope('randaug_layer_{}'.format(layer_num)):
            for (i, op) in enumerate(functions):
                image = tf.cond(tf.equal(i, op_to_select), lambda: op(image, training=True), lambda: image)
    return image
