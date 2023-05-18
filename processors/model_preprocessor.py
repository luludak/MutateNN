from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np
import math
import onnxruntime as rt
import json

import importlib
import tensorflow
import tensorflow as tf
from tensorflow import keras

from .vgg_preprocessor import preprocess_image as vgg_preprocess_image

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_SCALE_FACTOR = 0.017

class ModelPreprocessor:

    def __init__(self, data):
        self.data = data


    def preprocess(self, model_name, img_data, preprocessing_enabled):

        lower_model_name = model_name.lower()
        library_or_name = self.data["library"] if self.data["library"] is not None else lower_model_name


        if (not preprocessing_enabled):
            img_data = np.expand_dims(np.array(img_data), axis=0)
            if ((self.data["input"][1] == 3 and img_data.shape[1] != 3 or (self.data["input"][1] != 3  and img_data.shape[1] == 3))):
                img_data = np.transpose(img_data, (0, 3, 2, 1))
        
            return img_data
            
        #  -------------------- Model preprocessing --------------------

        # Note: We use preprocessing based on the repos of each model.
        # If extended, testing of model on a small dataset is necessary.

        # Special case handling for MobileNet - using Keras.
        if ("mobilenet" in lower_model_name):
            img_data = np.array(img_data)
            module_load = importlib.import_module("keras.applications.mobilenet_v2")

            img_data = np.expand_dims(module_load.preprocess_input(img_data), axis=0)
            # In case of tensor misalignment, apply transpose.
            if(self.data["input"][2] == 3):
                if (img_data.shape[1] == 3):
                    return np.transpose(img_data, (0, 2, 1, 3))
                elif (img_data.shape[3] == 3):
                    return np.transpose(img_data, (0, 2, 3, 1))
        
            if (self.data["input"][1] == 3 and img_data.shape[1] != 3 or (self.data["input"][1] != 3  and img_data.shape[1] == 3)):
                return np.transpose(img_data, (0, 3, 1, 2))


            return img_data

        img_data = np.array(img_data)
        img_data_orig = img_data
        img_data = np.expand_dims(img_data, axis=0)

        if(img_data.shape[1] != 3):
            img_data = np.transpose(img_data, (0, 3, 2, 1))

        preprocessed_value = img_data


        if("googlenet" in lower_model_name):
            preprocessed_value = self.preprocess_googlenet(img_data)
    
        elif ("densenet" in lower_model_name):
            
            # Note: this is the suggested preprocessing for densenet etc.
            preprocessed_value = vgg_preprocess_image(img_data_orig, img_data.shape[2], img_data.shape[2])
            return np.expand_dims(np.array(preprocessed_value), axis=0)

        elif ("shufflenet" in lower_model_name):

            preprocessed_value = self.preprocess_imagenet(preprocessed_value)
    
        elif("inception" in lower_model_name or "resnet" in lower_model_name):
            preprocessed_value = self.preprocess_inception(preprocessed_value)
        
        if ((self.data["input"][1] == 3 and preprocessed_value.shape[1] != 3 or (self.data["input"][1] != 3  and preprocessed_value.shape[1] == 3))):
            preprocessed_value = np.transpose(preprocessed_value, (0, 3, 2, 1))
    

        if(self.data["input"][2] == 3):
            if (preprocessed_value.shape[1] == 3):
                return np.transpose(preprocessed_value, (0, 2, 1, 3))
            elif (preprocessed_value.shape[3] == 3):
                return np.transpose(preprocessed_value, (0, 2, 3, 1))
        return preprocessed_value

    def preprocess_inception(self, img): 

        norm_img_data = np.array(img).astype(np.float32)
    
        for i in range(img.shape[1]):
            norm_img_data[0,i,:,:] /= 255.0
            norm_img_data[0,i,:,:] -= 0.5
            norm_img_data[0,i,:,:] *= 2.0

        return norm_img_data

    def preprocess_googlenet(self, img):
        img = np.array(img).astype(np.float32)
        #print(img.shape)
        img[0, 0, :, :] -= 123.68
        img[0, 1,:, :] -= 116.779
        img[0, 2,:, :] -= 103.939
        img[0, [0,1,2],:,:] = img[0, [2,1,0],:,:]

        return img  # * 0.017

    def preprocess_imagenet(self, img):
        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])
        norm_img_data = np.zeros(img.shape).astype('float32')
        for i in range(img.shape[1]):
            # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
            norm_img_data[0, i,:,:] = (img[0, i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
        return norm_img_data

    def preprocess_densenet(self, image, output_height, output_width):
        image = tf.squeeze(image, [0])
        if(image.shape[0] == 3):
            image = np.transpose(image, (1, 2, 0))
        
        image = tf.image.resize(image, [output_height, output_width])
        image = tf.image.central_crop(image, central_fraction=1)
        
        image.set_shape([output_height, output_width, 3])

        image = self._mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
        return np.expand_dims(image * _SCALE_FACTOR, axis=0)

    def _mean_image_subtraction(self, image, means):
        """Subtracts the given means from each image channel.
        For example:
            means = [123.68, 116.779, 103.939]
            image = _mean_image_subtraction(image, means)
        Note that the rank of `image` must be known.
        Args:
            image: a tensor of size [height, width, C].
            means: a C-vector of values to subtract from each channel.
        Returns:
            the centered image.
        Raises:
            ValueError: If the rank of `image` is unknown, if `image` has a rank other
            than three or if the number of channels in `image` doesn't match the
            number of values in `means`.
        """
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        num_channels = image.get_shape().as_list()[-1]
        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')

        channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
        for i in range(num_channels):
            channels[i] -= means[i]
        return tf.concat(axis=2, values=channels)
