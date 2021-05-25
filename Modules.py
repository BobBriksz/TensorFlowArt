import tensorflow as tf 
import IPython.display as display

import matplotlib.pyplot as plt 
import matplotlib as mpl 
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np 
import PIL.Image
import time
import functools


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        """expects float input in [0,1]"""
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])


        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {content_name:value 
                        for content_name, value 
                        in zip(self.content_layers, content_outputs)}
        
        style_dict = {style_name:value 
                      for style_name, value 
                      in zip(self.style_layers, style_outputs)}
       
        return {'content':content_dict, 'style': style_dict}


def tensor_to_image(tf_input):
    tf_input = tf_input*255
    tf_input = np.array(tf_input, dtype=np.uint8)
    if np.ndim(tf_input)>3:
        assert tf_input.shape[0] == 1
        tf_input = tf_input[0]
    return PIL.Image.fromarray(tf_input)

def load_img(image_path):
    max_dim = 512
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3) # Detects the image to perform appropriate operations
    img = tf.image.convert_image_dtype(img, tf.float32) #converts image to tensor dtype

    shape = tf.cast(tf.shape(img)[:-1], tf.float32) # Casts a Tensor to float 32

    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)

    return img[tf.newaxis, :]


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    tf_outs = [vgg.get_layer(layer).output for layer in layer_names]

    model = tf.keras.Model([vgg.input], tf_outs)

    return model

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0,clip_value_max=1.0)

