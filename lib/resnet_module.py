from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow.python.platform
import tensorflow as tf
import tensorflow.contrib.layers as L
import tensorflow.contrib.slim as slim

def resnet_module(input_tensor, num_outputs, trainable=True, kernel=(3,3), stride=1, scope='noscope'):

    fn_conv = slim.conv2d
    if len(input_tensor.shape) == 5:
        fn_conv = slim.conv3d

    num_inputs  = input_tensor.get_shape()[-1].value
    with tf.variable_scope(scope):
        #
        # shortcut path
        #
        shortcut = None
        if num_outputs == num_inputs and stride ==1 :
            shortcut = input_tensor
        else:
            shortcut = fn_conv(inputs      = input_tensor,
                               num_outputs = num_outputs,
                               kernel_size = 1,
                               stride      = stride,
                               trainable   = trainable,
                               padding     = 'same',
                               normalizer_fn = slim.batch_norm, 
                               activation_fn = None,
                               scope       = 'shortcut')
        #
        # residual path
        #
        residual = input_tensor
        #residual = L.batch_norm(inputs     = residual,
        #                        epsilon    = 0.00001,
        #                        activation_fn = tf.nn.relu,
        #                        scope      = 'resnet_bn1',
        #                        trainable  = trainable)
        residual = fn_conv(inputs      = residual,
                           num_outputs = num_outputs/4,
                           kernel_size = 1,
                           stride      = 1,
                           trainable   = trainable,
                           padding     = 'same',
                           normalizer_fn = slim.batch_norm,
                           activation_fn = tf.nn.relu,
                           scope       = 'resnet_conv1')

        residual = fn_conv(inputs      = residual,
                           num_outputs = num_outputs/4,
                           kernel_size = kernel,
                           stride      = stride,
                           trainable   = trainable,
                           padding     = 'same',
                           normalizer_fn = slim.batch_norm,
                           activation_fn = tf.nn.relu,
                           scope       = 'resnet_conv1')
        
        #residual = L.batch_norm(inputs     = residual, 
        #                        epsilon    = 0.00001,
        #                        activation_fn = tf.nn.relu,
        #                        scope      = 'resnet_bn2',
        #                        trainable  = trainable)
        residual = fn_conv(inputs      = residual,
                           num_outputs = num_outputs,
                           kernel_size = 1,
                           stride      = 1,
                           trainable   = trainable,
                           padding     = 'same',
                           normalizer_fn = slim.batch_norm,
                           activation_fn = None,
                           scope       = 'resnet_conv2')
        
        return tf.nn.relu(shortcut + residual)

def double_resnet(input_tensor, num_outputs, trainable=True, kernel=3, stride=1, scope='noscope'):

    with tf.variable_scope(scope):

        resnet1 = resnet_module(input_tensor=input_tensor,
                                trainable=trainable,
                                kernel=kernel,
                                stride=stride,
                                num_outputs=num_outputs,
                                scope='module1')
        
        resnet2 = resnet_module(input_tensor=resnet1,
                                trainable=trainable,
                                kernel=kernel,
                                stride=1,
                                num_outputs=num_outputs,
                                scope='module2')
        return resnet2
    
# script unit test
if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [50,28,28,1])
    net = double_resnet(input_tensor=x,num_outputs=32)

    import sys
    if 'save' in sys.argv:
        # Create a session
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        # Create a summary writer handle + save graph
        writer=tf.summary.FileWriter('double_resnet_graph')
        writer.add_graph(sess.graph)
