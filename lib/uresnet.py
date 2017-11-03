from __future__ import absolute_import
#from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.python.platform
import tensorflow as tf
import tensorflow.contrib.layers as L
import tensorflow.contrib.slim as slim
from resnet_module import double_resnet
from ssnet import ssnet_base

class uresnet(ssnet_base):

    def __init__(self, rows, cols, num_class, num_strides=5, base_num_outputs=16, debug=False):
        super(uresnet,self).__init__(rows=rows,
                                     cols=cols,
                                     num_class=num_class)
        self._base_num_outputs = int(base_num_outputs)
        self._num_strides = int(num_strides)
        self._debug = bool(debug)

    def _build(self,input_tensor):
        
        if self._debug: print(input_tensor.shape, 'input shape')

        with tf.variable_scope('UResNet'):

            conv_feature_map={}

            # assume zero padding in each layer (set as default in uresnet_layers.py)
            net = slim.conv2d(inputs      = input_tensor,
                              num_outputs = self._base_num_outputs,
                              kernel_size = [7,7],
                              stride      = 1,
                              trainable   = self._trainable,
                              normalizer_fn = None,
                              activation_fn = tf.nn.relu,
                              scope       = 'conv0')

            conv_feature_map[net.get_shape()[-1].value] = net

            #net = slim.max_pool2d(inputs      = net,
            #                      kernel_size = [2,2],
            #                      stride      = 2
            #                      scope       = 'pool0')
            if self._debug: print(net.shape, 'after conv0')
            
            # Encoding steps
            for step in xrange(self._num_strides):
                net = double_resnet(input_tensor = net, 
                                    num_outputs  = net.get_shape()[-1].value * 2,
                                    trainable    = self._trainable,
                                    kernel       = [3,3],
                                    stride       = 2,
                                    scope        = 'resnet_module%d' % step)
                if self._debug: print(net.shape, 'after resnet_module%d' % step)
                conv_feature_map[net.get_shape()[-1].value] = net
            # Decoding steps
            for step in xrange(self._num_strides):
                num_outputs = net.get_shape()[-1].value / 2
                net = slim.conv2d_transpose(inputs      = net,
                                            num_outputs = num_outputs,
                                            kernel_size = [3,3],
                                            stride      = 2,
                                            padding     = 'same',
                                            activation_fn = None,
                                            trainable   = self._trainable,
                                            scope       = 'deconv%d' % step)
                if self._debug: print(net.shape, 'after deconv%d' % step)                    
                net = tf.concat([net, conv_feature_map[num_outputs]],
                                axis=3, 
                                name='concat%d' % step)
                if self._debug: print(net.shape, 'after concat%d' % step)
                net = double_resnet(input_tensor = net, 
                                    num_outputs  = num_outputs,
                                    trainable    = self._trainable,
                                    kernel       = [3,3],
                                    stride       = 1,
                                    scope        = 'resnet_module%d' % (step+5))
                if self._debug: print(net.shape, 'after resnet_module%d' % (step + self._num_strides))
            # Final conv layers
            net = slim.conv2d(inputs      = net,
                              num_outputs = self._base_num_outputs,
                              padding     = 'same',
                              kernel_size = [7,7],
                              stride      = 1,
                              trainable   = self._trainable,
                              normalizer_fn = None,
                              activation_fn = tf.nn.relu,
                              scope       = 'conv1')
            if self._debug: print(net.shape, 'after conv1')
            net = slim.conv2d(inputs      = net,
                              num_outputs = self._num_class,
                              padding     = 'same',
                              kernel_size = [3,3],
                              stride      = 1,
                              trainable   = self._trainable,
                              normalizer_fn = None,
                              activation_fn = tf.nn.relu,
                              scope       = 'conv2')
            if self._debug: print(net.shape, 'after conv2')
            return net

if __name__ == '__main__':
  # some constants
  BATCH=2
  ROWS=512
  COLS=512
  NUM_CLASS=3
  # make network
  net = uresnet(rows=ROWS,
                cols=COLS,
                num_class=NUM_CLASS,
                debug=True)
  net.construct(trainable=True,use_weight=True)

  import sys
  if 'save' in sys.argv:
      # Create a session
      sess = tf.InteractiveSession()
      sess.run(tf.global_variables_initializer())
      # Create a summary writer handle + save graph
      writer=tf.summary.FileWriter('uresnet_graph')
      writer.add_graph(sess.graph)
