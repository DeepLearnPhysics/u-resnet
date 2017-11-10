from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np

class ssnet_base(object):

  def __init__(self, dims, num_class):
    self._dims = np.array(dims,np.int32)
    if not len(self._dims) in [3,4]:
      print('Error: len(dims) =',len(self._dims),'but only 3 (H,W,C) or 4 (H,W,D,C) supported!')
      raise NotImplementedError
    self._num_class = int(num_class)

  def _build(self,input_tensor):
    raise NotImplementedError

  def construct(self,trainable=True,use_weight=True):

    self._trainable  = bool(trainable)
    self._use_weight = bool(use_weight)

    entry_size = 1
    for dim in self._dims: 
      entry_size *= dim

    with tf.variable_scope('input_prep'):
      self._input_data   = tf.placeholder(tf.float32, [None, entry_size], name='input_data'  )
      self._input_weight = tf.placeholder(tf.float32, [None, entry_size], name='input_weight')
      self._input_label  = tf.placeholder(tf.float32, [None, entry_size], name='input_label' )
      
      shape_dim = [-1]
      for dim in self._dims: shape_dim.append(dim)

      data   = tf.reshape(self._input_data,   shape_dim,      name='data_reshape'  )
      label  = tf.reshape(self._input_label,  shape_dim[:-1], name='label_reshape' )
      weight = tf.reshape(self._input_weight, shape_dim[:-1], name='weight_reshape')

      label = tf.cast(label,tf.int64)
      
    net = self._build(input_tensor=data)

    self._softmax = None
    self._train   = None
    self._loss    = None
    self._accuracy_allpix = None
    self._accuracy_nonzero = None

    with tf.variable_scope('metrics'):
      self._accuracy_allpix = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net,len(self._dims)), label),tf.float32))
      nonzero_idx = tf.where(tf.reshape(data, shape_dim[:-1]) > 10.)
      nonzero_label = tf.gather_nd(label,nonzero_idx)
      nonzero_pred  = tf.gather_nd(tf.argmax(net,len(self._dims)),nonzero_idx)
      self._accuracy_nonzero = tf.reduce_mean(tf.cast(tf.equal(nonzero_label,nonzero_pred),tf.float32))
      self._softmax = tf.nn.softmax(logits=net)

    if self._trainable:
      with tf.variable_scope('train'):
        self._loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=net)
        if self._use_weight:
          self._loss = tf.multiply(weight,self._loss)
        self._train = tf.train.RMSPropOptimizer(0.0003).minimize(self._loss)
        self._loss = tf.reduce_mean(tf.reduce_sum(tf.reshape(self._loss,[-1, int(entry_size / self._dims[-1])]),axis=1))

      if len(self._dims) == 3:
        tf.summary.image('data_example',data,10)
      tf.summary.scalar('accuracy_all', self._accuracy_allpix)
      tf.summary.scalar('accuracy_nonzero', self._accuracy_nonzero)
      tf.summary.scalar('loss',self._loss)

  def train(self,sess,input_data,input_label,input_weight=None):

    feed_dict = self.feed_dict(input_data   = input_data,
                               input_label  = input_label,
                               input_weight = input_weight)

    ops = [self._train,self._loss,self._accuracy_allpix,self._accuracy_nonzero]

    return sess.run( ops, feed_dict = feed_dict )

  def inference(self,sess,input_data,input_label=None):
    
    feed_dict = self.feed_dict(input_data=input_data, input_label=input_label)

    ops = [self._softmax]
    if input_label is not None:
      ops.append(self._accuracy_allpix)
      ops.append(self._accuracy_nonzero)

    return sess.run( ops, feed_dict = feed_dict )

  def feed_dict(self,input_data,input_label=None,input_weight=None):

    if input_weight is None and self._use_weight:
      sys.stderr.write('Network configured to use loss pixel-weighting. Cannot run w/ input_weight=None...\n')
      raise TypeError

    feed_dict = { self._input_data : input_data }
    if input_label is not None:
      feed_dict[ self._input_label ] = input_label
    if input_weight is not None:
      feed_dict[ self._input_weight ] = input_weight
    
    return feed_dict

