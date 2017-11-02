import sys
import tensorflow as tf
import numpy as np

class ssnet_base(object):

  def __init__(self, rows, cols, num_class):
    self._rows = int(rows)
    self._cols = int(cols)
    self._num_class = int(num_class)

  def _build(self,input_tensor):
    raise NotImplementedError

  def construct(self,trainable=True,use_weight=True):

    self._trainable  = bool(trainable)
    self._use_weight = bool(use_weight)

    with tf.variable_scope('input_prep'):
      self._input_image  = tf.placeholder(tf.float32, [None, self._rows * self._cols], name='input_image' )
      self._input_weight = tf.placeholder(tf.float32, [None, self._rows * self._cols], name='input_weight')
      self._input_label  = tf.placeholder(tf.float32, [None, self._rows * self._cols], name='input_label' )
      
      image_nhwc = tf.reshape(self._input_image, [-1,self._rows,self._cols,1], name='image_reshape' )
      label_nhw  = tf.reshape(self._input_label, [-1,self._rows,self._cols  ], name='label_reshape' )
      weight_nhw = tf.reshape(self._input_label, [-1,self._rows,self._cols  ], name='weight_reshape')

      label_nhw = tf.cast(label_nhw,tf.int64)
      
    net = self._build(input_tensor=image_nhwc)

    self._softmax = None
    self._train   = None
    self._loss    = None
    self._accuracy_allpix = None
    self._accuracy_nonzero = None

    with tf.variable_scope('metrics'):
      self._accuracy_allpix = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net,3), label_nhw),tf.float32))
      nonzero_idx = tf.where(tf.reshape(image_nhwc, [-1,self._rows,self._cols]) > 10.)
      nonzero_label = tf.gather_nd(label_nhw,nonzero_idx)
      nonzero_pred  = tf.gather_nd(tf.argmax(net,3),nonzero_idx)
      self._accuracy_nonzero = tf.reduce_mean(tf.cast(tf.equal(nonzero_label,nonzero_pred),tf.float32))
      self._softmax = tf.nn.softmax(logits=net)

    if self._trainable:
      with tf.variable_scope('train'):
        if self._use_weight:
          loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_nhw, logits=net)
          self._loss = tf.reduce_mean(tf.multiply(weight_nhw, loss))
        else:
          self._loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_nhw, logits=net))
        self._train = tf.train.RMSPropOptimizer(0.0003).minimize(loss)

        tf.summary.image('data_example',image_nhwc,10)
        tf.summary.scalar('accuracy_all', self._accuracy_allpix)
        tf.summary.scalar('accuracy_nonzero', self._accuracy_nonzero)
        tf.summary.scalar('loss',self._loss)

  def train(self,sess,input_image,input_label,input_weight=None):

    feed_dict = self.feed_dict(input_image  = input_image,
                               input_label  = input_label,
                               input_weight = input_weight)

    ops = [self._train,self._loss,self._accuracy_allpix,self._accuracy_nonzero]

    return sess.run( ops, feed_dict = feed_dict )

  def inference(self,sess,input_image,input_label=None):
    
    feed_dict = self.feed_dict(input_image=input_image, input_label=input_label)

    ops = [self._softmax]
    if input_label is not None:
      ops.append(self._accuracy_allpix)
      ops.append(self._accuracy_nonzero)

    return sess.run( ops, feed_dict = feed_dict )

  def feed_dict(self,input_image,input_label=None,input_weight=None):

    if input_weight is None and self._use_weight:
      sys.stderr.write('Network configured to use loss pixel-weighting. Cannot run w/ input_weight=None...\n')
      raise TypeError

    feed_dict = { self._input_image : input_image }
    if input_label is not None:
      feed_dict[ self._input_label ] = input_label
    if input_weight is not None:
      feed_dict[ self._input_weight ] = input_weight
    
    return feed_dict

