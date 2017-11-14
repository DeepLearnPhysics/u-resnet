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

  def construct(self,trainable=True,use_weight=True, learning_rate=None, vertex=False):

    self._trainable  = bool(trainable)
    self._use_weight = bool(use_weight)
    self._learning_rate = learning_rate
    self._vertex = bool(vertex)

    entry_size = np.prod(self._dims)

    with tf.variable_scope('input_prep'):
      self._input_data   = tf.placeholder(tf.float32, [None, entry_size], name='input_data'  )
      self._input_weight = tf.placeholder(tf.float32, [None, entry_size], name='input_weight')
      self._input_label  = tf.placeholder(tf.float32, [None, entry_size], name='input_label' )
      self._input_label_vertex = tf.placeholder(tf.float32, [None, entry_size], name='input_label_vertex' )
      
      shape_dim = np.insert(self._dims, 0, -1)

      data   = tf.reshape(self._input_data,   shape_dim,      name='data_reshape'  )
      label  = tf.reshape(self._input_label,  shape_dim[:-1], name='label_reshape' )
      label_vertex = tf.reshape(self._input_label_vertex,  shape_dim[:-1], name='label_vertex_reshape' )
      weight = tf.reshape(self._input_weight, shape_dim[:-1], name='weight_reshape')

      label = tf.cast(label,tf.int64)
      label_vertex = tf.cast(label_vertex, tf.int64)
      
    net = self._build(input_tensor=data)

    self._softmax = None
    self._train   = None
    self._loss    = None
    self._accuracy_allpix = None
    self._accuracy_nonzero = None
    self._accuracy_vertex = None

    with tf.variable_scope('accum_grad'):
      self._accum_vars = [tf.Variable(tv.initialized_value(),
                                      trainable=False) for tv in tf.trainable_variables()]

    with tf.variable_scope('metrics'):
      if self._vertex:
        net, net_vertex = tf.split(net, [3,1], axis=3)
        self._accuracy_vertex = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net_vertex,len(self._dims)), label),tf.float32))
      self._accuracy_allpix = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net,len(self._dims)), label),tf.float32))
      nonzero_idx = tf.where(tf.reshape(data, shape_dim[:-1]) > 10.)
      nonzero_label = tf.gather_nd(label,nonzero_idx)
      nonzero_pred  = tf.gather_nd(tf.argmax(net,len(self._dims)),nonzero_idx)
      self._accuracy_nonzero = tf.reduce_mean(tf.cast(tf.equal(nonzero_label,nonzero_pred),tf.float32))
      self._softmax = tf.nn.softmax(logits=net)

    if self._trainable:
      if self._vertex:
        net, net_vertex = tf.split(net, [self._dims[-1]-1,1], axis=len(self._dims)-1)
        with tf.variable_scope('train_vertex'):
          self._loss_vertex = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_vertex, logits=net_vertex)
          if self._use_weight:
            self._loss_vertex = tf.multiply(weight, self._loss_vertex)
          self._loss_vertex = tf.reduce_mean(tf.reduce_sum(tf.reshape(self._loss_vertex,[-1, int(entry_size / self._dims[-1])]),axis=1))

      with tf.variable_scope('train'):
        self._loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=net)
        if self._use_weight:
          self._loss = tf.multiply(weight,self._loss)
        #self._train = tf.train.AdamOptimizer().minimize(self._loss)
        self._loss = tf.reduce_mean(tf.reduce_sum(tf.reshape(self._loss,[-1, int(entry_size / self._dims[-1])]),axis=1))
        
      if self._vertex:
        self._loss += self._loss_vertex

        if self._learning_rate < 0:
          opt = tf.train.AdamOptimizer()
        else:
          opt = tf.train.AdamOptimizer(self._learning_rate)        
        self._zero_gradients = [tv.assign(tf.zeros_like(tv)) for tv in self._accum_vars]
        self._accum_gradients = [self._accum_vars[i].assign_add(gv[0]) for
                                 i, gv in enumerate(opt.compute_gradients(self._loss))]
        self._apply_gradients = opt.apply_gradients(zip(self._accum_vars, tf.trainable_variables()))

      if len(self._dims) == 3:
        tf.summary.image('data_example',tf.image.grayscale_to_rgb(data,'gray_to_rgb'),10)
      tf.summary.scalar('accuracy_all', self._accuracy_allpix)
      tf.summary.scalar('accuracy_nonzero', self._accuracy_nonzero)
      tf.summary.scalar('loss',self._loss)
      tf.summary.scalar('vertex loss', self._loss_vertex)

  def zero_gradients(self, sess):
    return sess.run([self._zero_gradients])

  def accum_gradients(self, sess, input_data, input_label, input_label_vertex=None, input_weight=None):

    feed_dict = self.feed_dict(input_data  = input_data,
                               input_label  = input_label,
                               input_weight = input_weight
                               input_label_vertex = input_label_vertex)
    
    return sess.run([self._accum_gradients, self._loss, self._accuracy_allpix, self._accuracy_nonzero], feed_dict = feed_dict )

  def apply_gradients(self,sess):

    return sess.run( [self._apply_gradients], feed_dict = {})

#  def train(self,sess,input_data,input_label,input_weight=None):
#
#    feed_dict = self.feed_dict(input_data   = input_data,
#                               input_label  = input_label,
#                               input_weight = input_weight)
#
#    ops = [self._train,self._loss,self._accuracy_allpix,self._accuracy_nonzero]
#
#    return sess.run( ops, feed_dict = feed_dict )


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

