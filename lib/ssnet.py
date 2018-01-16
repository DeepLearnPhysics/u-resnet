from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

class ssnet_base(object):

  def __init__(self, dims, num_class):
    self._dims = np.array(dims,np.int32)
    self._num_class = int(num_class)
    if not len(self._dims) in [3,4]:
      print('Error: len(dims) =',len(self._dims),'but only 3 (H,W,C) or 4 (H,W,D,C) supported!')
      raise NotImplementedError

  def _build(self,input_tensor):
    raise NotImplementedError

  def construct(self,freeze=(False,False,False), use_weight=True, learning_rate=None, predict_vertex=False):

    self._freeze_base, self._freeze_class, self._freeze_vertex = freeze

    self._use_weight = bool(use_weight)
    self._learning_rate = learning_rate
    self._predict_vertex = bool(predict_vertex)

    entry_size = np.prod(self._dims)

    with tf.variable_scope('input_prep'):
      self._input_data          = tf.placeholder(tf.float32, [None, entry_size], name='input_data'          )
      self._input_class_weight  = tf.placeholder(tf.float32, [None, entry_size], name='input_class_weight'  )
      self._input_class_label   = tf.placeholder(tf.float32, [None, entry_size], name='input_class_label'   )
      self._input_vertex_label  = tf.placeholder(tf.float32, [None, entry_size], name='input_vertex_label'  )
      self._input_vertex_weight = tf.placeholder(tf.float32, [None, entry_size], name='input_vertex_weight' )
      
      shape_dim = np.insert(self._dims, 0, -1)

      data          = tf.reshape(self._input_data,          shape_dim,      name='data_reshape'          )
      class_label   = tf.reshape(self._input_class_label,   shape_dim[:-1], name='class_label_reshape'   )
      class_weight  = tf.reshape(self._input_class_weight,  shape_dim[:-1], name='class_weight_reshape'  )
      vertex_label  = tf.reshape(self._input_vertex_label,  shape_dim,      name='vertex_label_reshape'  )
      vertex_weight = tf.reshape(self._input_vertex_weight, shape_dim[:-1], name='vertex_weight_reshape' )

      class_label = tf.cast(class_label,tf.int64)
      nonzero_idx = tf.where(tf.reshape(data, shape_dim[:-1]) > tf.to_float(0.))


    # figure out 2d or 3d
    self._fn_conv = slim.conv2d
    self._fn_conv_transpose = slim.conv2d_transpose
    if len(data.shape) == 5:
      self._fn_conv = slim.conv3d
      self._fn_conv_transpose = slim.conv3d_transpose

    #
    # Call (child-implemented) build function to construct the base network architecture
    #
    net = self._build(input_tensor=data)

    # Next construct segmentation (required) and vertex (optional)
    head_vertex, head_class = (None,None)

    #
    # softmax classification
    #
    # define net
    head_class = self._fn_conv(inputs      = net,
                               num_outputs = self._num_class,
                               padding     = 'same',
                               kernel_size = 3,
                               stride      = 1,
                               trainable   = not self._freeze_class,
                               normalizer_fn = None,
                               activation_fn = None,
                               scope       = 'segmentation')
    if self._debug: print(head_class.shape, 'after segmentation layer')
    # define metrics
    self._class_accuracy_allpix   = None
    self._class_accuracy_nonzero  = None
    with tf.variable_scope('class_metrics'):
      self._class_accuracy_allpix = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(head_class,len(self._dims)), class_label),tf.float32))
      nonzero_class_label = tf.gather_nd(class_label,nonzero_idx)
      nonzero_class_pred  = tf.gather_nd(tf.argmax(head_class,len(self._dims)),nonzero_idx)
      self._class_accuracy_nonzero = tf.reduce_mean(tf.cast(tf.equal(nonzero_class_label,nonzero_class_pred),tf.float32))
      self._class_prediction = tf.nn.softmax(logits=head_class)
      
    #
    # sigmoid regression
    #
    if self._predict_vertex:
      # define net
      head_vertex = self._fn_conv(inputs      = net,
                                  num_outputs = 16, 
                                  padding     = 'same',
                                  kernel_size = 3,
                                  stride      = 1,
                                  trainable   = not self._freeze_vertex,
                                  normalizer_fn = None,
                                  activation_fn = tf.nn.relu,
                                  scope       = 'vertex_conv0')
      if self._debug: print(head_vertex.shape, 'after vertex_conv0 layer')
      
      head_vertex = self._fn_conv(inputs      = head_vertex,
                                  num_outputs = 8,
                                  padding     = 'same',
                                  kernel_size = 3,
                                  stride      = 1,
                                  trainable   = not self._freeze_vertex,
                                  normalizer_fn = None,
                                  activation_fn = tf.nn.relu,
                                  scope       = 'vertex_conv1')
      if self._debug: print(head_vertex.shape, 'after vertex_conv1 layer')
      
      head_vertex = self._fn_conv(inputs = head_vertex,
                                  num_outputs = 1,
                                  padding     = 'same',
                                  kernel_size = 3,
                                  stride      = 1,
                                  trainable   = not self._freeze_vertex,
                                  normalizer_fn = None,
                                  activation_fn = None,
                                  scope       = 'vertex')
      if self._debug: print(head_vertex.shape, 'after vertex layer')
      # create accuracy metrics
      self._vertex_accuracy_allpix  = None
      self._vertex_accuracy_nonzero = None
      with tf.variable_scope('vertex_metrics'):
        self._vertex_prediction = tf.nn.sigmoid(x=head_vertex)
        # evaluate sigmoid of label
        self._label_prediction = tf.nn.sigmoid(x=vertex_label)
        # head sigmoid monitoring
        self._max_sigmoid_head = [tf.reduce_max(self._vertex_prediction)]
        self._min_sigmoid_head = [tf.reduce_min(self._vertex_prediction)]
        self._mean_sigmoid_head = [tf.reduce_mean(self._vertex_prediction)]
        # label sigmoid monitoring
        self._max_sigmoid_label = [tf.reduce_max(vertex_label)]
        self._min_sigmoid_label = [tf.reduce_min(vertex_label)]
        self._mean_sigmoid_label = [tf.reduce_mean(vertex_label)]
        # head sigmoids where label prob > 0.5
        #right_idx = tf.to_int64(self._label_prediction > tf.to_float(0.5))
        right_idx = tf.where(tf.reshape(self._label_prediction, shape_dim[:-1])>tf.to_float(0.5))
        right_pred = tf.gather_nd(self._vertex_prediction, right_idx)
        self._max_right = [tf.reduce_max(right_pred)]
        self._min_right = [tf.reduce_min(right_pred)]
        self._mean_right = [tf.reduce_mean(right_pred)]

        vertex_score_threshold    = 0.5
        candidate_vertex_location = tf.to_int64(self._vertex_prediction  > tf.to_float(vertex_score_threshold))
        correct_vertex_location   = tf.to_int64(vertex_label > tf.to_float(vertex_score_threshold))
        self._vertex_accuracy_allpix = tf.reduce_mean(tf.cast(tf.equal(candidate_vertex_location,correct_vertex_location),tf.float32))
        # next provide non-zero mask
        union_idx = tf.maximum(candidate_vertex_location, correct_vertex_location)
        candidate_vertex_location = tf.gather_nd(candidate_vertex_location, tf.where(union_idx>tf.to_int64(0)))
        correct_vertex_location = tf.gather_nd(correct_vertex_location, tf.where(union_idx>tf.to_int64(0)))
        # candidate_vertex_location = tf.gather_nd(candidate_vertex_location, nonzero_idx)
        # correct_vertex_location   = tf.gather_nd(correct_vertex_location,   nonzero_idx)
        if tf.size(tf.where(union_idx>tf.to_int64(0)))==0:
          self._vertex_accuracy_nonzero = 1.
        else:
          self._vertex_accuracy_nonzero = tf.reduce_mean(tf.cast(tf.equal(candidate_vertex_location,correct_vertex_location),tf.float32))           

    # Now all tf.Variable that needs to be updated during training are defined. Collect into a container 
    with tf.variable_scope('accum_grad'):
      self._accum_vars = [tf.Variable(tv.initialized_value(),trainable=False) for tv in tf.trainable_variables()]

    # Define loss & solver
    self._softmax = None
    self._class_loss  = None
    self._vertex_loss = None
    self._total_loss  = None
    # 1st, segmentation
    if not self._freeze_class:
      with tf.variable_scope('class_train'):
        self._class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=class_label, logits=head_class)
        if self._use_weight:
          self._class_loss = tf.multiply(class_weight,self._class_loss)
        self._class_loss = tf.reduce_mean(tf.reduce_sum(tf.reshape(self._class_loss,[-1, int(entry_size / self._dims[-1])]),axis=1))
        self._total_loss = self._class_loss
    # 2nd, vertex (optional)
    if self._predict_vertex and not self._freeze_vertex:
      with tf.variable_scope('vertex_train'):
        self._vertex_loss = tf.squeeze(tf.nn.sigmoid_cross_entropy_with_logits(labels=vertex_label, logits=head_vertex),axis=len(self._dims))
        if self._use_weight:
          self._vertex_loss = tf.multiply(vertex_weight, self._vertex_loss)
        self._vertex_loss = tf.reshape(self._vertex_loss,[-1, int(entry_size / self._dims[-1])])
        self._vertex_loss = tf.reduce_mean(tf.reduce_sum(self._vertex_loss,axis=1))
        
        self._total_loss = self._vertex_loss
        if self._class_loss is not None:
          self._total_loss += self._vertex_loss

    if self._total_loss is None: return;

    if self._learning_rate < 0:
      opt = tf.train.AdamOptimizer()
    else:
      opt = tf.train.AdamOptimizer(self._learning_rate)

    self._zero_gradients = [tv.assign(tf.zeros_like(tv)) for tv in self._accum_vars]
    self._accum_gradients = [self._accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(opt.compute_gradients(self._total_loss))]
    self._apply_gradients = opt.apply_gradients(zip(self._accum_vars, tf.trainable_variables()))

    if len(self._dims) == 3:
      tf.summary.image('data_example',tf.image.grayscale_to_rgb(data,'gray_to_rgb'),10)

    tf.summary.scalar('total loss', self._total_loss)

    tf.summary.scalar('class accuracy', self._class_accuracy_allpix)
    tf.summary.scalar('class accuracy (nonzero)', self._class_accuracy_nonzero)
    if self._class_loss is not None: tf.summary.scalar('class loss', self._class_loss)

    if self._predict_vertex:
      tf.summary.scalar('vertex accuracy', self._vertex_accuracy_allpix)
      tf.summary.scalar('vertex accuracy (nonzero)', self._vertex_accuracy_nonzero)
      if self._vertex_loss is not None: tf.summary.scalar('vertex loss', self._vertex_loss)

  def stats(self,sess, input_data, input_class_weight, input_vertex_label, input_vertex_weight=None):
    feed_dict = self.feed_dict(input_data = input_data,
                               input_class_weight = input_class_weight,
                               input_vertex_label = input_vertex_label,
                               input_vertex_weight = input_vertex_weight)
    return sess.run([self._max_sigmoid_head, self._min_sigmoid_head, self._mean_sigmoid_head,
                     self._max_sigmoid_label, self._min_sigmoid_label, self._mean_sigmoid_label,
                     self._max_right, self._min_right, self._mean_right], feed_dict = feed_dict)
    

  def zero_gradients(self, sess):
    return sess.run([self._zero_gradients])

  def accum_gradients(self, sess, input_data, input_class_label, input_class_weight=None, input_vertex_label=None, input_vertex_weight=None):

    feed_dict = self.feed_dict(input_data          = input_data,
                               input_class_label   = input_class_label,
                               input_class_weight  = input_class_weight,
                               input_vertex_label  = input_vertex_label,
                               input_vertex_weight = input_vertex_weight)
    doc = []
    ops = []
    # overall
    ops += [self._accum_gradients, self._total_loss]
    doc += ['', 'total loss']
    # classification
    ops += [self._class_accuracy_allpix, self._class_accuracy_nonzero]
    doc += ['class acc. all', 'class acc. nonzero']
    if not self._freeze_class:
      ops += [self._class_loss]
      doc += ['class loss']

    if self._predict_vertex:
      ops += [self._vertex_accuracy_allpix, self._vertex_accuracy_nonzero]
      doc += ['vertex acc. all', 'vertex acc. nonzero']
      # vertex finding
      if not self._freeze_vertex:
        ops += [self._vertex_loss]
        doc += ['vertex loss']
    return sess.run(ops, feed_dict = feed_dict ), doc

  def apply_gradients(self,sess):

    return sess.run( [self._apply_gradients], feed_dict = {})

  def inference(self,sess,input_data,input_class_label=None,input_vertex_label=None):
    
    feed_dict = self.feed_dict(input_data         = input_data, 
                               input_class_label  = input_class_label, 
                               input_vertex_label = input_vertex_label)

    doc = ['class pred']
    ops = [self._class_prediction]
    if input_class_label is not None:
      ops += [self._class_accuracy_allpix, self._class_accuracy_nonzero]
      doc += ['class acc. all', 'class acc. nonzero']
    if input_vertex_label is not None:

      ops += [self._vertex_prediction, self._vertex_accuracy_allpix, self._vertex_accuracy_nonzer]
      doc += ['vertex pred', 'vertex acc. all', 'vertex acc. nonzero']

    return sess.run( ops, feed_dict = feed_dict ), doc

  def feed_dict(self,input_data,input_class_label=None,input_class_weight=None,input_vertex_label=None,input_vertex_weight=None):

    if input_class_weight is None and self._use_weight:
      sys.stderr.write('Network configured to use loss pixel-weighting. Cannot run w/ input_class_weight=None...\n')
      raise TypeError

    feed_dict = { self._input_data : input_data }
    if input_class_label is not None:
      feed_dict[ self._input_class_label   ] = input_class_label
    if input_class_weight is not None:
      feed_dict[ self._input_class_weight  ] = input_class_weight
    if input_vertex_label is not None:
      feed_dict[ self._input_vertex_label  ] = input_vertex_label
    if input_vertex_weight is not None:
      feed_dict[ self._input_vertex_weight ] = input_vertex_weight
    return feed_dict

