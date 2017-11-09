from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Basic imports
import os,sys,time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Import more libraries (after configuration is validated)
import tensorflow as tf
from uresnet import uresnet
from larcv import larcv
from larcv.dataloader2 import larcv_threadio
from config import ssnet_config

class ssnet_trainval(object):

  def __init__(self):
    self._cfg = ssnet_config()
    self._filler = None
    self._iteration = -1

  def __del__(self):
    if self._filler:
      self._filler.reset()

  def iteration_from_file_name(self,file_name):
    return int((file_name.split('-'))[-1])

  def override_config(self,file_name):
    self._cfg.override(file_name)
    self._cfg.dump()

  def initialize(self):
    # Instantiate and configure
    if not self._cfg.FILLER_CONFIG:
      'Must provide larcv data filler configuration file!'
      return

    self._filler = larcv_threadio()
    filler_cfg = {'filler_name' : 'ThreadProcessor',
                  'verbosity'   : 0, 
                  'filler_cfg'  : self._cfg.FILLER_CONFIG}
    self._filler.configure(filler_cfg)
    # Start IO thread
    self._filler.start_manager(self._cfg.MINIBATCH_SIZE)
    # Storage ID
    storage_id=0
    # Retrieve image/label dimensions
    self._filler.next()
    dim_data = self._filler.fetch_data(self._cfg.KEYWORD_DATA).dim()

    self._net = uresnet(rows=dim_data[1], 
                        cols=dim_data[2], 
                        num_class=3, 
                        base_num_outputs=self._cfg.BASE_NUM_FILTERS, 
                        debug=False)

    if self._cfg.TRAIN:
      self._net.construct(trainable=self._cfg.TRAIN,use_weight=True)
    else:
      self._net.construct(trainable=self._cfg.TRAIN,use_weight=False)

    self._iteration = 0

  def run(self,sess):
    # Configure global process (session, summary, etc.)
    # Create a bandle of summary
    merged_summary=tf.summary.merge_all()
    # Create a session
    #sess = tf.InteractiveSession()
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    writer = None
    if self._cfg.LOGDIR:
      # Create a summary writer handle
      writer=tf.summary.FileWriter(self._cfg.LOGDIR)
      writer.add_graph(sess.graph)
    saver = None
    if self._cfg.SAVE_FILE:
      # Create weights saver
      saver = tf.train.Saver()
      
    # Override variables if wished
    if self._cfg.LOAD_FILE:
      vlist=[]
      self._iteration = self.iteration_from_file_name(self._cfg.LOAD_FILE)
      parent_vlist = []
      if self._cfg.TRAIN: 
        parent_vlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      else:
        parent_vlist = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
      for v in parent_vlist:
        if v.name in self._cfg.AVOID_LOAD_PARAMS:
          print('\033[91mSkipping\033[00m loading variable',v.name,'from input weight...')
          continue
        print('\033[95mLoading\033[00m variable',v.name,'from',self._cfg.LOAD_FILE)
        vlist.append(v)
      reader=tf.train.Saver(var_list=vlist)
      reader.restore(sess,self._cfg.LOAD_FILE)
    
    for i in xrange(self._cfg.ITERATIONS):
      if self._cfg.TRAIN and self._iteration >= self._cfg.ITERATIONS:
        print('Finished training (iteration %d)' % self._iteration)
        break
      self._net.zero_gradients(sess = sess)
      batch_metrics = np.zeros((self._cfg.NUM_MINIBATCHES,3))
      for j in xrange(self._cfg.NUM_MINIBATCHES):
        minibatch_data   = self._filler.fetch_data(self._cfg.KEYWORD_DATA).data()
        minibatch_label  = self._filler.fetch_data(self._cfg.KEYWORD_LABEL).data()
        minibatch_weight = None
        if self._cfg.TRAIN:
          minibatch_weight = self._filler.fetch_data(self._cfg.KEYWORD_WEIGHT).data()
          self._filler.next()
          # perform per-event normalization                                                                                                  
          if self._cfg.NORMALIZE_WEIGHTS:
            minibatch_weight /= np.mean(minibatch_weight,axis=1, keepdims=True)

        _, loss, acc_all, acc_nonzero = self._net.accum_gradients(sess = sess,
                                                                  input_image = minibatch_data,
                                                                  input_label = minibatch_label,
                                                                  input_weight = minibatch_weight)
        batch_metrics[j,0] = loss
        batch_metrics[j,1] = acc_all
        batch_metrics[j,1] = acc_nonzero

      self._net.apply_gradients(sess = sess)
      self._iteration += 1
      msg = 'Training in progress @ step %d loss %g accuracy %g / %g           \r'
      msg = msg % (self._iteration,np.mean(batch_metrics,axis=0)[0], np.mean(batch_metrics,axis=0)[1], np.mean(batch_metrics,axis=0)[2])
      sys.stdout.write(msg)
      sys.stdout.flush()

      # Save log
      if self._cfg.TRAIN and self._cfg.SUMMARY_STEPS and ((self._iteration+1)%self._cfg.SUMMARY_STEPS) == 0:
        # Run summary
        feed_dict = self._net.feed_dict(input_image  = minibatch_data,
                                        input_label  = minibatch_label,
                                        input_weight = minibatch_weight)
        writer.add_summary(sess.run(merged_summary,feed_dict=feed_dict),self._iteration)
  
      # Save snapshot
      if self._cfg.TRAIN and self._cfg.CHECKPOINT_STEPS and ((self._iteration+1)%self._cfg.CHECKPOINT_STEPS) == 0:
        # Save snapshot
        ssf_path = saver.save(sess,self._cfg.SAVE_FILE,global_step=self._iteration)
        print()
        print('saved @',ssf_path)


