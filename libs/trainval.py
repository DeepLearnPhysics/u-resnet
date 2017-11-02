# Basic imports
import os,sys,time
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
# Import more libraries (after configuration is validated)
import tensorflow as tf
from libs.uresnet import uresnet
from larcv import larcv
from larcv.dataloader2 import larcv_threadio
from config import ssnet_config

class train_ssnet(object):

  def __init__(self):
    self._cfg = ssnet_config()
    self._filler = None

  def __del__(self):
    if self._filler:
      self._filler.reset()

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
    self._filler.start_manager(self._cfg.BATCH_SIZE)
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
      for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        if v.name in self._cfg.AVOID_LOAD_PARAMS:
          print '\033[91mSkipping\033[00m loading variable',v.name,'from input weight...'
          continue
        print '\033[95mLoading\033[00m variable',v.name,'from',self._cfg.LOAD_FILE
        vlist.append(v)
        reader=tf.train.Saver(var_list=vlist)
        reader.restore(sess,self._cfg.LOAD_FILE)
    
    # Run iterations
    for i in range(self._cfg.ITERATIONS):
  
      # Receive data (this will hang if IO thread is still running = this will wait for thread to finish & receive data)
      batch_data   = self._filler.fetch_data(self._cfg.KEYWORD_DATA).data()
      batch_label  = self._filler.fetch_data(self._cfg.KEYWORD_LABEL).data()
      batch_weight = None
      # Start IO thread for the next batch while we train the network
      if self._cfg.TRAIN:
        batch_weight = self._filler.fetch_data(self._cfg.KEYWORD_WEIGHT).data()
        self._filler.next()
        # perform per-event normalization
        if self._cfg.NORMALIZE_WEIGHTS:
          batch_weight /= np.mean(batch_weight,axis=1).reshape([batch_weight.shape[0],1])
    
        _,loss,acc_all,acc_nonzero = self._net.train(sess         = sess, 
                                                     input_image  = batch_data,
                                                     input_label  = batch_label,
                                                     input_weight = batch_weight)
        sys.stdout.write('Training in progress @ step %d loss %g accuracy %g / %g           \r' % (i,loss,acc_all,acc_nonzero))
      else:
        self._filler.next()
        softmax,acc_all,acc_nonzero = self._net.train(sess        = sess,
                                                      input_image = batch_data,
                                                      input_label = batch_label)
        sys.stdout.write('Training in progress @ step %d accuracy %g / %g                   \r' % (i,acc_all,acc_nonzero))

      sys.stdout.flush()

      # Save log every 20 steps
      if self._cfg.TRAIN and self._cfg.SUMMARY_STEPS and ((i+1)%self._cfg.SUMMARY_STEPS) == 0:
        # Run summary
        feed_dict = self._net.feed_dict(input_image  = batch_data,
                                        input_label  = batch_label,
                                        input_weight = batch_weight)
        writer.add_summary(sess.run(merged_summary,feed_dict=feed_dict),i)
  
      # If configured to save summary + snapshot, do so here.
      if self._cfg.TRAIN and self._cfg.CHECKPOINT_STEPS and ((i+1)%self._cfg.CHECKPOINT_STEPS) == 0:
        # Save snapshot
        ssf_path = saver.save(sess,self._cfg.SAVE_FILE,global_step=i)
        print
        print 'saved @',ssf_path


