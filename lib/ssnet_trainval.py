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
    self._cfg        = ssnet_config()
    self._input_main = None
    self._input_test = None
    self._output     = None
    self._iteration  = -1

  def __del__(self):
    if self._input_main:
      self._input_main.reset()
    if self._output:
      self._output.finalize()

  def _report(self,step,metrics,descr):
    msg = 'Training in progress @ step %-4d ... ' % step
    for i,desc in enumerate(descr):
      if not desc: continue
      msg += '%s=%6.6f   ' % (desc,metrics[i])
    msg += '\n'
    sys.stdout.write(msg)
    sys.stdout.flush()

  def iteration_from_file_name(self,file_name):
    return int((file_name.split('-'))[-1])

  def override_config(self,file_name):
    self._cfg.override(file_name)
    self._cfg.dump()

  def initialize(self):
    # Instantiate and configure
    if not self._cfg.MAIN_INPUT_CONFIG:
      print('Must provide larcv data filler configuration file!')
      return

    #
    # Data IO configuration
    #
    # Main input stream
    self._input_main = larcv_threadio()
    filler_cfg = {'filler_name' : 'ThreadProcessor',
                  'verbosity'   : 0, 
                  'filler_cfg'  : self._cfg.MAIN_INPUT_CONFIG}
    self._input_main.configure(filler_cfg)
    self._input_main.start_manager(self._cfg.MINIBATCH_SIZE)

    # Test input stream (optional)
    if self._cfg.TEST_INPUT_CONFIG:
      self._input_test = larcv_threadio()
      filler_cfg = {'filler_name' : 'ThreadProcessor',
                    'verbosity'   : 0,
                    'filler_cfg'  : self._cfg.TEST_INPUT_CONFIG}
      self._input_test.configure(filler_cfg)
      self._input_test.start_manager(self._cfg.TEST_BATCH_SIZE)

    # Output stream (optional)
    if self._cfg.ANA_OUTPUT_CONFIG:
      self._output = larcv.IOManager(self._cfg.ANA_OUTPUT_CONFIG)
      self._output.initialize()

    #
    # Network construction
    #
    # Retrieve image/label dimensions
    self._input_main.next(store_entries   = (not self._cfg.TRAIN),
                      store_event_ids = (not self._cfg.TRAIN))
    dim_data = self._input_main.fetch_data(self._cfg.KEYWORD_DATA).dim()
    dims = []
    self._net = uresnet(dims=dim_data[1:],
                        num_class=3, 
                        base_num_outputs=self._cfg.BASE_NUM_FILTERS, 
                        debug=False)

    if self._cfg.TRAIN:
      self._net.construct(trainable=self._cfg.TRAIN,
                          use_weight=self._cfg.USE_WEIGHTS,
                          learning_rate=self._cfg.LEARNING_RATE)
    else:
      self._net.construct(trainable=self._cfg.TRAIN,
                          use_weight=self._cfg.USE_WEIGHTS)
      
    #
    # Network variable initialization
    #
    # Set random seed for reproducibility
    tf.set_random_seed(self._cfg.TF_RANDOM_SEED)
    # Configure global process (session, summary, etc.)
    # Create a bandle of summary
    merged_summary=tf.summary.merge_all()
    # Initialize variables
    self._sess = tf.InteractiveSession()
    self._sess.run(tf.global_variables_initializer())
    self._writer = None
    if self._cfg.LOGDIR:
      # Create a summary writer handle
      self._writer=tf.summary.FileWriter(self._cfg.LOGDIR)
      self._writer.add_graph(self._sess.graph)
    saver = None
    if self._cfg.SAVE_FILE:
      # Create weights saver
      self._saver = tf.train.Saver()
      
    # Override variables if wished
    if self._cfg.LOAD_FILE:
      vlist=[]
      self._iteration = self.iteration_from_file_name(self._cfg.LOAD_FILE)
      parent_vlist = []
      parent_vlist = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
      for v in parent_vlist:
        if v.name in self._cfg.AVOID_LOAD_PARAMS:
          print('\033[91mSkipping\033[00m loading variable',v.name,'from input weight...')
          continue
        print('\033[95mLoading\033[00m variable',v.name,'from',self._cfg.LOAD_FILE)
        vlist.append(v)
      reader=tf.train.Saver(var_list=vlist)
      reader.restore(self._sess,self._cfg.LOAD_FILE)
    
    self._iteration = 0
    self._batch_metrics = None
    self._descr_metrics = None

  def train_step(self):

    # Nullify the gradients
    self._net.zero_gradients(self._sess)

    # Loop over minibatches
    for j in xrange(self._cfg.NUM_MINIBATCHES):
      minibatch_data   = self._input_main.fetch_data(self._cfg.KEYWORD_DATA).data()
      minibatch_label  = self._input_main.fetch_data(self._cfg.KEYWORD_LABEL).data()
      if self._cfg.DEBUG:
        print("min/max/mean for data ({}/{}/{}) and label ({}/{}/{})".format( minibatch_data.min(),
                                                                              minibatch_data.max(),
                                                                              minibatch_data.mean(),
                                                                              minibatch_label.min(),
                                                                              minibatch_label.max(),
                                                                              minibatch_label.mean() )
            )

      minibatch_weight = None
      if self._cfg.USE_WEIGHTS:
        minibatch_weight = self._input_main.fetch_data(self._cfg.KEYWORD_WEIGHT).data()
        # perform per-event normalization
        minibatch_weight /= (np.sum(minibatch_weight,axis=1).reshape([minibatch_weight.shape[0],1]))

      # train 
      res,doc = self._net.accum_gradients(sess         = self._sess,
                                          input_data   = minibatch_data,
                                          input_label  = minibatch_label,
                                          input_weight = minibatch_weight)

      if self._batch_metrics is None:
        self._batch_metrics = np.zeros((self._cfg.NUM_MINIBATCHES,len(res)-1),dtype=np.float32)
        self._descr_metrics = doc[1:]

      self._batch_metrics[j,:] = res[1:]

      self._input_main.next(store_entries   = (not self._cfg.TRAIN),
                            store_event_ids = (not self._cfg.TRAIN))

    # update
    self._net.apply_gradients(self._sess)

    # debug output
    if self._cfg.DEBUG:
      maxval, minval, meanval = self._net.stats(sess = self._sess, 
                                                input_data = minibatch_data,
                                                input_label = minibatch_label,
                                                input_weight = minibatch_weight)
      debug = 'max %g, min %g, mean %g \n'
      debug = debug % (np.squeeze(maxval), np.squeeze(minval), np.squeeze(meanval))
      sys.stdout.write(debug)
      sys.stdout.flush()

    # Report
    if (self._iteration+1) % self._cfg.REPORT_STEPS == 0:
      self._report(self._iteration,np.mean(self._batch_metrics,axis=0),self._descr_metrics)

    # Save log
    if self._cfg.TRAIN and self._cfg.SUMMARY_STEPS and ((self._iteration+1)%self._cfg.SUMMARY_STEPS) == 0:
      # Run summary
      feed_dict = self._net.feed_dict(input_data   = minibatch_data,
                                      input_label  = minibatch_label,
                                      input_weight = minibatch_weight)
      self._writer.add_summary(self._sess.run(merged_summary,feed_dict=feed_dict),self._iteration)
  
    # Save snapshot
    if self._cfg.TRAIN and self._cfg.CHECKPOINT_STEPS and ((self._iteration+1)%self._cfg.CHECKPOINT_STEPS) == 0:
      # Save snapshot
      ssf_path = self._saver.save(sess,self._cfg.SAVE_FILE,global_step=self._iteration)
      print()
      print('saved @',ssf_path)

  def ana_step(self):

    # Receive data (this will hang if IO thread is still running = this will wait for thread to finish & receive data)                                  
    batch_data   = self._input_main.fetch_data(self._cfg.KEYWORD_DATA).data()
    batch_label  = self._input_main.fetch_data(self._cfg.KEYWORD_LABEL).data()
    batch_weight = None
    softmax,acc_all,acc_nonzero = self._net.inference(sess        = self._sess,
                                                      input_data  = batch_data,
                                                      input_label = batch_label)    
    if self._output:
      for entry in xrange(len(softmax)):
        self._output.read_entry(entry)
        data  = np.array(batch_data[entry]).reshape(softmax.shape[1:-1])
        entries   = self._input_main.fetch_entries()
        event_ids = self._input_main.fetch_event_ids()

      for entry in xrange(len(softmax)):
        self._output.read_entry(entries[entry])
        data  = np.array(batch_data[entry]).reshape(softmax.shape[1:-1])
        label = np.array(batch_label[entry]).reshape(softmax.shape[1:-1])

        shower_score, track_score = (None,None)
        data_2d = len(softmax.shape) == 4
        # 3D case
        if data_2d:
          shower_score = softmax[entry,:,:,1]
          track_score  = softmax[entry,:,:,2]
        else:
          shower_score = softmax[entry,:,:,:,1]
          track_score  = softmax[entry,:,:,:,2]
        
        sum_score = shower_score + track_score
        shower_score = shower_score / sum_score
        track_score  = track_score  / sum_score
        
        ssnet_result = (shower_score > track_score).astype(np.float32) + (track_score >= shower_score).astype(np.float32) * 2.0
        nonzero_map  = (data > 1.0).astype(np.int32)
        ssnet_result = (ssnet_result * nonzero_map).astype(np.float32)

        if data_2d:
          larcv_data = self._output.get_data("image2d","data")
          larcv_out  = self._output.get_data("sparse2d","ssnet")
          vs = larcv.as_image2d(ssnet_result)
          sparse3d.set(vs,data.meta())
        else:
          larcv_data = self._output.get_data("sparse3d","data")
          larcv_out  = self._output.get_data("sparse3d","ssnet")
          vs = larcv.as_tensor3d(ssnet_result)
          sparse3d.set(vs,data.meta())
        self._output.save_entry()

    self._input_main.next(store_entries   = (not self._cfg.TRAIN),
                          store_event_ids = (not self._cfg.TRAIN))


  def batch_process(self):

    # Run iterations
    for i in xrange(self._cfg.ITERATIONS):
      if self._cfg.TRAIN and self._iteration >= self._cfg.ITERATIONS:
        print('Finished training (iteration %d)' % self._iteration)
        break

      # Start IO thread for the next batch while we train the network
      if self._cfg.TRAIN:
        self.train_step()
        
      else:
        self.ana_step()

      self._iteration += 1

  def reset(self):
    if self._input_main:
      self._input_main.reset()
      self._input_main = None

    if self._input_test:
      self._input_test.reset()
      self._input_test = None

    if self._output:
      self._output.finalize()
      self._output = None

    
