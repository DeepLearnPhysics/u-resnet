import sys,os
lib_path = os.path.join(os.environ['URESNETDIR'],'lib')
if not lib_path in sys.path:
    print lib_path
    sys.path.insert(0,lib_path)

import ssnet_trainval as api
import tensorflow as tf
import sys
import numpy as np
#
# What does this script do?
# 0) Run u-resnet inference: you need to give a config file in an argument
# 1) Compute class-wise accuracy
# 2) Store results in CSV file
# ... there is a separate example script, ana_csv.py, to analyze CSV file created in step 2)

# instantiate ssnet_trainval and configure
t = api.ssnet_trainval()
for argv in sys.argv:
    if argv.endswith('.cfg'): t.override_config(argv)

# initialize
t.initialize()

# get number of classes
num_class = t.num_class()

# Prepare CSV file
fout=open('ana.csv','w')
fout.write('entry,acc_all,acc_nonzero')
for i in np.arange(num_class):
    fout.write(',npx_class%d'          % i ) # pixel count for this class
    fout.write(',acc_class%d'          % i ) # average accuracy for this class
    fout.write(',mean_softmax_class%d' % i ) # mean softmax probability for this class
    fout.write(',std_softmax_class%d'  % i ) # std softmax probability for this class
fout.write('\n')

#
# run interactive analysis
#
ITERATIONS=5
for iteration in np.arange(ITERATIONS):

    # call "ana_step" API function: this runs network inference and returns results
    res = t.ana_step()
    # The results, "res", is a dictionary (try running ana_example.py to see by yourself)
    # Contents are:
    # 0) res['entries'    ] = 1D numpy array that holds TTree entry numbers analyzed in this batch
    # 1) res['acc_all'    ] = an average accuracy across all pixels over all images in a batch
    # 2) res['acc_nonzero'] = an average accuracy across all non-zero pixels over all images in a batch
    # 3) res['input'      ] = a batch of input images (numpy array), dimension [Batch,Row,Col,Channels]
    # 4) res['label'      ] = a batch of label images (numpy array), dimension [Batch,Row,Col,Channels]
    # 5) res['softmax'    ] = a batch of softmax images (numpy array), dimension [Batch,Row,Col,Channels]

    entries_batch = res['entries']
    softmax_batch = res['softmax']
    image_batch   = res['input'  ]
    label_batch   = res['label'  ]
    
    # Note: Next, we loop over images in this batch and compute analysis variables.
    #       We will re-compute acc_all and acc_nonzero per image, and will not use
    #       the return value in the "res" dictionary because acc_all and acc_nonzero
    #       in the dictionary is an average over images in the batch. We want more details.
    
    # Loop over each entry = image in this batch
    for index in np.arange(len(softmax_batch)):

        entry   = entries_batch[index]
        softmax = softmax_batch[index]
        label   = label_batch[index]

        # Let's squeeze the label: this changes dimension from (row,col,1) to (row,col)
        label = np.squeeze(label)

        # Compute acc_all
        prediction = np.argmax(softmax,axis=-1).astype(np.float32)
        acc_all    = float( (prediction == label).sum() ) / prediction.size

        # Compute acc_nonzero
        nonzero_px = np.where(label>0)
        nonzero_prediction = prediction[nonzero_px]
        nonzero_label      = label[nonzero_px]
        acc_nonzero = float( (nonzero_prediction == nonzero_label ).sum() ) / nonzero_prediction.size

        # Record parameters computed so far in a csv file
        fout.write('%d,%g,%g' % (entry,acc_all,acc_nonzero))

        # Next compute class-wise accuracy, mean/std score value
        for class_label in np.arange(num_class):

            # create a mask to select this class
            class_mask = np.where(label==class_label)
            
            # compute pixel count for this class
            npx = label[class_mask].size

            # if npx is non-zero, compute class accuracy, softmax score mean/std.
            class_acc = -1.
            class_score_mean = -1.
            class_score_std  = -1.
            if npx:
                # compute class accuracy
                class_prediction = prediction[class_mask]
                class_acc = float( (class_prediction == class_label).sum() ) / npx

                # compute softmax score mean value
                class_score = (softmax[:,:,class_label])[class_mask]
                class_score_mean = class_score.mean()
                class_score_std  = class_score.std()

            # Record in a csv file
            fout.write(',%d,%g,%g,%g' % (npx,class_acc,class_score_mean,class_score_std))

        # Insert EOL in csv to signal the end of this image
        fout.write('\n')
        
    print('Finished iteration %d...' % iteration)

# reset before exit to terminate threads
t.reset()
