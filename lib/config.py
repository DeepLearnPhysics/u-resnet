from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

class ssnet_config:

    NUM_CLASS         = 3
    BASE_NUM_FILTERS  = 16
    FILLER_CONFIG     = 'config/train_io.cfg'
    DRAINER_CONFIG    = ''
    LOGDIR            = 'ssnet_train_log'
    SAVE_FILE         = 'ssnet_checkpoint/uresnet'
    LOAD_FILE         = ''
    AVOID_LOAD_PARAMS = []
    LEARNING_RATE     = -1
    BATCH_SIZE        = 10
    MINIBATCH_SIZE    = 10
    NUM_MINIBATCHES   = 5
    ITERATIONS        = 100000
    DUMP_IMAGE        = False
    #TRAIN             = True
    FREEZE_BASE       = False
    FREEZE_CLASS      = False
    FREEZE_VERTEX     = True
    USE_WEIGHTS       = True
    CHECKPOINT_STEPS  = 200
    SUMMARY_STEPS     = 20
    KEYWORD_DATA      = 'data'
    KEYWORD_CLASS_LABEL  = 'class_label'
    KEYWORD_CLASS_WEIGHT = 'class_weight'
    KEYWORD_PRIMARY  = ''
    KEYWORD_SECONDARY = ''
    KEYWORD_VERTEX_WEIGHT = ''
    PREDICT_VERTEX       = False

    def __init__(self):
        pass

    def override(self,file_name):

        keys=[s for s in self.__class__.__dict__.keys() if s == s.upper()]
        if not os.path.isfile(file_name):
            print('Config file not found',file_name)
            raise IOError
        for line in open(file_name,'r').read().split('\n'):
            valid_line = str(line)
            if line.find('#')>=0:
                valid_line = line[0:line.find('#')]
            if valid_line.startswith(' '):
                valid_line = valid_line.strip(' ')
            words = valid_line.split()
            if len(words) == 0: continue
            if not len(words) == 2:
                print('Ignoring a line:',line)
                continue
            if not words[0] in keys:
                print('Ignoring a parameter in file:',words[0])
            valid=False

            exec('valid = (type(self.%s) == type(%s))' % (words[0],words[1]))
            if line.split()[0] != 'LEARNING_RATE' and not valid:
                print('Incompatible type: %s' % line)
                raise TypeError
                
            exec('self.%s = %s' % (words[0],words[1]))

    def dump(self):
        keys=[s for s in self.__class__.__dict__.keys() if s == s.upper()]
        for key in keys:
            val=None
            exec('val=str(self.%s)' % key)
            msg = key
            while len(msg)<20:
                msg += '.'
            print('%s %s' % (msg,val))

if __name__ == '__main__':
    k=ssnet_config()
    k.dump()
    import sys
    if len(sys.argv)>1:
        k.override(sys.argv[1])
        print('\n')
        k.dump()
