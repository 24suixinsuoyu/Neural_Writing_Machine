# -*- coding:utf-8 -*-
''' Sequence generation implemented in Tensorflow
author:

      iiiiiiiiiiii            iiiiiiiiiiii         !!!!!!!             !!!!!!    
      #        ###            #        ###           ###        I#        #:     
      #      ###              #      I##;             ##;       ##       ##      
            ###                     ###               !##      ####      #       
           ###                     ###                 ###    ## ###    #'       
         !##;                    `##%                   ##;  ##   ###  ##        
        ###                     ###                     $## `#     ##  #         
       ###        #            ###        #              ####      ####;         
     `###        -#           ###        `#               ###       ###          
     ##############          ##############               `#         #     
     
date:2016-12-07
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.dont_write_bytecode = True

import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle
import codecs

from preprocess import TextParser
from seq2seq_rnn import Model

def main():
    # get arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--style', default='zhaolei',
                       help='set the type of generating sequence,egs: novel, jay, linxi, tangshi, duilian')

    parser.add_argument('--save_dir', type=str, 
			 default='/home/pony/github/NeuralWritingMachine/save/',
                         help='model directory to store checkpointed models')

    parser.add_argument('--n', type=int, default=200,
                       help='number of words to sample')
    parser.add_argument('--start', default=u'我的姑娘你在哪儿',
                       help='prime text')
    parser.add_argument('--sample', type=str, default='weighted',
                       help='three choices:argmax,weighted,combined')

    args = parser.parse_args()
    sample(args)

def sample(args):
    # import configuration
    args.save_dir = args.save_dir + args.style + '/'
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab = cPickle.load(f)
    # import the trained model
    model = Model(saved_args, True)
    with tf.Session() as sess:
	# initialize the model
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
	
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
	    #args.start = args.start.decode('gbk')
            literature = model.sample(sess, words, vocab, args.n, args.start, args.sample)
    with codecs.open('/home/pony/github/NeuralWritingMachine/result/sequence.txt','a','utf-8') as f:
        f.write(args.style+'\tnum:'+str(args.n)+'\n')
        f.write(literature+'\n\n')
    print(literature.encode('utf-8'))

if __name__ == '__main__':
    main()
