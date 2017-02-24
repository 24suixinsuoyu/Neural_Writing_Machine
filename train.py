# -*-  coding:utf-8 -*-
''' model for automatic speech recognition implemented in Tensorflow
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
     
date:2016-12-01
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.dont_write_bytecode = True

import numpy as np
import tensorflow as tf

from preprocess import TextParser
from seq2seq_rnn import Model as Model_rnn
from utils import count_params
from utils import logging
from utils import build_weight

import argparse
import time
import os
from six.moves import cPickle

class Trainer():
    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('--style', default='zhaolei',
                       help='set the type of generating sequence,egs: novel, jay, linxi, tangshi, duilian')

        parser.add_argument('--data_dir', default='/home/pony/github/data/NWM/data/',
                       help='set the data directory which contains new.txt')

        parser.add_argument('--save_dir', default='/home/pony/github/data/NWM/save/',
                       help='set directory to store checkpointed models')

        parser.add_argument('--log_dir', default='/home/pony/github/data/NWM/log/',
                       help='set directory to store checkpointed models')

        parser.add_argument('--rnn_size', type=int, default=128,
                       help='set size of RNN hidden state')

        parser.add_argument('--embedding_size', type=int, default=128,
                       help='set size of word embedding')

        parser.add_argument('--num_layers', type=int, default=1,
                       help='set number of layers in the RNN')

        parser.add_argument('--model', default='seq2seq_rnn',
                       help='set the model')

        parser.add_argument('--rnncell', default='gru',
                       help='set the cell of rnn, eg. rnn, gru, or lstm')

        parser.add_argument('--attention', type=bool, default=False,
                       help='set attention mode or not')

        parser.add_argument('--batch_size', type=int, default=16,
        #parser.add_argument('--batch_size', type=int, default=32,
                       help='set minibatch size')

        parser.add_argument('--seq_length', type=int, default=32,
                       help='set RNN sequence length')

        parser.add_argument('--num_epochs', type=int, default=100000,
                       help='set number of epochs')

        parser.add_argument('--save_every', type=int, default=400,
        #parser.add_argument('--save_every', type=int, default=50,
                       help='set save frequency while training')

        parser.add_argument('--grad_clip', type=float, default=5,
                       help='set clip gradients when back propagation')

        parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='set learning rate')

        parser.add_argument('--decay_rate', type=float, default=1.0,
                       help='set decay rate for rmsprop')                       

        parser.add_argument('--keep', type=bool, default=False,
		       help='init from trained model')

	## pretrained has bug now, so don't use it
	parser.add_argument('--pretrained', type=bool, default=False,
		       help='init from pre-trained model')

        args = parser.parse_args()
        self.train(args)

    def train(self,args):
	''' import data, train model, save model
	'''
	args.data_dir = args.data_dir+args.style+'/'
	args.save_dir = args.save_dir+args.style+'/'
	if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
	print(args)
	if args.attention is True:
	    print('attention mode')
        text_parser = TextParser(args)
        args.vocab_size = text_parser.vocab_size

	if args.pretrained is True:
	    raise ValueError('pretrained has bug now, so don"t set it to be True now!!!')
	    if args.keep is False:
		raise ValueError('when pre-trained is True, keep must be true!')
	    print("pretrained and keep mode...")
	    print("restoring pretrained model file")
            ckpt = tf.train.get_checkpoint_state("/home/pony/github/jaylyrics_generation_tensorflow/data/pre-trained/")
	    if os.path.exists(os.path.join("./data/pre-trained/",'config.pkl')) and \
		os.path.exists(os.path.join("./data/pre-trained/",'words_vocab.pkl')) and \
		ckpt and ckpt.model_checkpoint_path:
                with open(os.path.join("./data/pre-trained/", 'config.pkl'), 'rb') as f:
                    saved_model_args = cPickle.load(f)
                with open(os.path.join("./data/pre-trained/", 'words_vocab.pkl'), 'rb') as f:
                    saved_words, saved_vocab = cPickle.load(f)
	    else:
		raise ValueError('configuration doesn"t exist!')
	else:
	    ckpt = tf.train.get_checkpoint_state(args.save_dir)
	    
        if args.keep is True and args.pretrained is False:
            # check if all necessary files exist 
	    if os.path.exists(os.path.join(args.save_dir,'config.pkl')) and \
		os.path.exists(os.path.join(args.save_dir,'words_vocab.pkl')) and \
		ckpt and ckpt.model_checkpoint_path:
                with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
                    saved_model_args = cPickle.load(f)
                with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'rb') as f:
                    saved_words, saved_vocab = cPickle.load(f)
	    else:
		raise ValueError('configuration doesn"t exist!')

	if args.model == 'seq2seq_rnn':
            model = Model_rnn(args)
	else:
	    # TO ADD OTHER MODEL
	    pass
	trainable_num_params = count_params(model,mode='trainable')
	all_num_params = count_params(model,mode='all')
	args.num_trainable_params = trainable_num_params
	args.num_all_params = all_num_params
        with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
            cPickle.dump(args, f)
        with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'wb') as f:
            cPickle.dump((text_parser.vocab_dict, text_parser.vocab_list), f)

        with tf.Session() as sess:
            if args.keep is True:
	        print('Restoring')
                model.saver.restore(sess, ckpt.model_checkpoint_path)
	    else:
		print('Initializing')
    	        sess.run(model.initial_op)

            sess.run(tf.assign(model.lr, args.learning_rate))
            for e in range(args.num_epochs):
                start = time.time()
		model.initial_state = tf.convert_to_tensor(model.initial_state) 
                state = model.initial_state.eval()
		total_loss = []
                for b in range(text_parser.num_batches):
                    x, y = text_parser.next_batch()
		    if args.attention is True:
		        attention_states = sess.run(tf.truncated_normal([args.batch_size,
						    model.attn_length, model.attn_size],
						    stddev=0.1,dtype=tf.float32))

	                feed = {model.input_data: x, model.targets: y, 
				model.initial_state: state, 
				model.attention_states:attention_states}

	    	    else:
                        feed = {model.input_data: x, 
				model.targets: y, 
				model.initial_state: state}

                    train_loss, state, _ = sess.run([model.cost, 
						     model.final_state, 
						     model.train_op], 
						     feed)
		    total_loss.append(train_loss)
                    print("{}/{} (epoch {}), train_loss = {:.3f}" \
                                .format(e * text_parser.num_batches + b, \
                                args.num_epochs * text_parser.num_batches, \
                                e, train_loss))
                    if (e*text_parser.num_batches+b)%args.save_every==0: 
                        checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                        model.saver.save(sess, checkpoint_path, global_step = e)
                        print("model has been saved in:"+str(checkpoint_path))
                end = time.time()
		delta_time = end - start
		ave_loss = np.array(total_loss).mean()
		logging(model,ave_loss,e,delta_time,mode='train')
		if ave_loss < 0.1:
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    model.saver.save(sess, checkpoint_path, global_step = e)
                    print("model has been saved in:"+str(checkpoint_path))
		    break

if __name__ == '__main__':
    trainer = Trainer()
