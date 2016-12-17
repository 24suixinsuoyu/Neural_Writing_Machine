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


import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
from decoder import attention_decoder

import numpy as np
import datetime
from utils import build_weight
from utils import random_pick

class Model():
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.rnncell == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.rnncell == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.rnncell == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("rnncell type not supported: {}".format(args.rnncell))

        cell = cell_fn(args.rnn_size)
        self.cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = self.cell.zero_state(args.batch_size, tf.float32)
	self.attn_length = 5
	self.attn_size = 32
	self.attention_states = tf.placeholder(tf.float32,[args.batch_size, self.attn_length, self.attn_size]) 
        with tf.variable_scope('rnnlm'):
            softmax_w = build_weight([args.rnn_size, args.vocab_size],name='soft_w')
            softmax_b = build_weight([args.vocab_size],name='soft_b')
            word_embedding = build_weight([args.vocab_size, args.embedding_size],name='word_embedding')
            inputs_list = tf.split(1, args.seq_length, tf.nn.embedding_lookup(word_embedding, self.input_data))
            inputs_list = [tf.squeeze(input_, [1]) for input_ in inputs_list]
        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

	if not args.attention:
            outputs, last_state = seq2seq.rnn_decoder(inputs_list, self.initial_state, self.cell, loop_function=loop if infer else None, scope='rnnlm')
	else:
            outputs, last_state = attention_decoder(inputs_list, self.initial_state, self.attention_states, self.cell, loop_function=loop if infer else None, scope='rnnlm')

        self.final_state = last_state
        output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])],
                args.vocab_size)
	# average loss for each word of each timestep
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.lr = tf.Variable(0.0, trainable=False)
	self.var_trainable_op = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, self.var_trainable_op),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, self.var_trainable_op))
	self.initial_op = tf.global_variables_initializer()
	self.logfile = args.log_dir+str(datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')+'.txt').replace(' ','').replace('/','')
	self.var_op = tf.global_variables()
	self.saver = tf.train.Saver(self.var_op,max_to_keep=5,keep_checkpoint_every_n_hours=1)

    def sample(self, sess, words, vocab, num=200, start=u'从前', sampling_type=1):

	state = sess.run(self.cell.zero_state(1, tf.float32))
        attention_states = sess.run(tf.truncated_normal([1, self.attn_length, self.attn_size],stddev=0.1,dtype=tf.float32))
        for word in start:
            x = np.zeros((1, 1))
            x[0, 0] = words[word]
	    if self.args.attention is True:
	        feed = {self.input_data: x, self.initial_state:state, self.attention_states:attention_states}
                [probs, state, attention_states] = sess.run([self.probs, self.final_state, self.attention_states], feed)
	    else:
                feed = {self.input_data: x, self.initial_state:state}
                [probs, state] = sess.run([self.probs, self.final_state], feed)
	    
        ret = start
        word = start[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = words[word]
	    if self.args.attention is True:
	        feed = {self.input_data: x, self.initial_state:state, self.attention_states:attention_states}
                [probs, state, attention_states] = sess.run([self.probs, self.final_state, self.attention_states], feed)
	    else:
                feed = {self.input_data: x, self.initial_state:state}
                [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]
	    sample = random_pick(p,word,sampling_type)
            pred = vocab[sample]
            ret += pred
            word = pred
        return ret
