# -*- coding:utf-8 -*-
''' lyrics generation
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

import os
import numpy as np
import re
import codecs
import collections
from six.moves import cPickle

class TextParser():
    def __init__(self, args):
	''' Initialize the basic directory, batch_size and sequence length
	'''
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.seq_length = args.seq_length
        self.input_file = os.path.join(self.data_dir, "new.txt")
	if args.pretrained is False:
            self.vocab_file = os.path.join(self.data_dir, "vocab.pkl")
            self.context_file = os.path.join(self.data_dir, "context.npy")
	else:
	    self.vocab_file = os.path.join("./data/pre-trained/", "vocab.pkl")
            self.context_file = os.path.join("./data/pre-trained/", "context.npy")

	if args.keep is False:
            print("keep:False, building dataset...")
            self.build_dataset()
	else:        
	    print("keep:True, loading dataset...")
            self.load_dataset()
	print("initialize mini-batch dataset...")
        self.init_batches()

    def build_dataset(self):
	''' parse all sentences to build a vocabulary 
	    dictionary and vocabulary list
	'''
        with codecs.open(self.input_file, "r",encoding='utf-8') as f:
            data = f.read()
	print data[-100:]
        wordCounts = collections.Counter(data)
        self.vocab_list = [x[0] for x in wordCounts.most_common()]
        self.vocab_size = len(self.vocab_list)
        self.vocab_dict = {x: i for i, x in enumerate(self.vocab_list)}
        with codecs.open(self.vocab_file, 'wb',encoding='utf-8') as f:
            cPickle.dump(self.vocab_list, f)
        self.context = np.array(list(map(self.vocab_dict.get, data)))
        np.save(self.context_file, self.context)
	self.num_batches = int(self.context.size / (self.batch_size * self.seq_length))


    def load_dataset(self):
	''' if vocabulary has existed, we just load it
	'''
        with open(self.vocab_file, 'rb') as f:
            self.vocab_list = cPickle.load(f)
        self.vocab_size = len(self.vocab_list)
	print('size of dictionary:'+str(self.vocab_size))
        self.vocab_dict = dict(zip(self.vocab_list, range(len(self.vocab_list))))
        with codecs.open(self.input_file, "r",encoding='utf-8') as f:
            data = f.read()
	print data[-100:]

	self.context = np.array(list(map(self.vocab_dict.get, data)))
        np.save(self.context_file, self.context)
        self.num_batches = int(self.context.size / (self.batch_size * self.seq_length))

    def init_batches(self):
	''' Split the dataset into mini-batches, 
	    xdata and ydata should be the same length here
	    we add a space before the context to make sense.
	'''
        self.num_batches = int(self.context.size / (self.batch_size * self.seq_length))
        self.context = self.context[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.context
        ydata = np.copy(self.context)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.pointer = 0

    def next_batch(self):
	''' pointer for outputing mini-batches when training
	'''
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
	if self.pointer == self.num_batches:
	    self.pointer = 0
        return x, y

# test code
if __name__ == '__main__':
    t = TextParser()
    t.build_dataset()
