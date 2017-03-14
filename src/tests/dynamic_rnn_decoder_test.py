#coding:utf-8

'''
tf1.0 support dynamic rnn decoder, which means the
 decoder does not make any assumptions of sequence
 length and batch size of the input;

but it contains training or inference and expects
  the user to create seperate functions for each;

`sequence_length` is needed at training time, i.e., 
when `inputs` is not None, for dynamic unrolling. 
At test time, when `inputs` is None, `sequence_length` 
is not needed

Under inference `inputs` is expected to be `None` 
and the input is inferred solely from the `decoder_fn`

Usage:

dynamic_rnn_decoder(cell, decoder_fn, inputs=None, sequence_length=None,
                        parallel_iterations=None, swap_memory=False,
                        time_major=False, scope=None, name=None):
'''

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.python.framework import dtypes
from tensorflow.contrib import layers
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.contrib.seq2seq.python.ops import attention_decoder_fn
from tensorflow.contrib.seq2seq.python.ops import decoder_fn as decoder_fn_lib
from tensorflow.contrib.seq2seq.python.ops import seq2seq


def _decoder_fn_with_context_state(inner_decoder_fn, name=None):
    """Wraps a given decoder function, adding context state to it.

    Given a valid `inner_decoder_fn`, returns another valid `decoder_fn` which
    first calls `inner_decoder_fn`, then overwrites the context_state, setting
    it to the current time.

    Args:
      inner_decoder_fn: A valid `decoder_fn` of the type passed into
        `dynamic_rnn_decoder`.

    Returns:
      A valid `decoder_fn` to be passed into `dynamic_rnn_decoder`.
    """

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
      with tf.name_scope(
          name, "decoder_fn_with_context_state",
          [time, cell_state, cell_input, cell_output, context_state]):
        done, next_state, next_input, emit_output, next_context_state = (
            inner_decoder_fn(time, cell_state, cell_input, cell_output,
                             context_state))
        next_context_state = time
        return done, next_state, next_input, emit_output, next_context_state
    return decoder_fn

def test_dynamic_rnn_decoder():
    with tf.Session() as sess:
        with tf.variable_scope(
          "root", initializer=tf.constant_initializer(0.5)) as varscope:
	    batch_size = 2
            encoder_embedding_size = 3
            decoder_embedding_size = 4
            encoder_hidden_size = 5
            decoder_hidden_size = encoder_hidden_size
            input_sequence_length = 6
            decoder_sequence_length = 7
            num_decoder_symbols = 20
            start_of_sequence_id = end_of_sequence_id = 1

            decoder_embeddings = tf.get_variable(
                "decoder_embeddings", [num_decoder_symbols, decoder_embedding_size],
            	initializer=tf.random_normal_initializer(stddev=0.1))

            inputs = tf.constant(
                0.5,
                shape=[input_sequence_length, batch_size, encoder_embedding_size])

            decoder_inputs = tf.constant(
                0.4,
                shape=[decoder_sequence_length, batch_size, decoder_embedding_size])

            decoder_length = tf.constant(
                decoder_sequence_length, dtype=dtypes.int32, shape=[batch_size,])
	    
	    with tf.variable_scope("rnn") as scope:
                # setting up weights for computing the final output
                output_fn = lambda x: layers.linear(x, num_decoder_symbols,
                                              scope=scope)

                # Define model
                encoder_outputs, encoder_state = rnn.dynamic_rnn(
                    cell=core_rnn_cell_impl.GRUCell(encoder_hidden_size),
                    inputs=inputs,
                    dtype=dtypes.float32,
                    time_major=True,
                    scope=scope)
		
	    with tf.variable_scope("decoder") as scope:
               # Train decoder
               decoder_cell = core_rnn_cell_impl.GRUCell(decoder_hidden_size)

               decoder_fn_train = _decoder_fn_with_context_state(
                   decoder_fn_lib.simple_decoder_fn_train(encoder_state=encoder_state))

               (decoder_outputs_train, decoder_state_train,
                   decoder_context_state_train) = seq2seq.dynamic_rnn_decoder(
                       cell=decoder_cell,
                       decoder_fn=decoder_fn_train,
                       inputs=decoder_inputs,
                       sequence_length=decoder_length,
                       time_major=True,
                       scope=scope)

               decoder_outputs_train = output_fn(decoder_outputs_train)

               # Setup variable reuse
               scope.reuse_variables()

               # Inference decoder
               decoder_fn_inference = _decoder_fn_with_context_state(
                   decoder_fn_lib.simple_decoder_fn_inference(
                       output_fn=output_fn,
                       encoder_state=encoder_state,
                       embeddings=decoder_embeddings,
                       start_of_sequence_id=start_of_sequence_id,
                       end_of_sequence_id=end_of_sequence_id,
                       maximum_length=decoder_sequence_length - 1,
                       num_decoder_symbols=num_decoder_symbols,
                       dtype=dtypes.int32))

               (decoder_outputs_inference, decoder_state_inference,
                   decoder_context_state_inference) = (seq2seq.dynamic_rnn_decoder(
               	       cell=decoder_cell,
                       decoder_fn=decoder_fn_inference,
                       time_major=True,
                       scope=scope)) 

	       output_train = tf.argmax(decoder_outputs_train, axis=2)
	       output_inference = tf.argmax(decoder_outputs_inference, axis=2)

	       tf.global_variables_initializer().run()
               (decoder_outputs_train_res, decoder_state_train_res,
         		decoder_context_state_train_res) = sess.run([
             			decoder_outputs_train, decoder_state_train,
             			decoder_context_state_train
         		])

               (decoder_outputs_inference_res, decoder_state_inference_res,
         		decoder_context_state_inference_res) = sess.run([
             			decoder_outputs_inference, decoder_state_inference,
             			decoder_context_state_inference
         		])

	       print np.shape(decoder_outputs_train_res)
	       print np.shape(decoder_outputs_inference_res)
	       output_train, output_inference = sess.run([output_train, output_inference])
	       print output_train
	       print output_inference
    
if __name__ == '__main__':
    test_dynamic_rnn_decoder()
    
