import time
import numpy as np
import tensorflow as tf
from pprint import pprint
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.seq2seq import ScheduledOutputTrainingHelper, BasicDecoder
from helpers import InferenceHelper
from model import TensorflowModel



class Sequence2Sequence(TensorflowModel):
    """
    A simple sequence2sequence implimentatation with
    dense input and dense output vectors
    """

    def __init__(self, rnn_size, num_layers, num_features):
        """
        Build the model
        @rnn_size: The number of hidden units in each layer
        @num_layers: The number of rnn layers
        @num_features: The size of the feature vector that is passed in at each timestep
        """
        self.create_model_inputs(num_features)
        self.create_model(num_features, rnn_size, num_layers)



    def create_model_inputs(self, num_features):
        """Create all of the palceholders"""
        self.input_data = tf.placeholder(tf.float32, [None, None, num_features], name='input')
        self.targets = tf.placeholder(tf.float32, [None, None, num_features], name='targets')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        self.target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
        self.max_target_sequence_length = tf.reduce_max(self.target_sequence_length, name='max_target_len')
        self.source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')



    def create_model(self, num_features, rnn_size, num_layers):
        # Pass the input data through the encoder. We'll ignore the encoder output, but use the state
        _, enc_state = self.create_encoding_layer(num_features, rnn_size, num_layers)

        # Pass encoder state and decoder inputs to the decoders
        training_dec_out, inference_dec_out = self.create_decoding_layer(num_features, rnn_size, num_layers, enc_state)

        # Store the two output heads
        self.training_decoder_output = training_dec_out
        self.inference_decoder_output = inference_dec_out



    def create_encoding_layer(self, num_features, rnn_size, num_layers):
        # Encoder inputs
        enc_input = self.input_data
        enc_sequence_length = self.source_sequence_length

        # RNN cell
        def make_cell(rnn_size):
          enc_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
          return enc_cell

        enc_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])

        enc_output, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_input, sequence_length=enc_sequence_length, dtype=tf.float32)

        return enc_output, enc_state



    def get_decoder_input(self):
        """Return the input for decoders.
        Appends the start token to the training inputs"""
        start_token = self.input_data[:,-1,:]
        first_timestep = tf.expand_dims(start_token,axis=1)
        training_dec_input = tf.concat([first_timestep, self.targets], 1) # Concat in time axis
        inference_dec_input = start_token
        return training_dec_input, inference_dec_input



    def create_decoding_layer(self, num_features, rnn_size, num_layers, enc_state):
        # Input variables
        target_sequence_length = self.target_sequence_length
        max_target_sequence_length =  self.max_target_sequence_length

        # 1. Construct the decoder cell
        def make_cell(rnn_size):
            dec_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                             initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return dec_cell

        dec_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])

        # 2. Dense layer to translate the decoder's output at each time
        # step into the desired output vector
        # TODO: Allow this to be extended
        output_layer = Dense(num_features, kernel_initializer=tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))

        # Get the input for the two decoders
        # The input to the training decoder is [start_token, targets]
        # The input to the inference decoder is [start_token]
        training_dec_input, start_token = self.get_decoder_input()

        # 3. Set up a training decoder and an inference decoder
        # Training Decoder
        with tf.variable_scope("decode"):

            # Helper for the training process. Used by BasicDecoder to read inputs.
            training_helper = ScheduledOutputTrainingHelper(inputs=training_dec_input,
                                                            sequence_length=target_sequence_length,
                                                            sampling_probability=0.5,
                                                            time_major=False)

            # Basic decoder
            training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                             training_helper,
                                                             enc_state,
                                                             output_layer=output_layer)

            # Perform dynamic decoding using the decoder
            training_decoder_output,_ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                         maximum_iterations=max_target_sequence_length)[0]

        # 5. Inference Decoder
        # Reuses the same parameters trained by the training process
        with tf.variable_scope("decode", reuse=True):

            # Helper for the inference process.
            last_encoder_input = self.input_data[:,-1,:]
            inference_helper = InferenceHelper(start_token=start_token)

            # Basic decoder
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                          inference_helper,
                                                          enc_state,
                                                          output_layer=output_layer) #########################

            # Perform dynamic decoding using the decoder
            inference_decoder_output,_ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                              maximum_iterations=max_target_sequence_length)[0]

        return training_decoder_output, inference_decoder_output




class TrainableSequence2Sequence(Sequence2Sequence):
    """
    A sequence to sequence model that can be trained and evaluated
    """

    def __init__(self,*args,**kwargs):
      super().__init__(*args,**kwargs)

      # Define two immportant loss ops
      self.train_loss_op = tf.nn.l2_loss(self.targets - self.training_decoder_output)
      self.eval_loss_op = tf.nn.l2_loss(self.targets - self.inference_decoder_output)

      # Adam optimiser with gradient clipping
      optimizer = tf.train.AdamOptimizer(self.lr)
      gradients = optimizer.compute_gradients(self.train_loss_op)
      capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]

      # Training operation
      self.train_op = optimizer.apply_gradients(capped_gradients)


    def train(self, sess, sources_batch, targets_batch, learning_rate, targets_lengths, sources_lengths):
        """Train the model on a batch of data"""
        # Feed
        feed = { self.input_data: sources_batch,
                 self.targets: targets_batch,
                 self.lr: learning_rate,
                 self.target_sequence_length: targets_lengths,
                 self.source_sequence_length: sources_lengths
              }

        # Run gradient descent on this batch
        _, loss = sess.run([self.train_op, self.train_loss_op], feed_dict=feed)
        return loss


    def evaluate(self, sess, sources_batch, targets_batch, learning_rate, targets_lengths, sources_lengths):
        """Evaluate the model on a batch of data. Return the loss on sources_batch"""
        # Feed
        feed = { self.input_data: sources_batch,
                 self.targets: targets_batch,
                 self.target_sequence_length: targets_lengths,
                 self.source_sequence_length: sources_lengths
              }

        return sess.run([self.train_loss_op, self.eval_loss_op], feed_dict=feed)


    def predict(self, sess, sources_batch, targets_lengths, sources_lengths):
        """Return the predictions for a set of sources"""
        # Feed
        feed = { self.input_data: sources_batch,
                 self.target_sequence_length: targets_lengths,
                 self.source_sequence_length: sources_lengths
               }

        # Return the output from the inference decoder
        return sess.run(self.inference_decoder_output, feed_dict=feed)

