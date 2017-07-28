import time
import numpy as np
import tensorflow as tf
from pprint import pprint
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.seq2seq import ScheduledOutputTrainingHelper, BasicDecoder
from .helpers import InferenceHelper
from .base import TensorflowModel



class Sequence2Sequence(TensorflowModel):
    """
    A simple sequence2sequence implimentatation with
    dense input and dense output vectors
    """

    def __init__(self, rnn_size, num_layers, num_features, logs_path, **kwargs):
        """
        Build the model
        @rnn_size: The number of hidden units in each layer
        @num_layers: The number of rnn layers
        @num_features: The size of the feature vector that is passed in at each timestep
        """
        super().__init__()
        self.create_model_inputs(num_features, **kwargs)
        self.create_model(num_features, rnn_size, num_layers)
        self.create_writer(logs_path)
        self.create_saver()



    def create_model_inputs(self, num_features, **kwargs):
        """
        Create all of the placeholders
        Uses tensors specified in **kwargs is available
        """
        if len(kwargs):
          self.input_data = kwargs["sources_batch"]
          self.targets = kwargs["targets_batch"]
          self.target_sequence_length = kwargs["targets_lengths"]
          self.source_sequence_length = kwargs["sources_lengths"]
        else:
          self.input_data = tf.placeholder(tf.float32, [None, None, num_features], name='input')
          self.targets = tf.placeholder(tf.float32, [None, None, num_features], name='targets')
          self.target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
          self.source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
        # Always just calculate these
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.max_target_sequence_length = tf.reduce_max(self.target_sequence_length, name='max_target_len')



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




        # Create the training operation by calling optimizer.apply_gradients
        train_op = optimizer.apply_gradients(zip(gradients, v))




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

      # Gradient clipping
      gradients, v = zip(*optimizer.compute_gradients(self.train_loss_op))
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
      self.grad_norm = tf.global_norm(clipped_gradients)

      # Training operation
      self.train_op = optimizer.apply_gradients(zip(gradients, v))

      # Create gradient summaries
      for i in range(0,2):
        with tf.name_scope('gradient-%i'%i):
          self.create_variable_summary(gradients[i], "unclipped")
          self.create_variable_summary(clipped_gradients[i], "clipped")

      # Create all the summaries for tensorboard
      self.create_variable_summaries()


    def create_variable_summary(self, var, name):
      """Create a variable summary for a single variable"""
      with tf.name_scope(name):
        tf.summary.scalar('mean', tf.reduce_mean(var))
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))

    def create_variable_summaries(self):
      """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
      with tf.name_scope('summaries'):
        tf.summary.scalar('grad_norm', self.grad_norm)
        tf.summary.scalar('train_loss', tf.reduce_min(self.train_loss_op))
        tf.summary.scalar('eval_loss', tf.reduce_min(self.eval_loss_op))

        # Merge all the test summaries
        self.test_summaries = tf.summary.merge_all()

        # Define additional train summaries
        tf.summary.scalar('learning_rate', self.lr)

        # Merge all train summaries
        self.train_summaries = tf.summary.merge_all()


    def train(self, sess, learning_rate, summarize=False, sources_batch=None, targets_batch=None, targets_lengths=None, sources_lengths=None):
        """Train the model on a batch of data"""
        feed = { self.lr: learning_rate }

        if sources_batch:
          feed[self.input_data] = sources_batch

        if targets_batch:
          feed[self.targets] = targets_batch

        if targets_lengths:
          feed[self.target_sequence_length] = targets_lengths

        if sources_lengths:
          feed[self.source_sequence_length] = sources_lengths

        if summarize:
          # Run the training operation and collect variable summaries
          ops = [self.train_op, self.global_step_op, self.train_loss_op, self.train_summaries]

          _, step, loss, summary = sess.run(ops, feed_dict=feed)

          self.writer.add_summary(summary, step)

        else:
          # Run the training operation without summaries
          ops = [self.train_op, self.global_step_op, self.train_loss_op]

          # Run gradient descent on this batch
          _, step, loss = sess.run(ops, feed_dict=feed)

        return loss



    def evaluate(self, sess, sources_batch=None, targets_batch=None, targets_lengths=None, sources_lengths=None):
        """Evaluate the model on a batch of data. Return the loss on sources_batch"""
        feed = {}

        if sources_batch:
          feed[self.input_data] = sources_batch

        if targets_batch:
          feed[self.targets] = targets_batch

        if targets_lengths:
          feed[self.target_sequence_length] = targets_lengths

        if sources_lengths:
          feed[self.source_sequence_length] = sources_lengths

        i = tf.train.global_step(sess, self.global_step)

        # Define the operations we want to run
        ops = [self.train_loss_op, self.eval_loss_op, self.test_summaries]

        # Evaluate the model without training
        train_loss, eval_loss, summary = sess.run(ops, feed_dict=feed)

        return train_loss, eval_loss



    def predict(self, sess, sources_batch=None, targets_lengths=None, sources_lengths=None):
        """Return the predictions for a set of sources"""
        feed = {}

        if sources_batch:
          feed[self.input_data] = sources_batch

        if targets_lengths:
          feed[self.target_sequence_length] = targets_lengths

        if sources_lengths:
          feed[self.source_sequence_length] = sources_lengths

        # Return all the data tensors
        outputs = [
          self.input_data,
          self.targets,
          self.inference_decoder_output
        ]

        return sess.run(outputs, feed_dict=feed)




