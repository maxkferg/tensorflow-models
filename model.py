
'''
A Simple Neural Network (LSTM) implementation example using TensorFlow library.
This model is used as a base class for models that can be trained

Author: Max Ferguson
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn, seq2seq
from tensorflow.contrib.seq2seq import Helper, TrainingHelper, BasicDecoder
from tensorflow.contrib.layers import xavier_initializer


class TensorflowModel:

    def __init__(self):
        self.loss = None
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)


    def get_training_outputs(self):
        """Return the training output tensors"""
        return [self.train_op, self.loss]

    def get_testing_outputs(self):
        """Return the testing output tensors"""
        return [self.train_op, self.loss]

    def get_training_feed_dict(self, *args, **kwargs):
        """Return the feed dict that should be used for training"""
        raise NotImplementedError

    def get_testing_feed_dict(self, *args, **kwargs):
        """Return the feed dict that should be used for testing"""
        raise NotImplementedError

    def predict(self, sess, *args, **kwargs):
        """Evaluate the model on a batch of data"""
        outputs = self.get_testing_outputs()
        feed = self.get_testing_feed_dict(*args, **kwargs)
        return sess.run(outputs, feed_dict=feed)


    def train(self,sess,inputs,outputs,rewards):
        """Train the model on a batch of data"""
        outputs = self.get_training_outputs()
        feed = self.get_training_feed_dict(*args, **kwargs)
        return sess.run(outputs, feed_dict=feed)

