
'''
A Simple Neural Network  implementation example using TensorFlow library.
This model is used as a base class for models that can be trained

Author: Max Ferguson
'''

from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.contrib import rnn, seq2seq
from tensorflow.contrib.seq2seq import Helper, TrainingHelper, BasicDecoder
from tensorflow.contrib.layers import xavier_initializer


class TensorflowModel:

    def __init__(self):
        self.loss = None
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

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


    def train(self, sess, inputs, outputs, rewards):
        """Train the model on a batch of data"""
        outputs = self.get_training_outputs()
        feed = self.get_training_feed_dict(*args, **kwargs)
        return sess.run(outputs, feed_dict=feed)


    def create_saver(self):
        """
        Create a saver object.
        Must be called after all the variables have been created
        """
        self.saver = tf.train.Saver()


    def save(self, sess, directory, name, step):
        """
        Saves session
        @directory: The directory to save into
        @name: The name of the model
        @step: The current step
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        filepath = os.path.join(directory, name)
        savepath = self.saver.save(sess, filepath, global_step=step)
        print("\nSaved model to ", savepath)


    def restore(self, sess, directory,  name):
        """
        Loads a session. Does not raise an error on failure
        @directory: The directory to save into
        @name: The name of the model
        """
        if not os.path.exists(directory):
            print("No such directory", directory)
            return

        savepath = tf.train.latest_checkpoint(directory)

        if savepath:
            self.saver.restore(sess, savepath)
            print("Restored model from", savepath)
        else:
            print("Unable to find checkpoint at",directory)