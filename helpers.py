import time
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.seq2seq import CustomHelper



class InferenceHelper(CustomHelper):
  """Base abstract class that allows the user to customize sampling."""

  def __init__(self, start_token):
    """Extend the custom helper"""

    super().__init__(
      initialize_fn=self.initialize_fn,
      sample_fn=self.sample_fn,
      next_inputs_fn=self.next_inputs_fn
    )
    self.start_token = start_token
    self.zeros = 0*tf.reduce_mean(start_token, axis=1) # Zeros in [batch_size]


  def initialize_fn(self,):
    """
    Callable that returns `(finished, next_inputs)` for the first iteration.
    """
    finished = tf.cast(self.zeros, tf.bool)
    next_inputs = self.start_token
    return (finished, next_inputs)


  def sample_fn(self, time, outputs, state):
    """
    Callable that takes `(time, outputs, state)`  and emits tensor `sample_ids`.
    """
    sample_ids = tf.cast((self.zeros - 1), tf.int32) # Never sampling
    return sample_ids


  def next_inputs_fn(self, time, outputs, state, sample_ids):
    """
    Callable that takes `(time, outputs, state, sample_ids)` and emits `(finished, next_inputs, next_state)`.
    """
    finished = tf.cast(self.zeros, tf.bool)
    next_inputs = outputs
    next_state = state
    return (finished, next_inputs, next_state)
