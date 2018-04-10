"""LSTM based  PTB Language model.
Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:
$ python lstm_langmodel.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import reader

class TrainConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  rnn_mode = 'block'

class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 1
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 1
  vocab_size = 10000
  rnn_mode = 'block'

def data_type():
  return tf.float32

class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)

class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config):
    self._is_training = is_training
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])    # input
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])   # output, length num_steps

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
        "embedding", [vocab_size, size], dtype=data_type())
      print(embedding)
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    output, state = self._build_rnn_graph_lstm(inputs, config, is_training)

    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
     # Reshape logits to be a 3-D tensor for sequence loss
    logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

    # Use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        self._targets,
        tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
        average_across_timesteps=False,
        average_across_batch=True)

    # Update the cost
    self._cost = tf.reduce_sum(loss)
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.train.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def _get_lstm_cell(self, config, is_training):
    if config.rnn_mode == 'basic':
      return tf.contrib.rnn.BasicLSTMCell(
          config.hidden_size, forget_bias=0.0, state_is_tuple=True,
          reuse=not is_training)
    if config.rnn_mode == 'block':
      return tf.contrib.rnn.LSTMBlockCell(
          config.hidden_size, forget_bias=0.0)
    raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

  def _build_rnn_graph_lstm(self, inputs, config, is_training):
    """Build the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def make_cell():
      cell = self._get_lstm_cell(config, is_training)
      if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=config.keep_prob)
      return cell

    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(config.batch_size, data_type())
    state = self._initial_state
    # Simplified version of tf.nn.static_rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use tf.nn.static_rnn() or tf.nn.static_state_saving_rnn().
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
    # outputs, state = tf.nn.static_rnn(cell, inputs,
    #                                   initial_state=self._initial_state)
    outputs = []
    with tf.variable_scope("RNN"):
      for time_step in range(self.num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        print(time_step)
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)
    output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
    return output, state

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input_data

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def initial_state_name(self):
    return self._initial_state_name

  @property
  def final_state_name(self):
    return self._final_state_name

import time
def run_epoch(session, model, data, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  step = 0
  for (x, y) in enumerate(reader.ptb_producer(data, model.batch_size, model.num_steps)):
    feed_dict = {}
    print('x is ', x)
    print('y is ', y)
    #session.run(x)
    session.run(y)
    print('x is ', x)
    print('y is ', y)
    feed_dict[model._input_data] = x
    feed_dict[model._targets] = y
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h
    #print(feed_dict)
    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size * 1 /
             (time.time() - start_time)))

    step += 1
  return np.exp(costs / iters)


#http://lib.csdn.net/article/98/59839?knId=1756
if __name__ == "__main__":
  data_path = './simple-examples/data/'
  raw_data = reader.ptb_raw_data(data_path)
  train_data, valid_data, test_data, _ = raw_data

  config = TrainConfig()
  test_config = TestConfig()

  with tf.Graph().as_default():
    init = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      with tf.variable_scope("Model", reuse=None, initializer=init):
        mtrain = PTBModel(is_training=True, config=config)
      tf.summary.scalar("Training Loss", mtrain.cost)
      tf.summary.scalar("Learning Rate", mtrain.lr)

    with tf.name_scope("Validation"):
      with tf.variable_scope("Model", reuse=True, initializer=init):
        mvalid = PTBModel(is_training=False, config=config)
      tf.summary.scalar("Validation Loss", mvalid.cost)
    """
    with tf.name_scope("Test"):
      test_input = PTBInput(config=test_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=init):
        mtest = PTBModel(is_training=False, config=test_config, input_=test_input)
    """
    #models = {"Train": mtrain, "Validation": mvalid, "Test": mtest}

    saver = tf.train.Saver()

    #tf.initialize_all_variables().run()
    tf.global_variables_initializer()

    # Start the training/validation/evaluation
    with tf.Session() as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        mtrain.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(mtrain.lr)))
        train_perplexity = run_epoch(session, mtrain, train_data, eval_op=mtrain.train_op, verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid, valid_data)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      #test_perplexity = run_epoch(session, mtest)
      #print("Test Perplexity: %.3f" % test_perplexity)

      print("Saving model to %s." % FLAGS.save_path)
      saver.save(session, "./my_model_final.ckpt")
