import tensorflow as tf
import numpy as np
import logging
import os


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('var_%s' % name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def raise_error(condition, msg):
    """raising error with a condition"""
    if condition:
        raise ValueError(msg)


def create_log(out_file_path: str):
    """ Simple logger """
    logger = logging.getLogger(out_file_path)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("H1, %(asctime)s %(levelname)8s %(message)s")
    for handler in [logging.FileHandler(out_file_path), logging.StreamHandler()]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def full_connected(x, weight_shape, scope=None, bias=True):
    """ fully connected layer: weight_shape: input size, output size """
    with tf.variable_scope(scope or "fully_connected", reuse=None):
        w = tf.get_variable("weight", shape=weight_shape, dtype=tf.float32)
        x = tf.matmul(x, w)
        if bias:
            return tf.add(x, tf.get_variable("bias", initializer=[0.0] * weight_shape[-1]))
        else:
            return x


class LanguageModel:
    """ Neural Language Model """

    def __init__(self,
                 checkpoint_dir: str,
                 model: str,
                 config: dict,
                 keep_prob: list,
                 keep_prob_r: list,
                 gradient_clip=None,
                 weight_decay=0.0,
                 weight_tying=False,
                 layer_norm=False,
                 optimizer='sgd',
                 batch_size=50,
                 ini_scale=0.01):

        raise_error(model not in ['lstm', 'rhn', 'kvp', 'hsg', 'hyper'], 'unknown model %s' % model)
        raise_error(optimizer not in ['sgd', 'adam'], 'unknown optimizer %s' % optimizer)
        self.__model = model      
        self.__checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.__checkpoint_dir):
            os.makedirs(self.__checkpoint_dir, exist_ok=True)
        self.__checkpoint = '%s/model.ckpt' % self.__checkpoint_dir
        self.__config = config
        self.__clip = gradient_clip
        self.__layer_norm = layer_norm
        self.__keep_prob = keep_prob
        self.__keep_prob_r = keep_prob_r
        self.__weight_decay = weight_decay
        self.__weight_tying = weight_tying
        self.__optimizer = optimizer
        self.__batch_size = batch_size
        self.__ini_scale = ini_scale
        self.__log = create_log('%s/log' % self.__checkpoint_dir)
        self.__log.info('BUILD GRAPH: %s' % self.__model)
        initializer = tf.random_uniform_initializer(-ini_scale, ini_scale, seed=0)
        with tf.variable_scope("language_model", initializer=initializer, reuse=None):
            self.__build_graph()

        self.__session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.__summary = tf.summary.merge_all()
        self.__writer_train = tf.summary.FileWriter('%s/summary_train' % self.__checkpoint_dir, self.__session.graph)
        self.__writer_valid = tf.summary.FileWriter('%s/summary_valid' % self.__checkpoint_dir)
        self.__session.run(tf.global_variables_initializer())

    def __build_graph(self):
        #########
        # setup #
        #########
        self.__inputs = tf.placeholder(tf.int32, [None, self.__config["num_steps"]], name="inputs")
        self.__targets = tf.placeholder(tf.int32, [None, self.__config["num_steps"]], name="output")
        self.__is_training = tf.placeholder_with_default(False, [])

        # dropout for embedding and output layer
        print(self.__keep_prob[0])
        __keep_prob_emb = tf.where(self.__is_training, self.__keep_prob[0], 1.0)
        __keep_prob_out = tf.where(self.__is_training, self.__keep_prob[1], 1.0)
        tf.summary.scalar('meta_keep_prob_emb', __keep_prob_emb)
        tf.summary.scalar('meta_keep_prob_out', __keep_prob_out)

        # dropout for lstm layer
        __keep_prob_r = tf.where(self.__is_training, self.__keep_prob_r, [1.0] * len(self.__keep_prob_r))
        variable_summaries(__keep_prob_r, 'meta_keep_prob_r')

        # weight decay
        __weight_decay = tf.where(self.__is_training, self.__weight_decay, 0)
        tf.summary.scalar('meta_weight_decay', __weight_decay)

        ################################
        # main lstm-based architecture #
        ################################

        # embedding
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [self.__config["vocab_size"], self.__config["embedding_size"]])
            inputs = tf.nn.embedding_lookup(embedding, self.__inputs)

        # lstm
        inputs = tf.nn.dropout(inputs, __keep_prob_emb)
        with tf.variable_scope("RNNCell"):
            if self.__model == 'hypernets':
                outputs, n_hidden = self.__hypernets(inputs, __keep_prob_r)
            elif self.__model in ['rhn', 'hsg']:
                outputs, n_hidden = self.__rhn(inputs, __keep_prob_r)
            elif self.__model == "kvp":
                outputs, n_hidden = self.__kvp(inputs, __keep_prob_r)
            else:  # lstm
                outputs, n_hidden = self.__lstm(inputs, __keep_prob_r)

        # weight is shared sequence direction (in addition to batch direction), so reshape to treat sequence direction
        # as batch direction. output: (batch, num_steps, last hidden size) -> (batch x num_steps, last hidden size)
        outputs = tf.nn.dropout(tf.reshape(outputs, [-1, n_hidden]), __keep_prob_out)
        # prediction
        with tf.variable_scope("fully_connected", reuse=None):
            weight = [n_hidden, self.__config["vocab_size"]]
            logit = full_connected(outputs, weight, bias=True, scope="fc")
            # Reshape logit to be a 3-D tensor for sequence loss
            logit = tf.reshape(logit, [self.__batch_size, self.__config["num_steps"], self.__config["vocab_size"]])
            self.__prediction = tf.nn.softmax(logit)
        
        ################
        # optimization #
        ################
        with tf.variable_scope("optimization"):
            self.__loss = tf.reduce_sum(
                tf.contrib.seq2seq.sequence_loss(
                    logits=logit,
                    targets=self.__targets,
                    weights=tf.ones([self.__batch_size, self.__config["num_steps"]], dtype=tf.float32),
                    average_across_timesteps=False, average_across_batch=True))
            self.__perplexity = tf.exp(self.__loss/self.__config['num_steps'])
            tf.summary.scalar('eval_loss', self.__loss)
            tf.summary.scalar('eval_perplexity', self.__perplexity)
            if self.__weight_decay != 0.0:  # L2 (weight decay)
                self.__loss += tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * __weight_decay

            self.__learning_rate = tf.placeholder(tf.float32, [], 'learning_rate')  # decay learning rate
            tf.summary.scalar('meta_learning_rate', self.__learning_rate)
            if self.__optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(self.__learning_rate)
            else:
                optimizer = tf.train.GradientDescentOptimizer(self.__learning_rate)
            
            if self.__clip is not None:
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.__loss, tf.trainable_variables()), self.__clip)
                self.__train_op = optimizer.apply_gradients(zip(grads, tf.trainable_variables()))
            else:
                self.__train_op = optimizer.minimize(self.__loss)
        
        # count trainable variables
        n_var = 0
        for var in tf.trainable_variables():
            variable_summaries(var, var.name.split(':')[0].replace('/', '-'))
            self.__log.info('%s: %i' % (var.name, int(np.prod(var.get_shape().as_list()))))
            n_var += int(np.prod(var.get_shape().as_list()))
        self.__log.info('total variables:%i' % n_var)
        # saver
        self.__saver = tf.train.Saver()
    
    def __kvp(self, inputs, keep_r):  # kvp-attention
        from lstm_cell import CustomLSTMCell, KVPAttentionWrapper
        cells = []
        for i in range(self.__config["n_lstm_layer"]):
            cell = CustomLSTMCell(num_units=self.__config['num_units'],
                                  recurrent_dropout=self.__config['recurrent_dropout'],
                                  dropout_keep_prob=keep_r[0])
            cells.append(cell)
        cells = tf.nn.rnn_cell.MultiRNNCell(cells)
        attention_layer = KVPAttentionWrapper(cells,
                                              attention_window=self.__config["attention_window"],
                                              sequence_length=self.__config["num_steps"],
                                              mode=self.__config["mode"])

        self.__initial_state = cells.zero_state(batch_size=self.__batch_size, dtype=tf.float32)
        outputs, self.__final_state = attention_layer(inputs, self.__initial_state)
        self.__alignment_history_size = attention_layer.alignment_history_size
        self.__alignment_history = attention_layer.alignment_history
        return outputs, attention_layer.n_hidden
    
    def __lstm(self, inputs, keep_r):  # vanilla LSTM: stacked LSTM layer
        from lstm_cell import CustomLSTMCell
        cells = []
        for i in range(self.__config["n_lstm_layer"]):
            cell = CustomLSTMCell(num_units=self.__config['num_units'],
                                  recurrent_dropout=self.__config['recurrent_dropout'],
                                  dropout_keep_prob=keep_r[0])
            cells.append(cell)
        cells = tf.nn.rnn_cell.MultiRNNCell(cells)
        outputs = []
        self.__initial_state = cells.zero_state(batch_size=self.__batch_size, dtype=tf.float32)
        state = self.__initial_state
        for time_step in range(self.__config["num_steps"]):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cells(inputs[:, time_step, :], state)
            outputs.append(cell_output)
        self.__final_state = state 
        return tf.stack(outputs, axis=1), self.__config["num_units"]
    
    def __hypernets(self, inputs, keep_r):  # hypernets
        from lstm_cell import HyperLSTMCell
        cells = []
        for i in range(self.__config["n_lstm_layer"]):
            cell = HyperLSTMCell(num_units=self.__config['num_units'],
                                 num_units_hyper=self.__config['num_units_hyper'],
                                 recurrent_dropout=self.__config['recurrent_dropout'],
                                 dropout_keep_prob=keep_r)
            cells.append(cell)
        cells = tf.nn.rnn_cell.MultiRNNCell(cells)
        outputs = []
        self.__initial_state = cells.zero_state(batch_size=self.__batch_size, dtype=tf.float32)
        state = self.__initial_state
        for time_step in range(self.__config["num_steps"]):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cells(inputs[:, time_step, :], state)
            outputs.append(cell_output)
        self.__final_state = state 
        return tf.stack(outputs, axis=1), self.__config["num_units"]

    def __rhn(self, inputs, keep_r):  # RHN, HSG
        from lstm_cell import CustomRNNCell
        cells = CustomRNNCell(recurrent_highway=self.__config['recurrent_highway'],
                              recurrence_depth=self.__config['recurrence_depth'],
                              highway_state_gate=self.__config['highway_state_gate'],
                              num_units=self.__config['num_units'],
                              dropout_keep_prob_in=keep_r[0],
                              dropout_keep_prob_h=keep_r[1])
        outputs = []
        self.__initial_state = cells.zero_state(batch_size=self.__batch_size, dtype=tf.float32)
        state = self.__initial_state
        for time_step in range(self.__config["num_steps"]):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cells(inputs[:, time_step, :], state)
            outputs.append(cell_output)
        self.__final_state = state 
        return tf.stack(outputs, axis=1), self.__config["num_units"]

    def train(self,
              max_max_epoch: int,
              max_epoch: int,
              batcher_train,
              batcher_valid,
              batcher_test,
              learning_rate: float,
              lr_decay: float = None,
              verbose=False):
        
        self.__log.info("max epoch (%i), max max epoch (%i)" % (max_epoch, max_max_epoch))
        ini_state = None
        result = []
        i_summary_train, i_summary_valid = 0, 0
        for e in range(max_max_epoch):
            # Train
            perplexity, loss = 0.0, 0.0
            for step, (inp, tar) in enumerate(batcher_train):
                feed_dict = dict(((self.__inputs, inp), (self.__targets, tar), (self.__is_training, True)))
                if ini_state is not None:
                    feed_dict[self.__initial_state] = ini_state
                if e >= max_epoch and lr_decay is not None:
                    learning_rate = learning_rate / lr_decay
                feed_dict[self.__learning_rate] = learning_rate
                tmp_loss, tmp_perp, ini_state, _, summary = self.__session.run(
                    [self.__loss, self.__perplexity, self.__final_state, self.__train_op, self.__summary],
                    feed_dict=feed_dict)
                perplexity += tmp_perp
                loss += tmp_loss
                self.__writer_train.add_summary(summary, i_summary_train)
                i_summary_train += 1
                if verbose and step % (batcher_train.iteration_number // 10) == 10:
                    self.__log.info("epoch %i-%i/%i perplexity: %.3f, loss: %0.3f"
                                    % (e, step, batcher_train.iteration_number, perplexity/step, loss/step))
            perplexity, loss = perplexity/step, loss/step

            # Valid
            perplexity_v, loss_v = 0.0, 0.0
            for step, (inp, tar) in enumerate(batcher_valid):
                feed_dict = dict(((self.__inputs, inp), (self.__targets, tar), (self.__is_training, False)))
                feed_dict[self.__learning_rate] = learning_rate
                tmp_loss, tmp_perp, summary = self.__session.run(
                    [self.__loss, self.__perplexity, self.__summary], feed_dict=feed_dict)
                perplexity_v += tmp_perp
                loss_v += tmp_loss
                self.__writer_valid.add_summary(summary, i_summary_valid)
                i_summary_valid += 1
            perplexity_v, loss_v = perplexity_v/step, loss_v/step
            self.__log.info("epoch %i, perplexity: train %0.3f, valid %0.3f" % (e, perplexity, perplexity_v))
            result.append([perplexity, perplexity_v, loss, loss_v])

        # Test
        perplexity_t, loss_t = 0.0, 0.0
        for step, (inp, tar) in enumerate(batcher_test):
            feed_dict = dict(((self.__inputs, inp), (self.__targets, tar), (self.__is_training, False)))
            tmp_loss, tmp_perp = self.__session.run([self.__loss, self.__perplexity], feed_dict=feed_dict)
            loss_t += tmp_loss
            perplexity_t += tmp_perp
        loss_t = loss_t/step
        perplexity_t = perplexity_t/step
        self.__log.info("test perplexity %0.3f" % perplexity_t)
        self.__saver.save(self.__session, self.__checkpoint)
        np.savez("%s/statistics.npz" % self.__checkpoint_dir,
                 loss=np.array(result),
                 test=np.array(perplexity_t, loss_t))
