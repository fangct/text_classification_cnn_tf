import tensorflow as tf
import os
import numpy as np


class CNN_MODEL(object):
    def __init__(self,
                 embed_dim,
                 filter_sizes='2,3,4',
                 max_sent_len=100,
                 embedding_mat=None,
                 word_nums=100,
                 filter_nums = 128,
                 label_nums=2,
                 learning_rate=0.1,
                 model_path='',
                 epoch=10,
                 batch_size=64,
                 dropout_prob=0.2):

        self.embed_dim = embed_dim
        self.filter_sizes = [int(filter) for filter in filter_sizes.split(',')]
        self.sent_len = max_sent_len
        self.pre_embedding = embedding_mat
        self.word_nums = word_nums
        self.filter_nums = filter_nums
        self.label_nums = label_nums
        self.basic_learning_rate = learning_rate
        self.model_path = model_path
        self.epoch = epoch
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob

        self.build_model()


    def place_holder(self):
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.sent_len], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.int64, shape=[None, self.label_nums], name='input_y')
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name='dropout')

    def embedding(self):
        self.W_embed = tf.Variable(self.pre_embedding, dtype=tf.float32, name='pre_embedding', trainable=True) # shape=(N_words, 300)
        word_embedding = tf.nn.embedding_lookup(self.W_embed, self.input_x) # shape=(1-, 100, 300)

        # expand the dimension of input x to 3 dim, shape=(-1, 100, 300, 1)
        self.word_embedding = tf.expand_dims(word_embedding, -1)


    def conv_layer(self):
        output_pool = []
        for filter_size in self.filter_sizes:
            with tf.variable_scope('conv-maxpool-%s' % filter_size, reuse=tf.AUTO_REUSE):
                filter_shape = [filter_size, self.embed_dim, 1, self.filter_nums]
                W_conv = tf.get_variable(shape=filter_shape, initializer=tf.truncated_normal_initializer(-0.1, 0.1), name='W_conv')
                b_conv = tf.get_variable(shape=self.filter_nums, initializer=tf.constant_initializer(0.1), name='b_conv')


                h_conv = tf.nn.conv2d(self.word_embedding, W_conv,
                                      [1, 1, 1, 1], padding='VALID', name='conv') # shape=(-1, 99/98/97, 1, 128)

                act_conv = tf.nn.relu(tf.nn.bias_add(h_conv, b_conv), name='rele')# shape=(-1, 99/98/97, 1, 128)
                conv_len = act_conv.get_shape()[1] # 99/98/97

                h_pool = tf.nn.max_pool(act_conv, ksize=[1, conv_len, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name='pool') # shape=(-1, 1, 1, 128)
                output_pool.append(h_pool)


        self.h_pool = tf.concat(output_pool, 1) #shape=(-1, 4, 1, 128)

        self.filter_nums_total = self.filter_nums * len(self.filter_sizes)
        self.final_feature = tf.reshape(self.h_pool, shape=[-1, self.filter_nums_total], name='final_feature') # shape=(-1, 4*128)


    def fc_layer(self):
        self.W_fc = tf.get_variable(name='W_fc', shape=[self.filter_nums_total, self.label_nums],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.b_fc = tf.get_variable(name='b_fc', shape=[self.label_nums], initializer=tf.constant_initializer(0.1))

        self.logits = tf.add(tf.matmul(self.final_feature, self.W_fc), self.b_fc)
        self.pred = tf.nn.softmax(self.logits) #shape=(-1, 2)

    def train_op(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y))
        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.basic_learning_rate, global_step, decay_rate=0.99, decay_steps=200, staircase=True)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step)

        self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def build_model(self):
        self.place_holder()
        self.embedding()
        self.conv_layer()
        self.fc_layer()
        self.train_op()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()


    def save_model(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        self.saver.save(self.sess, path)

    def gen_batch(self, data, batch_size):
        data = [np.array(d) for d in data]
        data_size = len(data[0])
        data_idx = np.random.permutation(np.arange(data_size))

        count = 0
        while True:
            if count + batch_size > data_size:
                count = 0
            start = count
            end = start + batch_size
            count = end
            yield data[0][data_idx[start:end]], data[1][data_idx[start:end]]

    def evaluate(self, data):
        data = [np.array(d) for d in data]
        x, y = data[0], data[1]
        # pred = self.sess.run(self.pred, feed_dict={self.input_x:x, self.dropout:1.0})
        # correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        result = self.sess.run(self.accuracy, feed_dict={self.input_x:x, self.input_y:y, self.dropout:1.0})
        return result

    def train(self, train_data, valid_data):
        acc = 0.0
        for e in range(self.epoch):
            for i in range(600):
                x, y = next(self.gen_batch(train_data, self.batch_size))
                loss, logits, _, lr = self.sess.run([self.loss, self.logits, self.train_step, self.learning_rate],
                                                    feed_dict={self.input_x:x, self.input_y:y, self.dropout:self.dropout_prob})

                if i % 100 == 0:
                    acc_test = self.evaluate(valid_data)
                    if acc_test > acc:
                        acc = acc_test
                        print('Accuracy improvement，save model to {}'.format(self.model_path))
                        self.save_model(self.model_path)
                    print('epoch: {}, step: {}, learning_rate: {}, loss: {}, acc: {}'.format(e, i, round(lr, 5), loss, acc))



