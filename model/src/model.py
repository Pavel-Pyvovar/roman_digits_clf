import os
import sys
import numpy as np

sys.path.extend(['..'])

from utils.config import process_config
from utils.utils import get_args

import tensorflow as tf
from tensorflow.layers import (conv2d, max_pooling2d, average_pooling2d, batch_normalization, dropout, dense)
from tensorflow.nn import (relu, softmax, leaky_relu)


class Model():
    """
    Model class represents one object of model.

    :param config: Parsed config file.
    :param session_config: Formed session config file, if necessary.

    :return Model
    """
    
    def __init__(self, config, session_config=None):

        # Configuring session
        self.config = config
        if session_config is not None:
            self.sess = tf.Session(config=session_config)
        else:
            self.sess = tf.Session()

        # Creating inputs to network
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(
                dtype=tf.float32,
                shape=(None, config.image_size, config.image_size, 3))
            self.y = tf.placeholder(dtype=tf.int32, shape=(None, 8))
            self.training = tf.placeholder(dtype=tf.bool, shape=())

        # Creating epoch counter
        self.global_epoch = tf.Variable(
            0, name='global_epoch', trainable=False, dtype=tf.int32)
        self.step = tf.assign(self.global_epoch, self.global_epoch + 1)

        # Building model
        self.__build_model()

        # Summary writer
        self.summ_writer_train = tf.summary.FileWriter(
            config.train_summary_dir, graph=self.sess.graph)
        self.summ_writer_test = tf.summary.FileWriter(config.test_summary_dir)

        self.sess.run(tf.global_variables_initializer())

        # Saver
        self.saver = tf.train.Saver(max_to_keep=1, name='saver')

    def __initialize_local(self):
        """
        Initialize local tensorflow variables.
        
        :return None
        """

        self.sess.run(tf.local_variables_initializer())

    def __block(self,
                inp,
                ch,
                num,
                c_ker=[(3, 3), (3, 3)],
                c_str=[(1, 1), (1, 1)],
                act=relu,
                mp_ker=(2, 2),
                mp_str=(2, 2),
                mode='conc'):
        """
        Create single convolution block of network.
        
        :param inp: Input Tensor of shape (batch_size, inp_size, inp_size, channels).
        :param ch: Number of channels to have in output Tensor.
                   (If mode is 'conc', number of channels will be ch * 2)
        :param num: Number of block for variable scope.
        :param c_ker: List of tuples with shapes of kernels for each convolution operation. 
        :param c_str: List of tuples with shapes of strides for each convolution operation.
        :param act: Activation function.
        :param mp_ker: Tuple-like pooling layer kernel size.
        :param mp_str: Tuple-like pooling layer stride size.
        :param mode: One of ['conc', 'mp', 'ap'] modes, where 'mp' and 'ap' are max- and average- 
                    pooling respectively, and 'conc' - concatenate mode. 
        
        :return Transformed Tensor
        """

        with tf.variable_scope('block_' + str(num)) as name:
            conv1 = conv2d(inp, ch, c_ker[0], strides=c_str[0])
            bn = batch_normalization(conv1)
            out = act(bn)
            if config.use_dropout_block:
                out = dropout(
                    out, config.dropout_rate_block, training=self.training)
#             print(out.shape)

            conv2 = conv2d(out, ch, c_ker[1], strides=c_str[1])
            bn = batch_normalization(conv2)
            out = act(bn)
#             print(out.shape)

            if mode == 'mp':
                out = max_pooling2d(out, mp_ker, strides=mp_str)
            elif mode == 'ap':
                out = average_pooling2d(out, mp_ker, mp_str)
            elif mode == 'conc':
                mp = max_pooling2d(out, mp_ker, strides=mp_str)
                ap = average_pooling2d(out, mp_ker, mp_str)
                out = tf.concat([mp, ap], -1)
            else:
                raise ValueError('Unknown mode.')

#             print(out.shape)
        return out

    def __build_model(self):
        """
        Build model.
        
        :return None
        """
        
        with tf.name_scope('layers'):
            out = self.__block(self.x, 16, 1, mode='conc')
            out = self.__block(out, 32, 2, mode='conc')
            out = self.__block(out, 64, 3, mode='conc')
            out = self.__block(out, 256, 4, c_str=[(1, 1), (2, 2)], mode='mp')

            dim = np.prod(out.shape[1:])
            out = tf.reshape(out, [-1, dim])
#             print(out.shape)

            with tf.variable_scope('dense') as scope:
                dense_l = dense(out, 128)
                out = batch_normalization(dense_l)
                out = leaky_relu(out, alpha=0.01)
                if config.use_dropout_dense:
                    out = dropout(
                        out,
                        rate=config.dropout_rate_dense,
                        training=self.training)
#                 print(out.shape)

                self.predictions = dense(out, 8, activation=softmax)

        with tf.name_scope('metrics'):
            amax_labels = tf.argmax(self.y, 1)
            amax_pred = tf.argmax(self.predictions, 1)

            cur_loss = tf.losses.softmax_cross_entropy(self.y,
                                                       self.predictions)
            self.loss, self.loss_update = tf.metrics.mean(cur_loss)

            cur_acc = tf.reduce_mean(
                tf.cast(tf.equal(amax_labels, amax_pred), dtype=tf.float32))
            self.acc, self.acc_update = tf.metrics.mean(cur_acc)

            self.optimize = tf.train.AdamOptimizer(
                self.config.learning_rate).minimize(cur_loss)

            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.acc)

        self.summary = tf.summary.merge_all()

    def train(self, dat, epochs, dat_v=None, batch=None):
        """
        Train model on data.
        
        :param dat: List of data to train on, like [X, y].
                    Where X is an array with size (None, image_size, image_size, 3) and
                    y is an array with size (None ,8).
        :param epochs: Number of epochs to run training procedure.
        :param dat_v: List of data to validate on, like [X, y].
                    Where X is an array with size (None, image_size, image_size, 3) and
                    y is an array with size (None ,8).
        :param batch: Batch size to train on.
        
        :return None
        """
        
        if batch is not None:
            steps = int(np.ceil(dat[0].shape[0] / batch))
        else:
            batch = dat[0].shape[0]
            steps = 1

        for epoch in range(epochs):
            self.__initialize_local()
            summary = tf.summary.Summary()

            for step in range(steps):
                start = step * batch
                finish = (
                    step + 1) * batch if step + 1 != steps else dat[0].shape[0]

                _, _, _ = self.sess.run(
                    [self.loss_update, self.acc_update, self.optimize],
                    feed_dict={
                        self.x: dat[0][start:finish],
                        self.y: dat[1][start:finish],
                        self.training: True
                    })

            summary, loss, acc, ep = self.sess.run(
                [self.summary, self.loss, self.acc, self.step])
            self.summ_writer_train.add_summary(summary, ep)
            print(
                'EP: {:3d}\tLOSS: {:.10f}\tACC: {:.10f}\t'.format(
                    ep, loss, acc),
                end='')

            if dat_v is not None:
                self.test(dat_v, batch=batch)
            else:
                print()

    def test(self, dat, batch=None):
        """
        Test model on specific data.
        
        :param dat: List of data to test on, like [X, y].
                    Where X is an array with size (None, image_size, image_size, 3) and
                    y is an array with size (None ,8)..
        :param batch: Batch size to use.
        
        :return None
        """
        
        if batch is not None:
            steps = int(np.ceil(dat[0].shape[0] / batch))
        else:
            steps = 1
            batch = dat[0].shape[0]

        self.__initialize_local()
        for step in range(steps):
            start = step * batch
            finish = (
                step + 1) * batch if step + 1 != steps else dat[0].shape[0]

            _, _ = self.sess.run(
                [self.loss_update, self.acc_update],
                feed_dict={
                    self.x: dat[0][start:finish],
                    self.y: dat[1][start:finish],
                    self.training: False
                })

        summary, loss, acc, ep = self.sess.run(
            [self.summary, self.loss, self.acc, self.global_epoch])
        self.summ_writer_test.add_summary(summary, ep)
        print('VALID\tLOSS: {:.10f}\tACC: {:.10f}'.format(loss, acc))

    def predict_proba(self, data, batch=None):
        """
        Predict probability of each class.
        
        :param data: An array to predict on with shape (None, image_size, image_size, 3)
        :param batch: Batch size to use.
        
        :return Array of predictions with shape (None, 8)
        """
        
        if batch is not None:
            steps = int(np.ceil(data.shape[0] / batch))
        else:
            steps = 1
            batch = data.shape[0]

        self.__initialize_local()

        preds_arr = []
        for step in range(steps):
            start = step * batch
            finish = (step + 1) * batch if step + 1 != steps else data.shape[0]

            preds = self.sess.run(
                self.predictions,
                feed_dict={
                    self.x: data[start:finish],
                    self.y: np.zeros((finish - start, 8)),
                    self.training: False
                })
            preds_arr.append(preds)

        return np.concatenate(preds_arr)

    def save_model(self, model_path=None):
        """
        Save model weights to the folder with weights.
        
        :param model_path: String-like path to save in.
        
        :return None
        """
        
        gstep = self.sess.run(self.global_epoch)
        if model_path is not None:
            self.saver.save(self.sess, model_path + 'model')
        else:
            self.saver.save(self.sess, config.checkpoint_dir + 'model')

    def load_model(self, model_path=None):
        """
        Load model weights.
        
        :param model_path: String-like path to load from.
        
        :return None
        """
        
        if model_path is not None:
            meta = [
                os.path.join(filename) for filename in os.listdir(model_path)
                if filename.endswith('.meta')
            ][0]
            self.saver = tf.train.import_meta_graph(
                os.path.join(model_path, meta))
            self.saver.restore(self.sess,
                               tf.train.latest_checkpoint(model_path))
        else:
            meta = [
                os.path.join(filename)
                for filename in os.listdir(self.config.checkpoint_dir)
                if filename.endswith('.meta')
            ][0]
            self.saver = tf.train.import_meta_graph(
                os.path.join(self.config.checkpoint_dir, meta))
            self.saver.restore(
                self.sess,
                tf.train.latest_checkpoint(self.config.checkpoint_dir))

    def close(self):
        """
        Close a session of model to load next one.
        
        :retun None
        """
        
        self.sess.close()
        tf.reset_default_graph()


def main(config):
    m = Model(config)
    m.load_model()

    # m.predict_proba()
    m.close()

if __name__ == "__main__":
    try:
        args = get_args()
        config = process_config(args.config)
        main(config)

    except Exception as e:
        print("missing or invalid arguments")
        print(e)
        exit(0)