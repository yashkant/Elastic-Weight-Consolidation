import tensorflow as tf
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, mean=0.0, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class Model:

    def __init__(self, x, y_):
        self.in_dim = int(x.get_shape()[1])
        self.out_dim = int(y_.get_shape()[1])
        self.x = x # This is a placeholder that'll be used to feed data
        # layer-1
        self.w1 = weight_variable([self.in_dim, 50])
        self.b1 = bias_variable([50])

        # layer-2
        self.w2 = weight_variable([50, self.out_dim])
        self.b2 = weight_variable([self.out_dim])

        # So, this variable could be used to access weights and biases outside the class!
        self.var_list = [self.w1, self.b1, self.w2, self.b2]

        # build-graph
        self.h1 = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
        self.y = tf.matmul(self.h1, self.w2) + self.b2
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y, labels=y_))
        self.vanilla_loss()
        self.correct_preds = (tf.equal(tf.argmax(self.y, axis=1), tf.argmax(y_, axis=1)))

        # accuracy
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_preds, tf.float32))

    # set vanilla loss
    def vanilla_loss(self):
        self.train_step = tf.train.GradientDescentOptimizer(.1).minimize(self.cross_entropy)
        
    def compute_fisher(self, imgset, sess, num_samples=200, plot_diffs=False, disp_freq=10):
            # computer Fisher information for each parameter

            # initialize Fisher information for most recent task
            self.fisher = []
            for v in range(len(self.var_list)):
                self.fisher.append(np.zeros(self.var_list[v].get_shape().as_list()))

            # sampling a random class from softmax
            probs = tf.nn.softmax(self.y)
            class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])

            if(plot_diffs):
                # track differences in mean Fisher info
                prev_fisher = deepcopy(self.fisher)
                mean_diffs = np.zeros(0)

            for i in range(num_samples):
                # select random input image
                im_ind = np.random.randint(imgset.shape[0])
                # compute first-order derivatives
                ders = sess.run(tf.gradients(tf.log(probs[0,class_ind]), self.var_list), feed_dict={self.x: imgset[im_ind:im_ind+1]})
                # square the derivatives and add to total
                for v in range(len(self.fisher)):
                    self.fisher[v] += np.square(ders[v])
                if(plot_diffs):
                    if i % disp_freq == 0 and i > 0:
                        # recording mean diffs of F
                        F_diff = 0
                        for v in range(len(self.fisher)):
                            F_diff += np.sum(np.absolute(self.fisher[v]/(i+1) - prev_fisher[v]))
                        mean_diff = np.mean(F_diff)
                        mean_diffs = np.append(mean_diffs, mean_diff)
                        for v in range(len(self.fisher)):
                            prev_fisher[v] = self.fisher[v]/(i+1)
                        plt.plot(range(disp_freq+1, i+2, disp_freq), mean_diffs)
                        plt.xlabel("Number of samples")
                        plt.ylabel("Mean absolute Fisher difference")
                        display.display(plt.gcf())
                        display.clear_output(wait=True)

            # divide totals by number of samples
            for v in range(len(self.fisher)):
                self.fisher[v] /= num_samples

    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []

        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())  # without sess ??

    def restore(self, sess):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                sess.run(self.var_list[v].assign(self.star_vars[v]))

    # having doubts with the graph here! I think let's try tensor-board maybe :D
    def set_ewc_loss(self, lam):
        if not hasattr(self, "ewc_loss"):
            self.ewc_loss = self.cross_entropy

        # I feel in the third step loss the values of fisher and star_vars would change for the
        # earlier task as well! Does the graph save values of the vars as well ?
        for v in range(len(self.var_list)):
            self.ewc_loss += (lam / 2) * tf.reduce_sum(tf.multiply(self.fisher[v].astype(np.float32),
                                                                   tf.square(self.var_list[v] - self.star_vars[v])))

        self.train_step = tf.train.GradientDescentOptimizer(.1).minimize(self.ewc_loss)
