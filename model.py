import tensorflow as tf
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, mean=0.0, stddev=1.0)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class Model:

    def __init__(self, x, y_):
        self.in_dim = int(x.get_shape()[1])
        self.out_dim = int(y_.get_shape()[1])

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
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=y_))
        self.train_step = None
        self.fisher = None
        self.vanilla_loss()
        self.correct_preds = (tf.equal(tf.argmax(self.y, axis=1), tf.argmax(y_, axis=1)))

        # accuracy
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_preds, tf.float32))

    # set vanilla loss
    def vanilla_loss(self):
        self.train_step = tf.train.GradientDescentOptimizer(.1).minimize(self.cross_entropy)

    # computes fisher
    def compute_fisher(self, imgset, sess, num_samples=200, plot_diffs=False, disp_freq=10):
        self.fisher = []
        for v in self.var_list:
            self.fisher.append(np.zeros(v.get_shape().as_list()))

        probs = tf.nn.softmax(self.y)
        class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])  # doubtful here, see usage somewhere!

        if plot_diffs:
            fisher_prev = deepcopy(self.fisher)
            mean_diffs = np.zeros(0)

        for i in range(num_samples):
            im_ind = np.random.randint(imgset.shape[0])
            # try eval here!
            dervs = sess.run(tf.gradients(tf.log(probs[0, class_ind]), self.var_list),
                             feed_dict={self.x: imgset[im_ind:im_ind + 1]})

            for j in range(len(self.var_list)):
                self.fisher[j] += np.square(dervs[j])

            if plot_diffs:
                if i%disp_freq == 0 and i > 0:
                    fisher_diff = 0

                    for k in range(len(self.fisher)):
                        fisher_diff += np.sum(np.absolute(self.fisher[k]/(i+1) - fisher_prev[k]))
                        mean_diff = np.mean(fisher_diff) #makes no sense to me!
                        mean_diffs = np.append(mean_diffs, mean_diff)

                    # replace previous
                    for l in range(len(fisher_prev)):
                        fisher_prev[v] = self.fisher[l]

                    plt.plot(range(disp_freq+1, i+2, disp_freq), mean_diffs)
                    plt.xlabel("Number of samples")
                    plt.ylabel("Mean absolute Fisher difference")
                    display.display(plt.gcf())
                    display.clear_output(wait=True)

        for i in range(len(self.fisher)):
            self.fisher[v] /= num_samples

    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []

        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval()) # without sess ??

    def restore(self, sess):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                sess.run(self.var_list[v].assign(self.star_vars[v]))
