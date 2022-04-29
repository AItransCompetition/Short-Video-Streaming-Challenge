# coding=utf-8
import numpy as np
import tensorflow as tf
__all__ = [tf]
import tflearn


GAMMA = 0.9
#A_DIM = 6
#A_DIM = 12
ENTROPY_WEIGHT = 0.15 #0.5
ENTROPY_WEIGHT_DECAY = 0.9998
# ENTROPY_WEIGHT = 1.0
ENTROPY_EPS = 1e-6
#S_INFO = 4
leaky = 0.2


class ActorNetwork(object):
    """
    Input to the network is the state, output is the distribution
    of all actions.
    """
    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.entropy_weight = ENTROPY_WEIGHT

        # Create the actor network
        self.inputs, self.out = self.create_actor_network()

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Selected action, 0-1 vector
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])

        # This gradient will be provided by the critic network
        self.act_grad_weights = tf.placeholder(tf.float32, [None, 1])

        # Compute the objective (log action_vector and entropy)
        self.obj = tf.reduce_sum(tf.multiply(tf.math.log(tf.reduce_sum(tf.multiply(self.out, self.acts),
                    reduction_indices=1, keep_dims=True)), - self.act_grad_weights)) \
                    + self.entropy_weight * tf.reduce_sum(tf.multiply(self.out, tf.math.log(self.out + ENTROPY_EPS)))
                       #+ self.entropy_weight * tf.reduce_sum(tf.multiply(self.out, tf.log(self.out + ENTROPY_EPS)))
        #r_batch
        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.obj, self.network_params)

        # Optimization Op
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate,momentum = 0.2,decay = 0.9).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

    def create_actor_network(self):
        with tf.variable_scope('actor'):
            # old version#
            # inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])
            # reverse_data = tf.transpose(inputs,[0,2,1])
            #
            # gru_net0 = tflearn.activations.leaky_relu(tflearn.layers.recurrent.gru(reverse_data, 16,activation = 'linear'),alpha=leaky)
            # dense_net_0 = tflearn.activations.leaky_relu(tflearn.fully_connected(gru_net0, 32, activation = 'linear'),alpha=leaky)
            # out = tflearn.fully_connected(dense_net_0, self.a_dim, activation='softmax')

            #SITI version#
            inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])

            split_0 = tf.transpose(inputs[:,0:3,:],[0,2,1])
            split_0_gru = tflearn.activations.leaky_relu(tflearn.layers.recurrent.gru(split_0, 16,activation = 'linear'),alpha=leaky)
            split_0_flat = tflearn.flatten(split_0_gru)

            split_1= tf.transpose(inputs[:,3:7,:],[0,2,1])
            split_1_gru = tflearn.activations.leaky_relu(tflearn.layers.recurrent.gru(split_1, 16,activation = 'linear'),alpha=leaky)
            split_1_flat = tflearn.flatten(split_1_gru)

            merge_net = tf.stack([split_0_flat, split_1_flat], axis=-1)
            # conv_net = out_1 = tflearn.conv_1d(merge_net, 32, 4, activation='relu')
            dense_net = tflearn.activations.leaky_relu(tflearn.fully_connected(merge_net, 32, activation = 'linear'),alpha=leaky)
            out = tflearn.fully_connected(dense_net, self.a_dim, activation='softmax')

            return inputs, out

    def train(self, inputs, acts, act_grad_weights):

        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_gradients(self, inputs, acts, act_grad_weights):
        return self.sess.run(self.actor_gradients, feed_dict={
            self.inputs: inputs,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def apply_gradients(self, actor_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.actor_gradients, actor_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })

    def set_entropy_weight(self):
        self.entropy_weight *= ENTROPY_WEIGHT_DECAY
        if self.entropy_weight < 0.1: self.entropy_weight = 0.1
        return self.entropy_weight

class CriticNetwork(object):
    """
    Input to the network is the state and action, output is V(s).
    On policy: the action must be obtained from the output of the Actor network.
    """
    def __init__(self, sess, state_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.lr_rate = learning_rate

        # Create the critic network
        self.inputs, self.out = self.create_critic_network()

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Network target V(s)
        self.td_target = tf.placeholder(tf.float32, [None, 1])

        # Temporal Difference, will also be weights for actor_gradients
        self.td = tf.subtract(self.td_target, self.out)

        # Mean square error
        self.loss = tflearn.mean_square(self.td_target, self.out)

        # Compute critic gradient
        self.critic_gradients = tf.gradients(self.loss, self.network_params)

        # Optimization Op
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate,momentum=0.2,decay = 0.9).\
            apply_gradients(zip(self.critic_gradients, self.network_params))

    def create_critic_network(self):
        with tf.variable_scope('critic'):
            #old version#
            # inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])
            #
            # reverse_data = tf.transpose(inputs,[0,2,1])
            #
            # gru_net0 = tflearn.activations.leaky_relu(tflearn.layers.recurrent.gru(reverse_data,16,activation = 'linear'),alpha=leaky)
            #
            # dense_net_0 = tflearn.activations.leaky_relu(tflearn.fully_connected(gru_net0, 32, activation = 'linear'),alpha=leaky)
            # dense_net_1 = tflearn.fully_connected(dense_net_0, 6, activation='linear')
            #
            #
            # out = tf.reduce_max(dense_net_1,reduction_indices = [1])

            #SITI version1#
            inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])

            split_0 = tf.transpose(inputs[:,0:3,:],[0,2,1])
            split_0_gru = tflearn.activations.leaky_relu(tflearn.layers.recurrent.gru(split_0, 16,activation = 'linear'),alpha=leaky)
            split_0_flat = tflearn.flatten(split_0_gru)

            split_1= tf.transpose(inputs[:,3:7,:],[0,2,1])
            split_1_gru = tflearn.activations.leaky_relu(tflearn.layers.recurrent.gru(split_1, 16,activation = 'linear'),alpha=leaky)
            split_1_flat = tflearn.flatten(split_1_gru)

            merge_net = tf.stack([split_0_flat, split_1_flat], axis=-1)
            # conv_net = tflearn.conv_1d(merge_net, 32, 4, activation='relu')
            dense_net_0 = tflearn.activations.leaky_relu(tflearn.fully_connected(merge_net, 32, activation = 'linear'),alpha=leaky)
            dense_net_1 = tflearn.fully_connected(dense_net_0, 6, activation='linear')
            out = tf.reduce_max(dense_net_1,reduction_indices = [1])

            return inputs, out

    def train(self, inputs, td_target):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_td(self, inputs, td_target):
        return self.sess.run(self.td, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def get_gradients(self, inputs, td_target):
        return self.sess.run(self.critic_gradients, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def apply_gradients(self, critic_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.critic_gradients, critic_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })

#前向TD_loss
def compute_gradients(s_batch, a_batch, r_batch, terminal, actor, critic):
    """
    batch of s, a, r is from samples in a sequence
    the format is in np.array([batch_size, s/a/r_dim])
    terminal is True when sequence ends as a terminal state
    """
    assert s_batch.shape[0] == a_batch.shape[0]
    assert s_batch.shape[0] == r_batch.shape[0]
    ba_size = s_batch.shape[0]


    ba_batch = critic.predict(s_batch)
    #za_size = z_batch.shape[0]

    #v_batch = np.zeros(r_batch.shape)
    #for i in range(za_size):
    #     v_batch[i,0] = max(z_batch[i,:])
    #    v_batch[i,0] = z_batch[i,0] \
    #                   + pow(z_batch[i,1],2) \
    #                   + pow(z_batch[i,2],3) \
    #                   + pow(z_batch[i,3],4)

    R_batch = np.zeros(r_batch.shape)
    v_batch = np.zeros(r_batch.shape)
    for i in range(ba_size):
        v_batch[i, 0] = ba_batch[i]

    if terminal:
        R_batch[-1, 0] = 0  # terminal state
    else:
        R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state

    for t in range(ba_size - 1):
        n = ba_size - t
        if(n > 8):
            n = 8
        temp = np.zeros(n)
        temp[-1] = r_batch[t + n - 1]

        for i in reversed(range(n - 1)):
            temp[i] = (1 - GAMMA)*r_batch[t + i] + GAMMA * temp[i + 1]

        R_batch[t, 0] = temp[0]

    td_batch = R_batch - v_batch

    actor_gradients = actor.get_gradients(s_batch, a_batch, td_batch)
    critic_gradients = critic.get_gradients(s_batch, R_batch)

    return actor_gradients, critic_gradients, td_batch

#后向TD_loss
# def compute_gradients(s_batch, a_batch, r_batch, terminal, actor, critic):#计算累计奖励函数值的梯度
#     """
#     batch of s, a, r is from samples in a sequence
#     the format is in np.array([batch_size, s/a/r_dim])
#     terminal is True when sequence ends as a terminal state
#     """
#     assert s_batch.shape[0] == a_batch.shape[0]
#     assert s_batch.shape[0] == r_batch.shape[0]
#     ba_size = s_batch.shape[0]
#     ba_batch = critic.predict(s_batch)
#     R_batch = np.zeros(r_batch.shape)
#     v_batch = np.zeros(r_batch.shape)
#     for i in range(ba_size):
#         v_batch[i, 0] = ba_batch[i]
#     if terminal:
#         R_batch[-1, 0] = 0  # terminal state
#     else:
#         R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state
#
#     for t in reversed(range(ba_size - 1)):
#         R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]
#
#     td_batch = R_batch - v_batch
#
#     actor_gradients = actor.get_gradients(s_batch, a_batch, td_batch)
#     critic_gradients = critic.get_gradients(s_batch, R_batch)
#
#     return actor_gradients, critic_gradients, td_batch

def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out


def compute_entropy(x):
    """
    Given vector x, computes the entropy
    H(x) = - sum( p * log(p))
    """
    H = 0.0
    for i in range(len(x)):
        if 0 < x[i] < 1:
            H -= x[i] * np.log2(x[i])
    return H


def build_summaries():
    td_loss = tf.Variable(0.)
    tf.summary.scalar("TD_loss", td_loss)
    eps_total_reward = tf.Variable(0.)
    tf.summary.scalar("Eps_total_reward", eps_total_reward)
    min_reward = tf.Variable(0.)
    tf.summary.scalar("Min_reward",min_reward)
    avg_entropy = tf.Variable(0.)
    tf.summary.scalar("Avg_entropy", avg_entropy)

    summary_vars = [td_loss, eps_total_reward,min_reward, avg_entropy]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars
