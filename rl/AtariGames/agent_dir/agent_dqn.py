from agent_dir.agent import Agent
from collections import deque
import tensorflow as tf
import numpy as np
import random

tf.logging.set_verbosity(tf.logging.INFO)

class DQN:
    
    def __init__(self):
        self.n_obs = [80, 80, 4]
        self.n_actions = 3
        self.gamma = 0.9    # discount factor
        self._build()

    def _build(self):
        self._add_placeholder()
        self._add_predictions()
        self._add_saver()


    def _add_placeholder(self):
        self.states = tf.placeholder(dtype=tf.float32, shape=[None, 84, 84, 4], name="states")
        # self.actions = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions], name="actions")
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name="rewards")
        self.next_states = tf.placeholder(dtype=tf.float32, shape=[None, 84, 84, 4], name="next_states")
        
        self.q_targets = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions], name="q_target")


    def _add_predictions(self):
        with tf.variable_scope("eval_net"):
            self.output_eval = self.predict_op(self.states, name="output_eval")
        with tf.variable_scope("target_net"):
            self.output_target = self.predict_op(self.next_states, name="output_target")

        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.output_eval - self.q_targets), axis=1))
        
        self.global_step = tf.Variable(0, trainable=False)
        with tf.variable_scope("optim"):
            learning_rate = tf.train.exponential_decay(1e-3, 
                                                       self.global_step, 
                                                       decay_steps=1000,
                                                       decay_rate=0.98,
                                                       staircase=True)    
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate).\
                            minimize(self.loss, global_step=self.global_step)


        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "target_net")
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'eval_net')
        self.assign_ops = [tf.assign(target_w, eval_w) for target_w, eval_w in zip(t_params, e_params)]


    def predict_op(self, input, name):
        with tf.variable_scope("conv-1"):
            filter1 = tf.get_variable(name="filter1", shape=[3, 3, 4, 64], 
                                      initializer=tf.contrib.layers.xavier_initializer())
            out_conv1 = tf.nn.conv2d(input=input, filter=filter1, 
                                     strides=[1, 2, 2, 1], padding="SAME")
            out_pool1 = tf.nn.max_pool(value=out_conv1, ksize=[1, 2, 2, 1], 
                                       strides=[1, 2, 2, 1], padding="SAME")

        with tf.variable_scope("conv-2"):
            filter2 = tf.get_variable(name="filter2", shape=[3, 3, 64, 64], 
                                      initializer=tf.contrib.layers.xavier_initializer())
            out_conv2 = tf.nn.conv2d(input=out_pool1, filter=filter2, 
                                     strides=[1, 2, 2, 1], padding="SAME")
            out_pool2 = tf.nn.max_pool(value=out_conv2, ksize=[1, 2, 2, 1], 
                                       strides=[1, 2, 2, 1], padding="SAME")

        with tf.variable_scope("out-layer"):
            flatten_vec = tf.layers.flatten(out_pool2, name="flatten_vec")
            weights = tf.get_variable(name="weights", shape=[flatten_vec.shape[-1], self.n_actions], 
                                      initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable(name="bias", shape=[self.n_actions], initializer=tf.zeros_initializer())
            output = tf.add(tf.matmul(flatten_vec, weights), bias, name=name)

        return output


    def _add_saver(self):
        pass


    def train(self, session, replay_buf, batch_size=32):
        mini_batch = random.sample(replay_buf, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        for sample in mini_batch:
            states.append(sample[0])
            actions.append(sample[1])
            rewards.append(sample[2])
            next_states.append(sample[3])
        states = np.array(states)       # shape: [batch_size, 80, 80, 4]
        actions = np.array(actions)     # shape: [batch_size]
        rewards = np.array(rewards)     # shape: [batch_size]
        next_states = np.array(next_states)     # shape: [batch_size, 80, 80, 4]
        
        # shape: [batch_size, n_actions]
        out_eval, out_target = session.run([self.output_eval, self.output_target], 
                                           feed_dict={self.states: states, 
                                                      self.next_states: next_states})
        q_targets = out_eval.copy()
        q_targets[:, actions] = rewards + self.gamma * np.max(out_target, axis=1)
        _, _step = session.run([self.train_op, self.global_step], 
                               feed_dict={self.states: states, 
                                          # self.actions: actions,
                                          self.rewards: rewards,
                                          self.q_targets: q_targets})

        # 将当前网络参数复制到 target_net 中
        if (_step > 100) and (_step % 100 == 0):
            session.run(self.assign_ops)


    def predict(self, session, states):
        return session.run(self.output_eval, feed_dict={self.states: states})


    def load(self, sess, checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            tf.logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            tf.logging.info("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)

        self.n_actions = self.env.action_space.n
        self.dqn = DQN()
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')

        ##################
        # YOUR CODE HERE #
        ##################


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.session = tf.Session()
        self.dqn.load(self.session, "./runs/checkpoints")


    def train(self):
        """
        Implement your training algorithm here
        """
        tf.logging.info("...... start training ......")
        self.init_game_setting()
        i_episode = -1
        running_reward = None
        
        replay_buf = deque(maxlen=1000)

        while True:
            i_episode += 1
            state = self.env.reset()
            done = False
            
            cumulate_reward = 0
            #playing one game
            while(not done):
                # self.env.env.render()
                action = self.make_action(state, test=False)
                next_state, reward, done, info = self.env.step(action)
                replay_buf.append([state, action, reward, next_state])
                state = next_state
                cumulate_reward += reward
            
            if i_episode > 10:
                self.dqn.train(self.session, replay_buf)
            # self.model.train(self.sess, episode_states, episode_actions, discount_rewards)

            running_reward = cumulate_reward if running_reward is None else running_reward * 0.99 + cumulate_reward * 0.01            
            if i_episode % 10 == 0:
                tf.logging.info('ep {}: reward: {}, mean reward: {:3f}'.format(i_episode, cumulate_reward, running_reward))
            else:
                tf.logging.info('\tep {}: reward: {}'.format(i_episode, cumulate_reward))


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        output = self.dqn.predict(self.session, [observation])
        return np.argmax(output[0])
        return self.env.get_random_action()

