from agent_dir.agent import Agent
from collections import deque
import tensorflow as tf
import numpy as np
import random
import os

tf.logging.set_verbosity(tf.logging.INFO)

class DQN:
    
    def __init__(self, hparams):
        self.n_obs = hparams.n_obs
        self.n_actions = hparams.n_actions
        self.gamma = hparams.gamma    # discount factor
        self.learning_rate = hparams.learning_rate
        self.use_dueling = hparams.use_dueling
        self._build()

    def _build(self):
        self._add_placeholder()
        self._add_predictions()
        self._add_saver()


    def _add_placeholder(self):
        states_shape = [None] + self.n_obs
        self.states = tf.placeholder(dtype=tf.float32, shape=states_shape, name="states")
        self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name="actions")
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name="rewards")
        self.next_states = tf.placeholder(dtype=tf.float32, shape=states_shape, name="next_states")


    def _add_predictions(self):
        with tf.variable_scope("eval_net"):
            self.output_eval = self.predict_op(self.states)
        with tf.variable_scope("target_net"):
            self.output_target = self.predict_op(self.next_states)

        self.q_targets = tf.stop_gradient(self.rewards + self.gamma * tf.reduce_max(self.output_target, axis=1),
                                          name="q_targets")
        a_indices = tf.stack([tf.range(tf.shape(self.actions)[0], dtype=tf.int32), self.actions], axis=1)
        self.q_eval_wrt_a = tf.gather_nd(params=self.output_eval, indices=a_indices)    # shape=(None, )

        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_targets, self.q_eval_wrt_a),
                                       name="loss")

        self.global_step = tf.Variable(0, trainable=False)
        with tf.variable_scope("optim"):
#             learning_rate = tf.train.exponential_decay(self.learning_rate, 
#                                                        self.global_step, 
#                                                        decay_steps=1000,
#                                                        decay_rate=0.98,
#                                                        staircase=True)
            self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).\
                            minimize(self.loss, global_step=self.global_step)

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "target_net")
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'eval_net')
        self.assign_ops = [tf.assign(target_w, eval_w) for target_w, eval_w in zip(t_params, e_params)]


    def predict_op(self, inputs):
        out_conv1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[8, 8], strides=[4, 4], 
            padding='valid', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())

        out_conv2 = tf.layers.conv2d(inputs=out_conv1, filters=64, kernel_size=[4, 4], strides=[2, 2], 
            padding='valid', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())

        out_conv3 = tf.layers.conv2d(inputs=out_conv2, filters=64, kernel_size=[3, 3], strides=[1, 1], 
            padding='valid', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        
        flatten_vec = tf.layers.flatten(out_conv3, name="flatten_vec")
        
        out1 = tf.layers.dense(inputs=flatten_vec, units=512, activation=tf.nn.leaky_relu, 
                               kernel_initializer=tf.contrib.layers.xavier_initializer())

        output = tf.layers.dense(inputs=out1, units=self.n_actions,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())

        if self.use_dueling:
            out1_state = tf.layers.dense(inputs=flatten_vec, units=512, activation=tf.nn.leaky_relu, 
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
            out_state = tf.layers.dense(inputs=out1_state, units=1, 
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
            output_action = output - tf.reduce_mean(output, axis=1, keepdims=True)
            output = tf.add(output_action, output_state)

        return output


    def _add_saver(self):
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)


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
        
        _, _step = session.run([self.train_op, self.global_step], 
                               feed_dict={self.states: states, 
                                          self.actions: actions,
                                          self.rewards: rewards,
                                          self.next_states: next_states})

    def cp2targetnet(self, session):
        # 将当前网络参数复制到 target_net 中
        # if (_step > 100) and (_step % 100 == 0):
        tf.logging.info("...... copy parameters ......")
        session.run(self.assign_ops)


    def predict(self, session, states):
        return session.run(self.output_eval, feed_dict={self.states: states})


    def save(self, sess, checkpoint_prefix):
        checkpoint_dir = os.path.dirname(checkpoint_prefix)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, checkpoint_prefix, global_step=self.global_step)


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

        self.hparams = tf.contrib.training.HParams(
            n_actions = self.env.action_space.n,
            total_episode = 100000,
            init_episode = 100,
            exploration_episode = 20000, 
            epsilon_init = 1.0,
            epsilon_min = 0.025,
            n_obs = list(self.env.env.observation_space.shape),
            gamma = 0.95,    # discount factor
            learning_rate = 0.00015,
            use_dueling = False,
            checkpoint_path = "./runs/checkpoints/dqn",
            history_rewards_file = "./history_rewards.npy",
            save_interval = 2000,
            target_update_interval = 1000,
            train_interval = 4,
            replay_size = 10000,
            )

        self.epsilon = self.hparams.epsilon_init
        self.epsilon_decay = (self.hparams.epsilon_init - self.hparams.epsilon_min)/self.hparams.exploration_episode

        self.dqn = DQN(self.hparams)
        self.session = tf.Session()
        if args.test_dqn:
            tf.logging.info('loading trained model')
        self.dqn.load(self.session, self.hparams.checkpoint_path)


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        self.i_episode = -1
        self.history_rewards = []
        self.replay_buf = deque(maxlen=self.hparams.replay_size)


    def train(self):
        """
        Implement your training algorithm here
        """
        tf.logging.info("...... start training ......")
        self.init_game_setting()

        running_reward = None
        i_step = 0
        while self.i_episode < self.hparams.total_episode:
            state = self.env.reset()
            done = False
            
            self.i_episode += 1
            cumulate_reward = 0
            episode_steps = 0
            # playing one game
            while(not done):
                i_step += 1
                episode_steps += 1
                # self.env.env.render()
                action = self.make_action(state, test=False)
                next_state, reward, done, info = self.env.step(action)
                self.replay_buf.append([state, action, reward, next_state])
                state = next_state
                cumulate_reward += reward
            
                if (self.i_episode >= self.hparams.init_episode) \
                        and (i_step % self.hparams.train_interval == 0):
                    self.dqn.train(self.session, self.replay_buf)
                if i_step % self.hparams.target_update_interval == 0:
                    self.dqn.cp2targetnet(self.session)

            running_reward = cumulate_reward if running_reward is None \
                            else running_reward * 0.99 + cumulate_reward * 0.01          
            self.history_rewards.append(running_reward)

            tf.logging.info('I_EPISODE: {:06d} | EPISODE_STEPS: {:03d} | I_STEP: {:09d} | '
                            'EPSILON: {:.5f} | CUR_REWARD: {:2.3f} | AVG_REWARD: {:2.3f}'.format(
                            self.i_episode, episode_steps, i_step, 
                            self.epsilon, cumulate_reward, running_reward))

            if self.i_episode % self.hparams.save_interval == 0:
                self.dqn.save(self.session, self.hparams.checkpoint_path)

        # 记录训练历史.
        np.save(self.hparams.history_rewards_file, self.history_rewards)


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
        # epsilon 逐渐减小
        if self.epsilon > self.hparams.epsilon_min:
            self.epsilon = self.hparams.epsilon_init - self.i_episode * self.epsilon_decay

        if test:
            output = self.dqn.predict(self.session, [observation])
            return np.argmax(output[0])
        else:
            # epsilon-greedy
            if random.random() < self.epsilon:
                return random.randrange(self.hparams.n_actions)
            output = self.dqn.predict(self.session, [observation])
            return np.argmax(output[0])
