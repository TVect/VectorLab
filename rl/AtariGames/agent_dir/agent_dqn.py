from agent_dir.agent import Agent
from collections import deque
import tensorflow as tf
import numpy as np
import random
import os

tf.logging.set_verbosity(tf.logging.INFO)


class QNetwork:
    
    def __init__(self, hparams, name):
        self.name = name
        
        self.n_obs = hparams.n_obs
        self.n_actions = hparams.n_actions
        self.gamma = hparams.gamma    # discount factor
        self.learning_rate = hparams.learning_rate
        self.use_dueling = hparams.use_dueling
        
        self.init_layers()
    
    
    def init_layers(self):
        self.conv_layer1 = tf.layers.Conv2D(filters=32, kernel_size=[8, 8], strides=[4, 4], padding='valid', 
                        activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.conv_layer2 = tf.layers.Conv2D(filters=64, kernel_size=[4, 4], strides=[2, 2], padding='valid', 
                        activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.conv_layer3 = tf.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding='valid', 
                        activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.flatten_layer = tf.layers.Flatten()
        
        self.action_dense_layer = tf.layers.Dense(units=512, activation=tf.nn.leaky_relu, 
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.action_out_layer = tf.layers.Dense(units=self.n_actions, 
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())

        if self.use_dueling:
            self.state_dense_layer = tf.layers.Dense(units=512, activation=tf.nn.leaky_relu, 
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.state_out_layer = tf.layers.Dense(units=1, 
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())

    
    def __call__(self, inputs):
        with tf.variable_scope(self.name):
            conv1 = self.conv_layer1.apply(inputs)
            conv2 = self.conv_layer2.apply(conv1)
            conv3 = self.conv_layer3.apply(conv2)
            flatten_vec = self.flatten_layer.apply(conv3)
            action_dense = self.action_dense_layer.apply(flatten_vec)
            action_out = self.action_out_layer.apply(action_dense)
            
            if self.use_dueling:
                state_dense = self.state_dense_layer.apply(flatten_vec)
                state_out = self.state_out_layer.apply(state_dense)
                return state_out + action_out - tf.reduce_mean(action_out, axis=1, keepdims=True)
    
            return action_out


class DQN:
    
    def __init__(self, hparams):
        self.hparams = hparams
        self.n_obs = hparams.n_obs
        self.n_actions = hparams.n_actions
        self.gamma = hparams.gamma    # discount factor
        self.learning_rate = hparams.learning_rate
        self.use_dueling = hparams.use_dueling
        self.use_ddqn = hparams.use_ddqn
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
        self.dones = tf.placeholder(dtype=tf.int32, shape=[None], name="dones")


    def _add_predictions(self):
        self.eval_net = QNetwork(self.hparams, name="eval_net")
        self.target_net = QNetwork(self.hparams, name="target_net")

        self.eval_output = self.eval_net(self.states)
        self.target_output = self.target_net(self.next_states)

        if self.use_ddqn:
            # 用 eval_net 选取最大的 action, 用 target_net 评估 action 对应的 value 值.
            eval_nextstates = self.eval_net(self.next_states)
            eval_argmax = tf.argmax(eval_nextstates, axis=1, output_type=tf.int32)
            eval_argmax_indices = tf.stack([tf.range(tf.shape(self.actions)[0], dtype=tf.int32), eval_argmax],
                                           axis=1)
            target_vals = tf.gather_nd(params=self.target_output, indices=eval_argmax_indices)
            self.q_targets = tf.stop_gradient(
                self.rewards + self.gamma * target_vals * tf.cast(1-self.dones, tf.float32),
                name="q_targets")
        else:
            target_vals = tf.reduce_max(self.target_output, axis=1)
            self.q_targets = tf.stop_gradient(
                self.rewards + self.gamma * target_vals * tf.cast(1-self.dones, tf.float32), 
                name="q_targets")

        a_indices = tf.stack([tf.range(tf.shape(self.actions)[0], dtype=tf.int32), self.actions], axis=1)
        self.q_eval_wrt_a = tf.gather_nd(params=self.eval_output, indices=a_indices)    # shape=(None, )

        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_targets, self.q_eval_wrt_a),
                                       name="loss")

        self.global_step = tf.Variable(0, trainable=False)
        with tf.variable_scope("optim"):
#             learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, decay_steps=1000, decay_rate=0.98)
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).\
                            minimize(self.loss, global_step=self.global_step)

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "target_net")
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'eval_net')
        self.assign_ops = [tf.assign(target_w, eval_w) for target_w, eval_w in zip(t_params, e_params)]

    def _add_saver(self):
        # self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'eval_net'), 
                                    max_to_keep=10)


    def train(self, session, replay_buf, batch_size=32):
        mini_batch = random.sample(replay_buf, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for sample in mini_batch:
            states.append(sample[0])
            actions.append(sample[1])
            rewards.append(sample[2])
            next_states.append(sample[3])
            dones.append(sample[4])
        states = np.array(states)       # shape: [batch_size, 80, 80, 4]
        actions = np.array(actions)     # shape: [batch_size]
        rewards = np.array(rewards)     # shape: [batch_size]
        next_states = np.array(next_states)     # shape: [batch_size, 80, 80, 4]
        
        _, _step = session.run([self.train_op, self.global_step], 
                               feed_dict={self.states: states, 
                                          self.actions: actions,
                                          self.rewards: rewards,
                                          self.next_states: next_states,
                                          self.dones: dones})

    def cp2targetnet(self, session):
        # 将当前网络参数复制到 target_net 中
        # if (_step > 100) and (_step % 100 == 0):
        tf.logging.info("...... copy parameters ......")
        session.run(self.assign_ops)


    def predict(self, session, states):
        return session.run(self.eval_output, feed_dict={self.states: states})


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
            init_step = 10000,
            exploration_step = 1000000, 
            epsilon_init = 1.0,
            epsilon_min = 0.025,
            n_obs = list(self.env.env.observation_space.shape),
            gamma = 0.99,    # discount factor
            learning_rate = 0.00015,
            use_dueling = False,
            use_ddqn = False,
            checkpoint_path = "./checkpoints/agent_dqn/dqn",
            history_rewards_file = "./history_rewards.npy",
            save_interval = 100000,
            target_update_interval = 1000,
            train_interval = 4,
            replay_size = 10000,
            )

        self.epsilon = self.hparams.epsilon_init
        self.epsilon_decay = (self.hparams.epsilon_init - self.hparams.epsilon_min)/self.hparams.exploration_step

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
        self.i_step = -1
        self.history_rewards = []
        self.replay_buf = deque(maxlen=self.hparams.replay_size)


    def train(self):
        """
        Implement your training algorithm here
        """
        tf.logging.info("...... start training ......")
        self.init_game_setting()

        running_reward = None
        for i_episode in range(self.hparams.total_episode):
            state = self.env.reset()
            done = False

            cumulate_reward = 0
            episode_steps = 0
            # playing one game
            while(not done):
                self.i_step += 1
                episode_steps += 1
                # self.env.env.render()
                action = self.make_action(state, test=False)
                next_state, reward, done, info = self.env.step(action)
                self.replay_buf.append([state, action, reward, next_state, int(done)])
                state = next_state
                cumulate_reward += reward
            
                if (self.i_step >= self.hparams.init_step) \
                        and (self.i_step % self.hparams.train_interval == 0):
                    self.dqn.train(self.session, self.replay_buf)
                if (self.i_step >= self.hparams.init_step) \
                        and (self.i_step % self.hparams.target_update_interval == 0):
                    self.dqn.cp2targetnet(self.session)

            running_reward = cumulate_reward if running_reward is None \
                            else running_reward * 0.99 + cumulate_reward * 0.01          
            self.history_rewards.append(running_reward)

            tf.logging.info('I_EPISODE: {:06d} | EPISODE_STEPS: {:03d} | I_STEP: {:09d} | '
                            'EPSILON: {:.5f} | CUR_REWARD: {:2.3f} | AVG_REWARD: {:2.3f}'.format(
                            i_episode, episode_steps, self.i_step, 
                            self.epsilon, cumulate_reward, running_reward))

            if self.i_step % self.hparams.save_interval == 0:
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
            self.epsilon = self.epsilon - self.epsilon_decay

        if test:
            output = self.dqn.predict(self.session, [observation])
            return np.argmax(output[0])
        else:
            # epsilon-greedy
            if random.random() < self.epsilon:
                return random.randrange(self.hparams.n_actions)
            output = self.dqn.predict(self.session, [observation])
            return np.argmax(output[0])
