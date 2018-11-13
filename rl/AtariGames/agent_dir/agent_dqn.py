from agent_dir.agent import Agent
import tensorflow as tf
from collections import deque


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
        self.states = tf.placeholder(dtype=tf.float32, shape=[None, 80, 80, 4], name="states")
        self.actions = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions], name="actions")
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="rewards")
        self.next_states = tf.placeholder(dtype=tf.float32, shape=[None, 80, 80, 4], name="next_states")
        
        self.q_target = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions], name="q_target")


    def _add_predictions(self):
        with tf.variable_scope("eval_net"):
            self.output_eval = self.predict_op(self.states, name="output_eval")
        with tf.variable_scope("target_net"):
            self.output_target = self.predict_op(self.next_states, name="output_target")
        
        self.loss = tf.reduce_mean(tf.square(self.output_eval - self.q_target))


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


    def train(self, session):
        pass


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
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        tf.logging.info("...... start training ......")
        self.init_game_setting()
        i_episode = -1
        running_reward = None
        
        replay_buf = []
        replay_buf = deque(maxlen=10)

        while True:
            state = self.env.reset()
            done = False

            #playing one game
            while(not done):
                self.env.env.render()
                action = self.make_action(state, test=False)
                next_state, reward, done, info = self.env.step(action)
                replay_buf.append([state, action, reward, next_state])
                state = next_state

            print("replay_reward:", [(rep[1], rep[2]) for rep in replay_buf])
            # self.model.train(self.sess, episode_states, episode_actions, discount_rewards)


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
        return self.env.get_random_action()

