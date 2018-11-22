from agent_dir.agent import Agent
import numpy as np
import tensorflow as tf
import os

tf.logging.set_verbosity(tf.logging.INFO)

def prepro(I):
    """
    This preprocessing code is from
        https://gist.github.com/greydanus/5036f784eec2036252e1990da21eda18
    prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector    
    """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1    # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


class Policy:
    
    def __init__(self):
        # hyperparameters
        self.n_obs = 80 * 80           # dimensionality of observations
        self.h = 200                   # number of hidden layer neurons
        self.n_actions = 3             # number of available actions
        self.learning_rate = 1e-3
        self.gamma = .99               # discount factor for reward
        self.decay = 0.99              # decay rate for RMSProp gradients
        self.save_path='models/pong.ckpt'
        self._build()
    
    
    def _build(self):
        self._add_placeholder()
        self._add_predictions_v2()
        self._add_saver()
    
    
    def _add_placeholder(self):
        self.states = tf.placeholder(dtype=tf.float32, shape=[None, self.n_obs], name="states")
        self.actions = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions], name="actions")
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="rewards")
    

    def _add_predictions_v2(self):
        tf_model = {}
        with tf.variable_scope('layer_one',reuse=False):
            xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(self.n_obs), dtype=tf.float32)
            tf_model['W1'] = tf.get_variable("W1", [self.n_obs, self.h], initializer=xavier_l1)
        with tf.variable_scope('layer_two',reuse=False):
            xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(self.h), dtype=tf.float32)
            tf_model['W2'] = tf.get_variable("W2", [self.h, self.n_actions], initializer=xavier_l2)

        hidden = tf.matmul(self.states, tf_model['W1'])
        hidden = tf.nn.relu(hidden)
        logp = tf.matmul(hidden, tf_model['W2'])
        self.act_probs = tf.nn.softmax(logp)

        tf_discounted_epr = self.rewards
        tf_mean, tf_variance= tf.nn.moments(tf_discounted_epr, [0], shift=None, name="reward_moments")
        tf_discounted_epr -= tf_mean
        tf_discounted_epr /= tf.sqrt(tf_variance + 1e-6)

        with tf.variable_scope("loss"):
            # self.loss = tf.nn.l2_loss(self.actions - self.act_probs)
            self.loss = -self.actions * tf.log(self.act_probs)
        
        self.global_step = tf.Variable(0, trainable=False)
        with tf.variable_scope("optim"):
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=self.decay)
            tf_grads = optimizer.compute_gradients(self.loss, 
                                                   var_list=tf.trainable_variables(), 
                                                   grad_loss=tf_discounted_epr)
            self.optim = optimizer.apply_gradients(tf_grads, global_step=self.global_step)


    def _add_saver(self):
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)


    def train(self, sess, states, actions, rewards):
        _, _loss = sess.run([self.optim, self.loss], 
                            feed_dict={self.actions: np.vstack(actions), 
                                       self.states: np.vstack(states),
                                       self.rewards: np.vstack(rewards)})
        return _loss


    def predict(self, sess, states):
        return sess.run(self.act_probs, feed_dict={self.states: states})


    def save(self, sess, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, 
                        os.path.join(checkpoint_dir, "pg-model"), 
                        global_step=self.global_step)


    def load(self, sess, checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            tf.logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            tf.logging.info("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())


class Agent_PG(Agent):

    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_PG,self).__init__(env)
        self.n_actions = 3
        self.n_obs = 80 * 80

        if args.test_pg:
            #you can load your model here
            tf.logging.info('...... loading trained model ......')
            self.model = Policy()
        else:
            self.model = Policy()


    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        self.sess = tf.Session()
        self.model.load(self.sess, "./runs/checkpoints")


    def train(self):
        """
        Implement your training algorithm here
        """
        tf.logging.info("...... start training ......")
        self.init_game_setting()
        i_episode = -1
        running_reward = None
        while True:
            i_episode += 1
            state = self.env.reset()
            prev_x = None
            done = False
            episode_rewards = []
            episode_states = []
            episode_actions = []
            #playing one game
            while(not done):
                # self.env.env.render()
                action = self.make_action(state, test=False)

                action_vec = np.zeros(self.n_actions)
                action_vec[action] = 1
                episode_actions.append(action_vec)

                cur_x = prepro(state)
                diff_states = cur_x - prev_x if prev_x is not None else np.zeros(self.n_obs)
                episode_states.append(diff_states)
                prev_x = cur_x

                state, reward, done, info = self.env.step(action+1)
                episode_rewards.append(reward)

            # 做 discount reward，需要在 某个人得分之后 将 running_add 重新置为 0
            discount_rewards = np.zeros_like(episode_rewards)
            running_add = 0
            for t in reversed(range(0, len(episode_rewards))):
                if episode_rewards[t] != 0:
                    running_add = 0
                running_add = running_add * 0.99 + episode_rewards[t]
                discount_rewards[t] = running_add
            # discount_rewards -= np.mean(discount_rewards)
            # discount_rewards /= (np.std(discount_rewards) + 1e-3)

            # episode_actions = np.eye(self.actions_size)[episode_actions]
            self.model.train(self.sess, episode_states, episode_actions, discount_rewards)

            # print progress console
            reward_sum = np.sum(episode_rewards)
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01            
            if i_episode % 10 == 0:
                tf.logging.info('ep {}: reward: {}, mean reward: {:3f}'.format(i_episode, reward_sum, running_reward))
            else:
                tf.logging.info('\tep {}: reward: {}'.format(i_episode, reward_sum))

            if i_episode % 100 == 0:
                self.model.save(self.sess, "./runs/checkpoints")


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        probs = self.model.predict(self.sess, [prepro(observation)])
        if test:
            return probs[0].argmax() + 1
        else:
            return np.random.choice(self.n_actions, p=probs[0])

