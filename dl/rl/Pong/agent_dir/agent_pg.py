from agent_dir.agent import Agent
import scipy
import scipy.misc
import numpy as np
import cv2


def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
#     y = 0.2126 * o[:, :, 2] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 0]
#     return cv2.resize(y, (80, 80))
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=0) / 127.5 - 1.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(576, 50) 
        self.fc2 = nn.Linear(50, 6) 

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 576)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)
        self.model_path = "./save/model"

        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            self.net = torch.load(self.model_path)
        else:
            self.net = Net()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters())
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
        print("start training ......")
        for i_episode in range(3):
            state = self.env.reset()
            done = False
            episode_reward = 0.0
            episode_states = [prepro(state)]
            episode_actions = []
            #playing one game
            while(not done):
                # self.env.env.render()
                action = self.make_action(state, test=True)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_actions.append(action)
                episode_states.append(prepro(state))

            
            print("training episode: {}, reward: {} .......".format(i_episode, episode_reward))
            self.optimizer.zero_grad()
            output = self.net(Variable(torch.Tensor(episode_states)))
            target = Variable(torch.LongTensor(episode_actions))
            # target = Variable(torch.LongTensor(np.eye(self.env.action_space.n)[episode_actions]))
            loss = -episode_reward * self.criterion(output[:-1], target)
            loss.backward()
            self.optimizer.step()
        torch.save(self.net, self.model_path)
        ##################
        # YOUR CODE HERE #
        ##################
        pass


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
        ##################
        # YOUR CODE HERE #
        ##################
        actions = self.net(Variable(torch.Tensor([prepro(observation)])))
        return actions.argmax().tolist()
        return self.env.get_random_action()

