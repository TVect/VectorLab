使用 Policy Gradient 玩 atari Pong

Please don't revise test.py, environment.py, agent_dir/agent.py

# 环境安装

安装 OpenAI Gym Atari 环境

- Linux 下安装
```
pip install opencv-python gym 
pip install gym[atari]
```

- windows 下安装
```
pip install opencv-python gym
pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
```

gym 文档参见 http://gym.openai.com/docs/

# How to run :
training policy gradient:
* `$ python main.py --train_pg`

testing policy gradient:
* `$ python test.py --test_pg`

training DQN:
* `$ python main.py --train_dqn`

testing DQN:
* `$ python test.py --test_dqn`
