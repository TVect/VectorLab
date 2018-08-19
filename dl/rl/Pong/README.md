ʹ�� Policy Gradient �� atari Pong

Please don't revise test.py, environment.py, agent_dir/agent.py

# ������װ

��װ OpenAI Gym Atari ����

- Linux �°�װ
```
pip install opencv-python gym gym[atari]
```

- windows �°�װ
```
pip install opencv-python gym
pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
```

# How to run :
training policy gradient:
* `$ python main.py --train_pg`

testing policy gradient:
* `$ python test.py --test_pg`

training DQN:
* `$ python main.py --train_dqn`

testing DQN:
* `$ python test.py --test_dqn`
