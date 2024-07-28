import logging
import platform
import random
import numpy as np
import collections
import torch
import torch.nn.functional as F
import unittest


class ReplayBuffer:
    """ 经验回放池 """

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class TestReplayBuffer(unittest.TestCase):
    def test_sample(self):
        state = [1]
        # state = np.array(state, dtype=np.float32)
        next_state = [2]
        # next_state = np.array(next_state, dtype=np.float32)
        r = ReplayBuffer(10)
        r.add(state, action=0, reward=0, next_state=next_state, done=0)
        r.add(next_state, action=1, reward=1, next_state=state, done=1)
        b_s, b_a, b_r, b_ns, b_d = r.sample(2)
        transition_dict = {
            'states': b_s,
            'actions': b_a,
            'next_states': b_ns,
            'rewards': b_r,
            'dones': b_d
        }
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)
        print(states, actions, rewards, next_states, dones)

    def test_array(self):
        state = (0, 0)
        state = np.array(state, dtype=np.float32)
        print(state)
        state = np.array(state, dtype=np.float32)
        print(state)

    def test_tensor(self):
        state = (0, 0)
        print(torch.tensor(state, dtype=torch.float))
        state = np.array(state, dtype=np.float32)
        print(torch.tensor(state, dtype=torch.float))
        input_tensor = torch.randn(2, 5, dtype=torch.float)
        print(input_tensor)


class Qnet(torch.nn.Module):
    """ 只有一层隐藏层的Q网络 """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        # 输入层 -> 隐藏层
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # 隐藏层 -> 输出层
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class VAnet(torch.nn.Module):
    """ 只有一层隐藏层的A网络和V网络 """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 共享网络部分
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        A = self.fc_A(x)
        V = self.fc_V(x)
        Q = V + A - A.mean(1).view(-1, 1)  # Q值由V值和A值计算得到
        return Q


class DQN:
    """ DQN算法 """

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device, dqn_type='VanillaDQN'):
        """
        初始化qdn网络
        :param state_dim: 输入层-状态空间 维度
        :param hidden_dim: 隐藏层 维度
        :param action_dim: 输出层-动作空间 维度(表示有几个动作)
        :param learning_rate: 学习率
        :param gamma: 折扣因子
        :param epsilon: 贪婪策略
        :param target_update: 目标网络更新频率
        :param device: gpu or cpu
        """
        self.action_dim = action_dim
        if dqn_type == 'DuelingDQN':  # Dueling DQN采取不一样的网络框架
            self.q_net = VAnet(state_dim, hidden_dim, self.action_dim).to(device)
            self.target_q_net = VAnet(state_dim, hidden_dim, self.action_dim).to(device)
        else:
            self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
            self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device
        self.dqn_type = dqn_type

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            # 随机探索
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        if self.dqn_type == 'DoubleDQN':
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else:
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)

        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1


# 定义环境
class SysEnv:
    def __init__(self, grid_width, grid_height):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.state = [self.__pos2int(0, 0)]
        self.goal = [self.__pos2int(grid_width - 1, grid_height - 1)]

    def reset(self):
        self.state = [self.__pos2int(0, 0)]
        return self.state

    def step(self, action):
        x, y = self.__int2pos(self.state[0])
        if action == 0:  # 上
            y = min(self.grid_height - 1, y + 1)
        elif action == 1:  # 下
            y = max(0, y - 1)
        elif action == 2:  # 左
            x = min(0, x - 1)
        elif action == 3:  # 右
            x = max(self.grid_width - 1, x + 1)
        old_state = self.state
        self.state = [self.__pos2int(x, y)]
        done = self.state == self.goal
        if done:
            reward = 1000
        else:
            if old_state == self.state:
                reward = -10
            else:
                reward = -(abs(self.__int2pos(self.goal[0])[0] - x) + abs(self.__int2pos(self.goal[0])[1] - y))
        return self.state, reward, done

    def __pos2int(self, x: int, y: int):
        """二维坐标转一维"""
        return self.grid_width * y + x

    def __int2pos(self, posval: int):
        return posval % self.grid_width, posval / self.grid_width


def train_dqn(epochs=1000):
    system_name = platform.system()
    if system_name == "Darwin":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    replay_buffer = ReplayBuffer(2000)
    state_dim = 1
    action_dim = 4
    hidden_dim = 64
    lr = 0.001
    gamma = 0.95
    epsilon = 0.9
    target_update = 10
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device, "DuelingDQN")
    env = SysEnv(10, 10)
    rewards = []
    for e in range(epochs):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.take_action(state)
            # 根据ai选择的动作 计算下个下一个状态和当前动作的奖励，以及是否完成任务
            next_state, reward, done = env.step(action)
            # 加入经验池子
            replay_buffer.add(state, action, reward, next_state, done)
            #  切换状态
            state = next_state
            # 累计奖励
            total_reward += reward
            # 当积累足够多的经验后，才开始训练
            if replay_buffer.size() >= 32:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(32)
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d
                }
                agent.update(transition_dict)

        rewards.append(total_reward)
        if e % 10 == 0:
            print(f'Epoch: {e}, Total Reward: {total_reward}')

    return rewards
