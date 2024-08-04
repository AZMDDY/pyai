import logging
import platform
import random
import time

import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from Gridworld import Gridworld


class ReplayBuffer:
    """ 经验回放池 """

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return state, action, reward, next_state, done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN1:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, epsilon_decay, target_update,
                 device, is_eval=False):
        """
        状态空间为一阶向量
        :param state_dim:
        :param hidden_dim:
        :param action_dim:
        :param learning_rate:
        :param gamma:
        :param epsilon:
        :param epsilon_decay:
        :param target_update:
        :param device:
        :param is_eval:
        """
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        if not is_eval:
            self.q_net = QNet(state_dim, hidden_dim, self.action_dim).to(device)
            self.target_q_net = QNet(state_dim, hidden_dim, self.action_dim).to(device)
            # 使用Adam优化器
            self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        else:
            self.load_model()
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device
        self.epsilon_decay = epsilon_decay  # 探索衰减因子
        self.is_eval = is_eval

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon and not self.is_eval:
            # 随机探索
            action = np.random.randint(self.action_dim)
        else:
            state = torch.from_numpy(state).float().to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = (torch.tensor(np.array(transition_dict['states']), dtype=torch.float)
                  .view(-1, self.state_dim).to(self.device))
        actions = (torch.tensor(np.array(transition_dict['actions']))
                   .view(-1, 1).to(self.device))
        rewards = (torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float)
                   .view(-1, 1).to(self.device))
        next_states = (torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float)
                       .view(-1, self.state_dim).to(self.device))
        dones = (torch.tensor(np.array(transition_dict['dones']), dtype=torch.float)
                 .view(-1, 1).to(self.device))
        # print(states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)
        q_values = self.q_net(states).gather(1, actions)  # Q值

        max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
        max_next_q_values = self.target_q_net(next_states).gather(1, max_action)

        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1
        return dqn_loss.item()

    def save_model(self):
        torch.save(self.q_net, 'dqn_q_net_model.pth')
        torch.save(self.target_q_net, 'dqn_target_q_net_model.pth')
        torch.save(self.optimizer, 'dqn_optimizer_model.pth')

    def load_model(self):
        self.q_net = torch.load('dqn_q_net_model.pth')
        self.target_q_net = torch.load('dqn_target_q_net_model.pth')
        self.optimizer = torch.load('dqn_optimizer_model.pth')


action_set = {
    0: 'u',
    1: 'd',
    2: 'l',
    3: 'r',
}


def train_dqn(epochs=1000):
    system_name = platform.system()
    if system_name == "Darwin":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.error(device)
    replay_buffer = ReplayBuffer(100)
    state_dim = 4 * 4 * 4
    action_dim = 4  # 4个动作
    hidden_dim = 128  # 隐藏层个数
    lr = 0.001
    gamma = 0.95
    epsilon = 0.9
    epsilon_decay = 0.995
    target_update = 10
    agent = DQN1(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, epsilon_decay, target_update, device)
    rewards = []
    losses = []
    # 记录开始时间
    start_time = time.time()
    for e in range(epochs):
        game = Gridworld(size=4, mode='static')
        state1 = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        # print(state1.shape)
        done = False
        total_reward = 0
        total_loss = 0
        while not done:
            action_ = agent.take_action(state1)
            action = action_set[action_]
            game.makeMove(action)
            state2 = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
            reward = game.reward()
            if reward != -1:
                done = True
            # 加入经验池子
            replay_buffer.add(state1, action_, reward, state2, done)
            #  切换状态
            state1 = state2
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
                loss = agent.update(transition_dict)
                total_loss += loss
        rewards.append(total_reward)
        losses.append(total_loss)
        if agent.epsilon > 0.01:
            agent.epsilon *= agent.epsilon_decay
        if e % 10 == 0:
            # 记录每次迭代结束时间
            end_time = time.time()
            print(f'Epoch: {e}, Total loss: {total_loss}, Duration: {end_time - start_time:.2f} seconds')
            start_time = time.time()
    agent.save_model()
    return rewards


train_dqn()
