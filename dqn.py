import logging
import platform
import random
import time
from enum import Enum

import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


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


class Qnet(torch.nn.Module):
    """ 只有一层隐藏层的Q网络 """

    def __init__(self, input_shape, hidden_dim, action_dim):
        """
        :param input_shape: 状态空间
        :param hidden_dim: 隐藏层的维度
        :param action_dim: 动作空间的维度
        """
        super(Qnet, self).__init__()
        self.input_shape = input_shape
        c, h, w = self.input_shape
        logging.error(self.input_shape)
        # 卷积层
        self.conv1 = nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2最大池化
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2最大池化
        # 全连接层
        self.fc1 = torch.nn.Linear(self._get_conv_output_size(), hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def _get_conv_output_size(self):
        # 计算卷积和池化后的输出尺寸
        with torch.no_grad():
            c, h, w = self.input_shape
            dummy_input = torch.zeros(1, c, h, w)
            output_feat = self.pool2(F.relu(self.conv2(self.pool1(F.relu(self.conv1(dummy_input))))))
            return int(np.prod(output_feat.size()))

    def forward(self, x):
        # 前向传播
        # 卷积层 + ReLU + 池化
        x = F.relu(self.conv1(x))
        x = self.pool1(x)  # 池化层，减少特征图大小
        x = F.relu(self.conv2(x))
        x = self.pool2(x)  # 池化层，进一步减少特征图大小
        # 展平特征图以适应全连接层
        x = x.view(x.size(0), -1)  # 展平操作
        # 全连接层 + ReLU
        x = F.relu(self.fc1(x))
        # 输出层
        x = self.fc2(x)
        return x


class VAnet(torch.nn.Module):
    """ 只有一层隐藏层的A网络和V网络 """

    def __init__(self, input_shape, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        self.input_shape = input_shape
        c, h, w = self.input_shape
        # 卷积层
        self.conv1 = nn.Conv2d(c, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        # 池化层
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2最大池化
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2最大池化
        # 全连接层
        self.fc1 = torch.nn.Linear(self._get_conv_output_size(), hidden_dim)
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def _get_conv_output_size(self):
        # 计算卷积和池化后的输出尺寸
        with torch.no_grad():
            c, h, w = self.input_shape
            dummy_input = torch.zeros(1, c, h, w)
            output_feat = self.pool2(F.relu(self.conv2(self.pool1(F.relu(self.conv1(dummy_input))))))
            return int(np.prod(output_feat.size()))

    def forward(self, x):
        # 前向传播
        # 卷积层 + ReLU + 池化
        x = F.relu(self.conv1(x))
        x = self.pool1(x)  # 池化层，减少特征图大小
        x = F.relu(self.conv2(x))
        x = self.pool2(x)  # 池化层，进一步减少特征图大小
        # 展平特征图以适应全连接层
        x = x.view(x.size(0), -1)  # 展平操作
        # 全连接层 + ReLU
        x = F.relu(self.fc1(x))
        A = self.fc_A(x)
        V = self.fc_V(x)
        Q = V + A - A.mean(1).view(-1, 1)  # Q值由V值和A值计算得到
        return Q


class DqnType(Enum):
    VANILLA_DQN = 1
    DUELING_DQN = 2
    DOUBLE_DQN = 3


class DQN:
    """ DQN算法 """

    def __init__(self, input_shape, hidden_dim, action_dim, learning_rate, gamma, epsilon, epsilon_decay, target_update,
                 device, dqn_type=DqnType.VANILLA_DQN, is_eval=False):
        """
        初始化qdn网络
        :param input_shape: 输入层-状态空间
        :param hidden_dim: 隐藏层 维度
        :param action_dim: 输出层-动作空间 维度(表示有几个动作)
        :param learning_rate: 学习率
        :param gamma: 折扣因子
        :param epsilon: 贪婪策略(探索因子)
        :param epsilon_decay: 探索衰减因子
        :param target_update: 目标网络更新频率
        :param device: gpu or cpu
        :param dqn_type: dqn网络类型
        :param is_eval: 是否是评测模式
        """
        self.action_dim = action_dim
        if not is_eval:
            if dqn_type == DqnType.DUELING_DQN:  # Dueling DQN采取不一样的网络框架
                self.q_net = VAnet(input_shape, hidden_dim, self.action_dim).to(device)
                self.target_q_net = VAnet(input_shape, hidden_dim, self.action_dim).to(device)
            else:
                self.q_net = Qnet(input_shape, hidden_dim, self.action_dim).to(device)
                self.target_q_net = Qnet(input_shape, hidden_dim, self.action_dim).to(device)
            # 使用Adam优化器
            self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        else:
            self.load_model()
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device
        self.dqn_type = dqn_type
        self.epsilon_decay = epsilon_decay  # 探索衰减因子
        self.is_eval = is_eval

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon and not self.is_eval:
            # 随机探索
            action = np.random.randint(self.action_dim)
        else:
            # 增加一个批次的维度
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        # logging.error(states.shape)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        if self.dqn_type == DqnType.DOUBLE_DQN:
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

    def save_model(self):
        torch.save(self.q_net, 'dqn_q_net_model.pth')
        torch.save(self.target_q_net, 'dqn_target_q_net_model.pth')
        torch.save(self.optimizer, 'dqn_optimizer_model.pth')

    def load_model(self):
        self.q_net = torch.load('dqn_q_net_model.pth')
        self.target_q_net = torch.load('dqn_target_q_net_model.pth')
        self.optimizer = torch.load('dqn_optimizer_model.pth')


# 定义环境
class SysEnv:
    def __init__(self, grid_width, grid_height):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.state = (0, 0)
        self.goal = (grid_width - 1, grid_height - 1)
        self.mapinfo = np.zeros((self.grid_width, self.grid_height, 2))
        self.max_step = 20
        self.fill_static_map_info()

    def reset(self):
        self.state = (0, 0)
        self.mapinfo = np.zeros((self.grid_width, self.grid_height, 2))
        self.mapinfo[self.state[0]][self.state[1]][0] = 1
        self.fill_static_map_info()
        self.max_step = 20
        return np.transpose(self.mapinfo, (2, 0, 1))

    def step(self, action):
        x, y = self.state
        if action == 0:  # 上
            y = min(self.grid_height - 1, y + 1)
        elif action == 1:  # 下
            y = max(0, y - 1)
        elif action == 2:  # 左
            x = max(0, x - 1)
        elif action == 3:  # 右
            x = min(self.grid_width - 1, x + 1)
        old_state = self.state
        self.state = (x, y)
        reward = -1
        # 碰壁
        if old_state == self.state:
            reward += -50
        # 撞墙
        # if self.mapinfo[x][y][1] == 1:
        #     reward += -50
        #     self.state = old_state

        done = self.state == self.goal
        if done:
            reward += 1000
        else:
            goal_x, goal_y = self.goal
            reward += - (abs(goal_x - x) + abs(goal_y - y))
        self.mapinfo[old_state[0]][old_state[1]][0] = 0
        self.mapinfo[self.state[0]][self.state[1]][0] = 1
        # logging.error("pos {}, action: {}".format(self.state, action))
        return np.transpose(self.mapinfo, (2, 0, 1)), reward, done

    def fill_static_map_info(self):
        pass
        # self.mapinfo[5][3][1] = 1
        # self.mapinfo[5][4][1] = 1
        # self.mapinfo[5][5][1] = 1
        # self.mapinfo[5][6][1] = 1
        # self.mapinfo[5][7][1] = 1
        # self.mapinfo[5][8][1] = 1

    def __get_zeros(self, pos, view_distance):
        x, y = pos
        x_start, y_start = max(x - view_distance, 0), max(y - view_distance, 0)
        x_end, y_end = min(x + view_distance, self.grid_width - 1), min(y + view_distance, self.grid_height - 1)
        return self.mapinfo[x_start:(x_end + 1), y_start:(y_end + 1)]


def train_dqn(epochs=100000):
    system_name = platform.system()
    if system_name == "Darwin":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.error(device)
    replay_buffer = ReplayBuffer(10000)
    input_shapes = (2, 10, 10)  # C，W，H 格式
    action_dim = 4  # 4个动作
    hidden_dim = 56  # 隐藏层个数
    lr = 0.001
    gamma = 0.95
    epsilon = 0.9
    epsilon_decay = 0.995
    target_update = 10
    agent = DQN(input_shapes, hidden_dim, action_dim, lr, gamma, epsilon, epsilon_decay, target_update, device,
                DqnType.DOUBLE_DQN)
    env = SysEnv(10, 10)
    rewards = []
    # 记录开始时间
    start_time = time.time()
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
        if agent.epsilon > 0.01:
            agent.epsilon *= agent.epsilon_decay
        if e % 10 == 0:
            # 记录每次迭代结束时间
            end_time = time.time()
            print(f'Epoch: {e}, Total Reward: {total_reward}, Duration: {end_time - start_time:.2f} seconds')
            start_time = time.time()
    agent.save_model()
    return rewards


def usc_dqn():
    system_name = platform.system()
    if system_name == "Darwin":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    input_shape = (2, 10, 10)
    action_dim = 4
    hidden_dim = 56
    lr = 0.001
    gamma = 0.95
    epsilon = 0.9
    epsilon_decay = 0.995
    target_update = 10
    agent = DQN(input_shape, hidden_dim, action_dim, lr, gamma, epsilon, epsilon_decay, target_update, device,
                DqnType.DOUBLE_DQN, True)
    env = SysEnv(10, 10)
    state = env.reset()
    done = False
    while not done:
        action = agent.take_action(state)
        # 根据ai选择的动作 计算下个下一个状态和当前动作的奖励，以及是否完成任务
        next_state, reward, done = env.step(action)
        state = next_state
        # logging.error("pos {}, action: {}".format(state, action))


if __name__ == '__main__':
    # 配置日志系统
    logging.basicConfig(
        level=logging.DEBUG,  # 设置日志级别
        format='%(asctime)s - %(levelname)s - %(message)s '
               '[in %(filename)s:%(lineno)d(%(funcName)s)]',  # 设置日志格式
        datefmt='%Y-%m-%d %H:%M:%S'  # 设置时间格式
    )

    rewards = train_dqn(10000)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward Over Episodes')
    plt.show()
    # usc_dqn()
