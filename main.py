import logging
import platform
import random
import numpy as np
import pygame
import torch
import dqn
import unittest
import matplotlib.pyplot as plt


class Play:
    def __init__(self, title: str, win_width: int, win_hight: int, cell_width: int, cell_height: int, grid_width: int,
                 grid_height: int):
        """
        初始化窗口信息
        :param win_width: 窗口宽度
        :param win_hight: 窗口高度
        :param cell_width: 方格的宽度
        :param cell_height: 方格的高度
        :param grid_width: 网格的宽度(单位是方格，表示有多少个方格)
        :param grid_height: 网格的高度(单位是方格，表示有多少个方格)
        """
        self.win_width = win_width
        self.win_hight = win_hight
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.grid_width = grid_width
        self.grid_height = grid_height

        pygame.init()
        self.screen = pygame.display.set_mode((self.win_width, self.win_hight))
        pygame.display.set_caption(title)
        self.font = pygame.font.Font(None, 18)

    def __conv_pos(self, x: int, y: int):
        """
        将实际的画布坐标，转化成方格在网格中的坐标
        """
        return (x * self.cell_width + self.cell_width + self.cell_width / 2,
                self.grid_height * self.cell_height - self.cell_height / 2 - self.cell_height * y)

    def draw_in_cell(self, x: int, y: int, content: str):
        """
        在每个格子中写内容
        :param x: 格子的位置x
        :param y: 格子的位置y
        :param content: 要写的内容
        :return:
        """
        text = self.font.render(content, True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = self.__conv_pos(x, y)
        self.screen.blit(text, text_rect)

    def __draw_grid(self):
        # 绘制网格
        # 绘制竖线
        for x in range(1, self.grid_width + 2):
            pygame.draw.line(self.screen, (0, 0, 0), (x * self.cell_width, 0),
                             (x * self.cell_width, self.cell_height * self.grid_height))
        # x轴坐标
        for x in range(self.grid_width):
            self.draw_in_cell(x, -1, str(x))
        # 绘制横线
        for y in range(0, self.grid_height + 1):
            pygame.draw.line(self.screen, (0, 0, 0), (self.cell_width, y * self.cell_height),
                             (self.cell_width * (1 + self.grid_width), y * self.cell_height))

        # Y轴坐标
        for y in range(self.grid_width):
            self.draw_in_cell(-1, y, str(y))

    def draw_per_frame(self):
        """绘制每一帧"""
        pass

    def run(self):
        # 主循环
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # 填充背景颜色
            self.screen.fill((255, 255, 255))  # 白色背景

            # 绘制网格
            self.__draw_grid()

            self.draw_in_cell(1, 1, 'A')
            self.draw_per_frame()
            # 更新显示
            pygame.display.flip()

            # 设置帧率
            pygame.time.Clock().tick(60)

        pygame.quit()


class TPlay(Play):
    def __init__(self, title: str, win_width: int, win_hight: int, cell_width: int, cell_height: int, grid_width: int,
                 grid_height: int):
        super().__init__(title, win_width, win_hight, cell_width, cell_height, grid_width, grid_height)

    def draw_per_frame(self):
        self.draw_in_cell(9, 9, 'B')


if __name__ == '__main__':
    # 配置日志系统
    logging.basicConfig(
        level=logging.DEBUG,  # 设置日志级别
        format='%(asctime)s - %(levelname)s - %(message)s '
               '[in %(filename)s:%(lineno)d(%(funcName)s)]',  # 设置日志格式
        datefmt='%Y-%m-%d %H:%M:%S'  # 设置时间格式
    )
    # unittest.main()
    # p = TPlay("dqn", 500, 500, 40, 40, 10, 10)
    # p.run()

    rewards = dqn.train_dqn(10000)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward Over Episodes')
    plt.show()
    #dqn.usc_dqn()
