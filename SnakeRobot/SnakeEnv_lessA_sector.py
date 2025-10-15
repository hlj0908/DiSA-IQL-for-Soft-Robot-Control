# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 22:21:42 2023

@author: Xinda Qi
"""

import math
import numpy as np
# from scipy import integrate
import matplotlib.pyplot as plt
import time
import numpy as np
import random
import torch
from gym import spaces

class SnakeEnv:
    """
    This is the class for the simulation of the sithering locomotion of snakes
    Focusing on the dynamics of the center of mass of snakes

    """

    def __init__(self, config, links=48, length=0.5, seed=None):

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

        # define the constants and matrices
        self.length = length
        self.links = links                                  # default to be 24, 4 seg per actuator, N， need to be times of 12
        self.h = self.length / self.links                        # the length for each segment
        self.e = np.ones((self.links, 1))                    # unit vector
        self.T1 = np.tri(self.links, self.links, k=0)       # lower triangle matrix
        self.T2 = np.tile(np.arange(self.links, 0, -1), (self.links,1) )        # a graduate matrix
        self.I = self.h * self.T1 - self.h/self.links * self.T2
        self.rho = 1                                        # linear density
        self.mu_f = 1                                       # the friction coefficients in forward
        self.mu_t = 1.5                                       # the friction coefficients in trasverse
        self.g = 9.8
        self.eps = 0.000000001


        # initialize the center of the snake
        self.X = 0
        self.Y = 0
        self.ANG = 0.07        # adjust the initial angle so that it face forward
        self.X_d = 0        # derivaive of X
        self.Y_d = 0
        self.ANG_d = 0
        self.X_dd = 0       # second derivative of X
        self.Y_dd = 0
        self.ANG_dd = 0
        self.t = 0          # the time of the simulation
        self.t_end = 25      # the end time of the simulation
        self.dt = config.getfloat('sample_time')      # the time step of the calculation % try 0.01 or 0.05
        self.count = 0


        # the structural data for the soft robot
        a = np.ones((self.links//12, 1))
        b = np.zeros((self.links//12,1))
        self.ch_1 = np.concatenate((a, b, b, -a, -a, b, b, a, a, b, b, -a))
        self.ch_2 = np.concatenate((b, a, a, b, b, -a, -a, b, b, a, a, b))
        self.ch_3 = np.concatenate((-a, b, b, a, a, b, b, -a, -a, b, b, a))
        self.ch_4 = np.concatenate((b, -a, -a, b, b, a, a, b, b, -a, -a, b))

        # for the record of the simulation datas
        self.X_record = []
        self.Y_record = []
        self.cur_v_record = []
        self.cur_a_record = []

        # for the period bias data
        self.B1_pre = self.B2_pre = self.B3_pre = self.B4_pre = 0


        # initialize the curvatures
        self.curs = np.zeros((self.links, 1))                 # the array curvatures
        self.curs_v = np.zeros((self.links, 1))
        self.curs_a = np.zeros((self.links, 1))

        # self.zeros = np.zeros((self.links, 1))
        self.curs_pre = self.curs
        self.curs_v_pre = self.curs_v

        # calculate the states at the start point
        self.cal_accer()

        # set the random target for the robot
        self.goal = [config.getfloat('goal_x'), config.getfloat('goal_y')]

        self.random_field = config.getboolean('random_field')
        self.r_upper = config.getfloat('radius_upper')
        self.r_lower = config.getfloat('radius_lower')

        self.expand_sector = config.getboolean('expand_sector')
        self.expand_sector_2 = config.getboolean('expand_sector_2')
        self.expand_sector_3 = config.getboolean('expand_sector_3')
        self.sr_upper = config.getfloat('sr_upper')
        self.sr_lower = config.getfloat('sr_lower')
        self.expand_order = config.getfloat('expand_order')
        self.max_episode = config.getfloat('max_episode')

        # self.goal_x = self.goal[0] + np.random.uniform(-0.2, 0.2)
        # self.goal_y = self.goal[1] + np.random.uniform(-0.2, 0.2)

        # use the fixed goal to test the performance first
        self.goal_x = self.goal[0]
        self.goal_y = self.goal[1]


        self.dis_0 = math.sqrt((self.goal_x-self.X)**2 + (self.goal_y-self.Y)**2)
        self.adj_dis = config.getfloat('adjacency_dis')       # adjacency of 3cm circle
        self.done = 0
        self.achieve_goal = 0
        self.final_reward = 50

        # get the draw argument from the config file
        self.draw = config.getboolean('draw')
        # draw the snake robot
        if self.draw:
            plt.plot(self.goal_x, self.goal_y, "X", color="red")
            plt.grid(True)
            plt.axis("equal")
            plt.xlim(-1.5, 1.5)
            plt.ylim(-1.5, 1.5)

        # calculate the pos and angle errors
        self.dis = []
        self.ang_dis = []

        # imformation for the RL agent
        # the observasions / states are: [dx, dy,dtheta, b1, b2, b1_pre, b2_pre]
        self.observation_space = 7
        obs_dim =7
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # the actions are: bias in chambers (4) + F/B
        # self.action_space = 5
        # self.act_limit = 4                                     # use 4 or 4.28 as the bias limit
        # the new action space would be equvelent chamber (2) + F/B
        self.action_space = 3
        act_dim = 3
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        self.act_limit = 8                                      # the limit of the equivent chamber doubles

        self.goal = [config.getfloat('goal_x'), config.getfloat('goal_y')]
        self.episode_length = int(config.getfloat('episode_length'))    # the step# limit of one episode
        self.side = config['side'] if 'side' in config else 'all'

    def reset(self, episode=0, A=8, B1=0, B2=0, B3=0, B4=0):
        """
        Reset the enviornment

        """

        # initialize the center of the snake
        self.X = 0
        self.Y = 0
        self.ANG = 0.07        # adjust the initial angle so that it face forward
        self.X_d = 0        # derivaive of X
        self.Y_d = 0
        self.ANG_d = 0
        self.X_dd = 0       # second derivative of X
        self.Y_dd = 0
        self.ANG_dd = 0
        self.t = 0          # the time of the simulation
        self.t_end = 25      # the end time of the simulation
        # self.dt = 0.01      # the time step of the calculation % can try 0.01 or 0.05
        self.count = 0

        # also reset the goal here
        if self.random_field:
            r = np.random.uniform(self.r_lower, self.r_upper)

            if hasattr(self, "side") and self.side == "left":  # left
                ang = np.random.uniform(math.pi / 2, 3 * math.pi / 2)
            elif hasattr(self, "side") and self.side == "right":  # right
                ang = np.random.uniform(-math.pi / 2, math.pi / 2)
            else:  # all
                ang = np.random.uniform(0, 2 * math.pi)

            self.goal_x = r * math.cos(ang)
            self.goal_y = r * math.sin(ang)

        elif self.expand_sector:
            sector_width = (episode / self.max_episode) ** self.expand_order * math.pi / 2
            r = np.random.uniform(self.sr_lower, self.sr_upper)
            ang = np.random.uniform(math.pi/2-sector_width, math.pi/2+sector_width)
            self.goal_x = r * math.cos(ang)
            self.goal_y = r * math.sin(ang)

        elif self.expand_sector_2:    # with slower end increase
            if episode / self.max_episode < 0.4:
                sector_width = (episode / self.max_episode * 1.25) ** self.expand_order * math.pi / 2
            else:
                sector_width = ((episode / self.max_episode * 1.25) - 0.25) * math.pi / 2
            r = np.random.uniform(self.sr_lower, self.sr_upper)
            ang = np.random.uniform(math.pi / 2 - sector_width, math.pi / 2 + sector_width)
            self.goal_x = r * math.cos(ang)
            self.goal_y = r * math.sin(ang)

        elif self.expand_sector_3:     # start at the typical targets
            if episode / self.max_episode < 0.4:
                sector_width = (episode / self.max_episode * 1.25) ** self.expand_order * math.pi
            else:
                sector_width = ((episode / self.max_episode * 1.25) - 0.25) * math.pi
            r = np.random.uniform(self.sr_lower, self.sr_upper)     # sample a random r
            ang_can = np.random.uniform(0, sector_width)                # sample the valid range
            if ang_can < sector_width / 4:                          # the ang are counter-clockwise
                ang = ang_can
            elif ang_can < sector_width / 2:
                ang = math.pi/2 - (sector_width/2-ang_can)
            elif ang_can < sector_width * 3/4:
                ang = math.pi/2 + (ang_can-sector_width/2)
            else:
                ang = math.pi - (sector_width-ang_can)
            self.goal_x = r * math.cos(ang)                         # finally get the positions
            self.goal_y = r * math.sin(ang)

        else:
            # use a smaller random range
            self.goal_x = self.goal[0] + np.random.uniform(-0.2, 0.2)
            self.goal_y = self.goal[1] + np.random.uniform(-0.2, 0.2)


        # initialize the curvatures
        # self.curs = np.zeros((self.links, 1))                 # the array curvatures

        # start with a wave like form
        P1 = A * np.sin(0)
        P2 = A * np.sin(0 + math.pi/2)
        P3 = A * np.sin(0 + math.pi)
        P4 = A * np.sin(0 + math.pi/2*3)
        self.curs = P1 * self.ch_1 + P2 * self.ch_2 + P3 * self.ch_3 + P4 * self.ch_4
        self.curs_v = np.zeros((self.links, 1))
        self.curs_a = np.zeros((self.links, 1))

        # self.zeros = np.zeros((self.links, 1))
        # for previous data
        self.curs_pre = self.curs
        self.curs_v_pre = self.curs_v
        self.B1_pre = B1
        self.B2_pre = B2
        self.B3_pre = B3
        self.B4_pre = B4

        # calculate the states at the start point
        self.cal_accer()

        # for the record of the simulation datas
        self.X_record = []
        self.Y_record = []
        self.cur_v_record = []
        self.cur_a_record = []

        # reset the target
        # self.goal_x = self.goal[0] + np.random.uniform(-0.2, 0.2)
        # self.goal_y = self.goal[1] + np.random.uniform(-0.2, 0.2)
        self.dis_0 = math.sqrt((self.goal_x-self.X)**2 + (self.goal_y-self.Y)**2)

        self.done = 0
        self.achieve_goal = 0

        # draw the snake robot
        if self.draw:
            plt.plot(self.goal_x, self.goal_y, "X", color="red")
            plt.grid(True)
            plt.axis("equal")
            plt.xlim(-1.5, 1.5)
            plt.ylim(-0.5, 0.5)

        # reset pos and angle errors
        self.dis = []
        self.ang_dis = []

        # calculate reward once
        _, ang_dis = self.cal_reward()

        # return initial state/observation
        # The states includes the current pos, current bias and previous bias
        # use relative pos / ang as states
        # obs = [self.goal_x-self.X, self.goal_y-self.Y, ang_dis, \
        #        self.X_d, self.Y_d, self.ANG_d, \
        #        B1,B2,B3,B4, self.B1_pre,self.B2_pre,self.B3_pre,self.B4_pre]
        obs = [self.goal_x-self.X, self.goal_y-self.Y, ang_dis, \
               B1,B2, self.B1_pre,self.B2_pre]

        obs = np.array([
            float(self.goal_x - self.X),
            float(self.goal_y - self.Y),
            float(ang_dis),
            float(B1),
            float(B2),
            float(self.B1_pre),
            float(self.B2_pre)
        ], dtype=np.float32)

        info = {}  # 如果你没有 info，可以返回空字典
        return obs, info




    def calculate(self):
        """
        Perform the simulation in a time period

        """

        while self.t < self.t_end:

            # record the center first
            self.X_record.append(self.X)
            self.Y_record.append(self.Y)

            # get the input for continous simulation
            # self.pressure()
            self.pressure_segments(A=8, B2=-3)

            self.cur_v_record.append(self.curs_v[0][0])
            self.cur_a_record.append(self.curs_a[0][0])

            # update the position and velocities of the center of the snake
            # use semi euler method, update the position first
            self.ANG = self.ANG + self.ANG_d * self.dt
            self.X = self.X + self.X_d * self.dt
            self.Y = self.Y + self.Y_d * self.dt

            # then calculate the speed based on the new position
            self.cal_accer()

            self.ANG_d = self.ANG_d + self.ANG_dd * self.dt
            self.X_d = self.X_d + self.X_dd * self.dt
            self.Y_d = self.Y_d + self.Y_dd * self.dt

            # finally, step the time forward
            self.t += self.dt

    def pressure(self):
        """
        generate the curvature input to the model

        """

        x = np.arange(0, self.length, self.h).reshape((self.links,1))
        k = 2 * math.pi
        w = 2*math.pi * 1
        A = 3
        B = -2

        # get the curvatures
        self.curs = A * np.sin(w * self.t + k * x) + B
        self.curs_v = w*A * np.cos(w * self.t + k * x)
        self.curs_a = -w**2 * A * np.sin(w * self.t + k * x)



    def pressure_segments(self, A=8, w=2*math.pi, B1=0, B2=0, B3=0, B4=0):
        """
        generate the curvatures of the snake robot based on the pressures

        """

        P1 = A * np.sin(w * self.t) + B1
        P2 = A * np.sin(w * self.t + math.pi/2) + B2
        P3 = A * np.sin(w * self.t + math.pi) + B3
        P4 = A * np.sin(w * self.t + math.pi/2*3) + B4

        P1_v = w*A * np.cos(w * self.t)
        P2_v = w*A * np.cos(w * self.t + math.pi/2)
        P3_v = w*A * np.cos(w * self.t + math.pi)
        P4_v = w*A * np.cos(w * self.t + math.pi/2*3)

        P1_a = -w**2 * A * np.sin(w * self.t)
        P2_a = -w**2 * A * np.sin(w * self.t + math.pi/2)
        P3_a = -w**2 * A * np.sin(w * self.t + math.pi)
        P4_a = -w**2 * A * np.sin(w * self.t + math.pi/2*3)

        # get the curvatures
        self.curs = P1 * self.ch_1 + P2 * self.ch_2 + P3 * self.ch_3 + P4 * self.ch_4
        self.curs_v = P1_v * self.ch_1 + P2_v * self.ch_2 + P3_v * self.ch_3 + P4_v * self.ch_4
        self.curs_a = P1_a * self.ch_1 + P2_a * self.ch_2 + P3_a * self.ch_3 + P4_a * self.ch_4



    def step_small(self, B1=0, B2=0, B3=0, B4=0, A=8, w=2*math.pi):
        """
        Use the new pressure value in the next step to update the ODE in one step

        """

        # record the positions
        self.X_record.append(self.X)
        self.Y_record.append(self.Y)

        # calculate the curvatures based on the bias
        P1 = A * np.sin(w * self.t) + B1 + A
        P2 = A * np.sin(w * self.t + math.pi/2) + B2 + A
        P3 = A * np.sin(w * self.t + math.pi) + B3 + A
        P4 = A * np.sin(w * self.t + math.pi/2*3) + B4 + A

        self.curs = P1 * self.ch_1 + P2 * self.ch_2 + P3 * self.ch_3 + P4 * self.ch_4

        # calculate the curvature speed and acceleration
        if self.t < 2*self.dt:
            # send in smooth start estimation
            P1_v = w*A * np.cos(w * self.t)
            P2_v = w*A * np.cos(w * self.t + math.pi/2)
            P3_v = w*A * np.cos(w * self.t + math.pi)
            P4_v = w*A * np.cos(w * self.t + math.pi/2*3)

            P1_a = -w**2 * A * np.sin(w * self.t)
            P2_a = -w**2 * A * np.sin(w * self.t + math.pi/2)
            P3_a = -w**2 * A * np.sin(w * self.t + math.pi)
            P4_a = -w**2 * A * np.sin(w * self.t + math.pi/2*3)

            self.curs_v = P1_v * self.ch_1 + P2_v * self.ch_2 + P3_v * self.ch_3 + P4_v * self.ch_4
            self.curs_a = P1_a * self.ch_1 + P2_a * self.ch_2 + P3_a * self.ch_3 + P4_a * self.ch_4

        else:
            # calculate the derivatives
            self.curs_v = (self.curs - self.curs_pre) / self.dt
            self.curs_a = (self.curs_v - self.curs_v_pre) / self.dt

        # record the cuvatures
        # self.cur_v_record.append(self.curs_v[0][0])
        # self.cur_a_record.append(self.curs_a[0][0])

        # use semi euler method, update the position first
        self.ANG = self.ANG + self.ANG_d * self.dt
        self.X = self.X + self.X_d * self.dt
        self.Y = self.Y + self.Y_d * self.dt

        # then calculate the speed based on the new position
        self.cal_accer()

        self.ANG_d = self.ANG_d + self.ANG_dd * self.dt
        self.X_d = self.X_d + self.X_dd * self.dt
        self.Y_d = self.Y_d + self.Y_dd * self.dt

        # update the previous data and step the clock forward
        self.curs_pre = self.curs
        self.curs_v_pre = self.curs_v
        self.t += self.dt


        # the position / angle data can be aquired from the attribute directly
        # return self.X, self.Y, self.ANG



    def step_pre(self, action, A=8, w=2*math.pi):
        """
        Use the new pressure value (p_x, p_v, p_a) in one sinusoidal wave
        to update the dynamic of the snake after one period
        this is period_step

        """

        # get the bias from the actions
        B1, B2 = action
        B3, B4 = 0, 0

        T = 2*math.pi/w
        s_stamp = self.t
        e_stamp = self.t + T
        while self.t < e_stamp:
            # simulate in one period
            t = self.t - s_stamp

            # record the positions
            self.X_record.append(self.X)
            self.Y_record.append(self.Y)

            # it is noticed that we should use self.t in sin function to keep smooth calculation
            # if using t, a time discretization error will be brought, which can be improved by using smaller dt
            P1 = A * np.sin(w * self.t) + (B1-self.B1_pre)*(t/T)+self.B1_pre
            P2 = A * np.sin(w * self.t + math.pi/2) + (B2-self.B2_pre)*(t/T)+self.B2_pre
            P3 = A * np.sin(w * self.t + math.pi) + (B3-self.B3_pre)*(t/T)+self.B3_pre
            P4 = A * np.sin(w * self.t + math.pi/2*3) + (B4-self.B4_pre)*(t/T)+self.B4_pre

            P1_v = w*A * np.cos(w * self.t) + (B1-self.B1_pre)*(1/T)
            P2_v = w*A * np.cos(w * self.t + math.pi/2) + (B2-self.B2_pre)*(1/T)
            P3_v = w*A * np.cos(w * self.t + math.pi) + (B3-self.B3_pre)*(1/T)
            P4_v = w*A * np.cos(w * self.t + math.pi/2*3) + (B4-self.B4_pre)*(1/T)

            P1_a = -w**2 * A * np.sin(w * self.t)
            P2_a = -w**2 * A * np.sin(w * self.t + math.pi/2)
            P3_a = -w**2 * A * np.sin(w * self.t + math.pi)
            P4_a = -w**2 * A * np.sin(w * self.t + math.pi/2*3)

            # get the curvatures
            self.curs = P1 * self.ch_1 + P2 * self.ch_2 + P3 * self.ch_3 + P4 * self.ch_4
            self.curs_v = P1_v * self.ch_1 + P2_v * self.ch_2 + P3_v * self.ch_3 + P4_v * self.ch_4
            self.curs_a = P1_a * self.ch_1 + P2_a * self.ch_2 + P3_a * self.ch_3 + P4_a * self.ch_4

            # record the cuvatures
            # self.cur_v_record.append(self.curs_v[0][0])
            # self.cur_a_record.append(self.curs_a[0][0])

            # update the position and velocities of the center of the snake
            # use semi euler method, update the position first
            self.ANG = self.ANG + self.ANG_d * self.dt
            self.X = self.X + self.X_d * self.dt
            self.Y = self.Y + self.Y_d * self.dt

            # then calculate the speed based on the new position
            self.cal_accer()

            self.ANG_d = self.ANG_d + self.ANG_dd * self.dt
            self.X_d = self.X_d + self.X_dd * self.dt
            self.Y_d = self.Y_d + self.Y_dd * self.dt

            # step the time forward
            self.t += self.dt
            self.count += 1

            # plot the snake robot if needed
            if self.draw and self.count % 1 == 0:
                self.display()

            # calculate the current distance to see if reach the target
            dis = math.sqrt((self.goal_x-self.X)**2 + (self.goal_y-self.Y)**2)
            if dis <= self.adj_dis:
                self.done = 1
                self.achieve_goal = 1
                print("Done, reach the goal!")
                # stop early once meet the target
                break

        # terminate the running if exceed the total time
        if self.t >= self.episode_length:
            self.done = 1


        # update the bias history
        self.B1_pre = B1
        self.B2_pre = B2
        self.B3_pre = B3
        self.B4_pre = B4

        # return next_obs, reward, done
        reward, ang_dis  = self.cal_reward()

        obs = [self.goal_x-self.X, self.goal_y-self.Y, ang_dis, \
               self.X_d, self.Y_d, self.ANG_d, \
               B1,B2,B3,B4, self.B1_pre,self.B2_pre,self.B3_pre,self.B4_pre]

        # return transition and other information
        return obs, reward, self.done, self.achieve_goal

    def step(self, action, A=8, w=2*math.pi):
        """
        Use the new pressure value (p_x, p_v, p_a) in one sinusoidal wave
        to update the dynamic of the snake after one period

        This is for the go forward and back direction, with respect with its main direction
        use the accumulate time for the calculation

        direc = 0,0.5 -> back
            = 0.5,1 -> forward
        direc -> [0,1]
        the new direc is seperated by 0 now

        """

        # get the bias from the actions
        # B1, B2, B3, B4, direc = action
        a1, a2, direc = action
        B1 = float(a1) * self.act_limit
        B2 = float(a2) * self.act_limit
        B3, B4 = 0, 0

        T = 2*math.pi/w
        # use an internal clock
        t = 0
        # determine the direction of the step
        if direc < 0:
            c = -1
        else:
            c = 1

        while t < T:
            # simulate in one period

            # record the positions
            self.X_record.append(self.X)
            self.Y_record.append(self.Y)
            # record the speed
            # self.speed_x.append(self.X_d)
            # self.speed_y.append(self.Y_d)
            # self.total_speed.append(math.sqrt(self.X_d**2 + self.Y_d**2))

            # it is noticed that we should use self.t in sin function to keep smooth calculation
            # if using t, a time discretization error will be brought, which can be improved by using smaller dt
            P1 = A * np.sin(w * c*t) + (B1-self.B1_pre)*(t/T)+self.B1_pre
            P2 = A * np.sin(w * c*t + math.pi/2) + (B2-self.B2_pre)*(t/T)+self.B2_pre
            P3 = A * np.sin(w * c*t + math.pi) + (B3-self.B3_pre)*(t/T)+self.B3_pre
            P4 = A * np.sin(w * c*t + math.pi/2*3) + (B4-self.B4_pre)*(t/T)+self.B4_pre

            P1_v = c * w*A * np.cos(w * c*t) + (B1-self.B1_pre)*(1/T)
            P2_v = c * w*A * np.cos(w * c*t + math.pi/2) + (B2-self.B2_pre)*(1/T)
            P3_v = c * w*A * np.cos(w * c*t + math.pi) + (B3-self.B3_pre)*(1/T)
            P4_v = c * w*A * np.cos(w * c*t + math.pi/2*3) + (B4-self.B4_pre)*(1/T)

            P1_a = c**2 * (-w**2) * A * np.sin(w * c*t)
            P2_a = c**2 * (-w**2) * A * np.sin(w * c*t + math.pi/2)
            P3_a = c**2 * (-w**2) * A * np.sin(w * c*t + math.pi)
            P4_a = c**2 * (-w**2) * A * np.sin(w * c*t + math.pi/2*3)

            # get the curvatures
            self.curs = P1 * self.ch_1 + P2 * self.ch_2 + P3 * self.ch_3 + P4 * self.ch_4
            self.curs_v = P1_v * self.ch_1 + P2_v * self.ch_2 + P3_v * self.ch_3 + P4_v * self.ch_4
            self.curs_a = P1_a * self.ch_1 + P2_a * self.ch_2 + P3_a * self.ch_3 + P4_a * self.ch_4

            # record the cuvatures
            # self.cur_v_record.append(self.curs_v[0][0])
            # self.cur_a_record.append(self.curs_a[0][0])

            # update the position and velocities of the center of the snake
            # use semi euler method, update the position first
            self.ANG = self.ANG + self.ANG_d * self.dt
            self.X = self.X + self.X_d * self.dt
            self.Y = self.Y + self.Y_d * self.dt

            # then calculate the speed based on the new position
            self.cal_accer()

            self.ANG_d = self.ANG_d + self.ANG_dd * self.dt
            self.X_d = self.X_d + self.X_dd * self.dt
            self.Y_d = self.Y_d + self.Y_dd * self.dt

            # step the time forward
            t += self.dt
            self.count += 1

            # plot the snake robot if needed
            if self.draw and self.count % 5 == 0:
                self.display()


            # calculate the current distance to see if reach the target
            dis = math.sqrt((self.goal_x-self.X)**2 + (self.goal_y-self.Y)**2)
            if dis <= self.adj_dis:
                self.done = 1
                self.achieve_goal = 1
                print("Done, reach the goal!")
                # stop early once meet the target
                break

        # terminate the running if exceed the total time
        if self.t >= self.episode_length:
            self.done = 1

        # update the global time
        self.t += T

        # return next_obs, reward, done
        reward, ang_dis  = self.cal_reward()

        # obs = [self.goal_x-self.X, self.goal_y-self.Y, ang_dis, \
        #        self.X_d, self.Y_d, self.ANG_d, \
        #        B1,B2,B3,B4, self.B1_pre,self.B2_pre,self.B3_pre,self.B4_pre]
        obs = [self.goal_x-self.X, self.goal_y-self.Y, ang_dis, \
               B1,B2, self.B1_pre,self.B2_pre]

        # update the bias history
        self.B1_pre = B1
        self.B2_pre = B2
        self.B3_pre = B3
        self.B4_pre = B4

        obs = np.array([
            float(self.goal_x - self.X),
            float(self.goal_y - self.Y),
            float(ang_dis),
            float(B1),
            float(B2),
            float(self.B1_pre),
            float(self.B2_pre)
        ], dtype=np.float32)

        terminated = self.done  # Whether reached the destination/failed
        truncated = False  # If you don't have pre-termination logic such as max steps, set it to False
        info = {"achieve_goal": self.achieve_goal}

        # return transition and other information
        return np.array(obs, dtype=np.float32), reward, terminated, truncated, info

    def step_with_presure_dynamics(self, action, A=8, w=2*math.pi):
        """
        Use the new pressure value (p_x, p_v, p_a) in one sinusoidal wave
        to update the dynamic of the snake after one period

        This is for the go forward and back direction, with respect with its main direction
        use the accumulate time for the calculation

        direc = 0,0.5 -> back
            = 0.5,1 -> forward
        direc -> [0,1]

        In this simulation, design a dynamic for the pressure in the soft chamber
        where the direct input is the air flow (with certain pressure) and the pressure is the result
        we can make this dynamic very "fast" and pretty "statble"

        And the relation between curvature and presure is also the linear relationship
        which means that we negilect the inertia and external friction force,
        which is relatively small compared with the pressure force and elasticity of soft body
        And we also assume a linear elasticity material
        """

        """
        TODO: implement the period step with pressure dynamic
        And also a simple and have nice PD controller
        And make the curvature / pressure as the observations
        """

        pass


    def cal_reward(self):
        """
        Calculate the reward of the agent

        """
        dis = math.sqrt((self.goal_x-self.X)**2 + (self.goal_y-self.Y)**2)

        """notice here, since we considered the posiblity of go forward and backward,
        when we calculate the ang reward, it should within [-pi/2, pi/2]
        """
        cur_direc = math.atan2((self.goal_y-self.Y), (self.goal_x-self.X))
        # process the ANG to (-pi, pi]
        Ang = self.ANG
        while Ang > 2*math.pi:
            Ang -= 2*math.pi
        while Ang < -2*math.pi:
            Ang += 2*math.pi
        if Ang > math.pi:
            Ang -= 2*math.pi
        elif Ang <= -math.pi:
            Ang += 2*math.pi

        # get the angle diff in global
        ang_dis = cur_direc - Ang
        if ang_dis > math.pi:
            ang_dis -= 2 * math.pi
        elif ang_dis <= -math.pi:
            ang_dis += 2 * math.pi

        # 再压到 [-pi/2, pi/2]
        if ang_dis > math.pi / 2:
            ang_dis -= math.pi
        elif ang_dis < -math.pi / 2:
            ang_dis += math.pi

        # update the target errors
        self.dis.append(dis)
        #self.ang_dis.append(abs(ang_dis))
        self.ang_dis.append(float(abs(ang_dis)))

        # balance the rewards
        # reward = -1* ( dis/self.dis_0 + abs(ang_dis)/math.pi) + self.achieve_goal * self.final_reward
        reward = -1 * ( 0.15 * dis/self.dis_0 + 1 * abs(ang_dis)/math.pi) + 1 * self.achieve_goal * self.final_reward
        reward_2 = -1 * ( dis/self.dis_0 + 2 * math.exp(-6 * abs(ang_dis)/ (math.pi/2) )) + self.achieve_goal * self.final_reward
        reward_3 = -1 * (0.01 * 1) + 1 * self.achieve_goal * self.final_reward

        # print("re", reward)
        return reward, ang_dis
        # return reward_2, ang_dis


    def cal_accer(self):
        """
        Calculate the derivative values for the ODE update in step funtion

        """

        # input the curvatures
        # self.curs = curs
        # self.curs_v = curs_v
        # self.curs_a = curs_a

        # calculate the angles and the positions
        self.angs = self.ANG * self.e + np.matmul(self.I, self.curs)
        cangs = np.cos(self.angs)
        sangs = np.sin(self.angs)
        self.xs = self.X * self.e + np.matmul(self.I, cangs)
        self.ys = self.Y * self.e + np.matmul(self.I, sangs)


        self.angs_d = self.ANG_d*self.e + np.matmul(self.I, self.curs_v)
        self.xs_d = self.X_d * self.e + np.matmul(self.I, -sangs*self.angs_d)
        self.ys_d = self.Y_d * self.e + np.matmul(self.I, cangs*self.angs_d)


        # calculate the friction force of different part of the snakes
        self.f_x = -self.rho*self.h*self.g * ( self.mu_t * (self.xs_d*(-sangs)+self.ys_d*cangs) * (-sangs)\
                + self.mu_f * (self.xs_d*cangs+self.ys_d*sangs) * cangs
                )
        self.f_y = -self.rho*self.h*self.g * ( self.mu_t * (self.xs_d*(-sangs)+self.ys_d*cangs) * (cangs)\
                + self.mu_f * (self.xs_d*cangs+self.ys_d*sangs) * sangs
                )

        # we can use another types of the force discription
        # axial = abs(self.xs_d*cangs + self.ys_d*sangs) + self.eps
        # trans = abs(self.xs_d*(-sangs) + self.ys_d*cangs) + self.eps
        # mu_dis = self.mu_f * axial/(axial+trans) + self.mu_t * trans/(axial+trans)
        # self.f_x = -self.rho*self.h*self.g * mu_dis * self.xs_d
        # self.f_y = -self.rho*self.h*self.g * mu_dis * self.ys_d

        # we can calculate the angular acceleration first
        right_eq_0 = 1/self.rho/self.h * ( (self.xs-self.X).T @ self.f_y - (self.ys-self.Y).T @ self.f_x )
        right_eq_1 = -1 * (self.I @ cangs).T @ (self.I @ (-sangs * self.angs_d * self.angs_d))
        right_eq_2 = -1 * (self.I @ cangs).T @ (self.I @ (cangs * (self.I @ self.curs_a)) )
        right_eq_3 = (self.I @ sangs).T @ (self.I @ (-cangs * self.angs_d * self.angs_d))
        right_eq_4 = (self.I @ sangs).T @ (self.I @ (-sangs * (self.I @ self.curs_a)) )

        left_eq_0 = (self.I @ cangs).T @ (self.I @ cangs) + (self.I @ sangs).T @ (self.I @ sangs)

        self.ANG_dd = (right_eq_0 + right_eq_1 + right_eq_2 + right_eq_3 + right_eq_4) / left_eq_0

        # get the total force
        totalF_x = np.sum(self.f_x)
        totalF_y = np.sum(self.f_y)

        # the Newton's Law for the center of the snake
        # integral the following accelaration can get the motion of next step
        self.X_dd = 1/self.rho/self.h/self.links * totalF_x
        self.Y_dd = 1/self.rho/self.h/self.links * totalF_y



    def display(self):
        """
        Display the motion of the snakes if called within a scope

        """

        plt.scatter(self.xs,self.ys, s=1)
        plt.plot(self.goal_x, self.goal_y, "X", color="red")
        plt.grid(True)
        plt.axis("equal")
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.show(False)
        plt.draw()
        plt.clf()

    def render(self):
        """
        Display the motion of the snakes if called within a scope
        use this to see the evaluation episode

        """

        plt.scatter(self.xs,self.ys, s=1)
        plt.plot(self.goal_x, self.goal_y, "X", color="red")
        plt.grid(True)
        plt.axis("equal")
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.scatter(self.xs,self.ys, s=1)
        plt.show(False)
        plt.draw()
        plt.clf()



if __name__ == '__main__':
    import configparser
    # read the hyper-parameters from config file
    config = configparser.ConfigParser()
    config.read('config_show.ini')

    Env = SnakeEnv(config['ENV_CONFIG'])

    start = time.time()

    # simulate, compare the different calculation methods
    # Env.calculate()
    # while Env.t < 25:
    #     Env.step(A=8, B1=0, B2=-3, B4=0)      # 0.75
    # while Env.t < 25:
    #     Env.period_step(A=8, B1=0, B2=-3, B4=0)

    while Env.t < 25:
        Env.step([0,8,1])

    end = time.time()
    print(end-start)


    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(Env.X_record, Env.Y_record)
    ax.set_aspect('equal', adjustable='box')
    plt.show()

    # plot the xs and ys of the snake robot
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    x = Env.xs
    y = Env.ys
    ax2.scatter(x,y)
    ax2.set_aspect('equal', adjustable='box')
    plt.show()

    # monitor the curvatures
    # plt.plot(Env.cur_v_record)
    # plt.show()
    # plt.plot(Env.cur_a_record)
    # plt.show()







