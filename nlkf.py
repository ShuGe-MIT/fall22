# collaborator: Evan Vogelbaum
"""This file contains stubs for implementation of the unscented Kalman filter and particle filter"""
from model import NonLinearKFModel
from numpy.typing import ArrayLike
import numpy as np
from scipy.stats import multivariate_normal as mn
from numpy.random import multivariate_normal as rmn


class ParticleFilter:

    def __init__(self, params: NonLinearKFModel, num_particles):
        self.params = params
        """Other initializations can go here"""
        self.field = self.params.field
        self.n = num_particles
        self.stack = []
        for _ in range(self.n):
            pos = rmn(np.zeros(2), self.params.Lambda)
            self.stack.append(np.array([pos[0], pos[1], 0, 0]))
        self.w = np.ones(self.n) / self.n

    def forward(self, meas):
        
        T = self.params.T
        stack = []
        for i in range(self.n):
            x,y,vx,vy = self.stack[i]
            Ex, Ey = self.field(x, y)
            chi = np.array([x + vx*T, y + vy*T, vx + Ex * T, vy + Ey * T])
            stack.append(rmn(chi, self.params.Q))
        self.stack = stack

        for i in range(self.n):
            self.w[i] = self.w[i] * mn.pdf(meas, mean=self.stack[i][0:2].reshape(-1), cov=self.params.R)
        self.w = self.w / np.sum(self.w)

    ## Define method to implement the particle filter
    def run_n_steps(self, N: int, measurements: ArrayLike) -> ArrayLike:
        """Given N (number of steps) and a 2xN measurement, runs N steps of the KF and returns a 4 x N trajectory prediction

        Args:
            N ([int]): Number of steps
            measurements ([ArrayLike]): Measurement array

        Returns:
            ArrayLike: Trajectory
        """
        est_trajectory = np.zeros((4, N))
        for n in range(N):
            self.forward(measurements[:, n])
            est_trajectory[:, n] = np.average(self.stack, axis=0, weights=self.w)
            # np.average(self.stack, axis=0, weights=self.w)

        return est_trajectory