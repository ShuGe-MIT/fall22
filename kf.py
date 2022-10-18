# collaborator: Evan Vogelbaum
"""This file contains stubs for implementation of the Kalman filter"""
from model import KFModel
from numpy.typing import ArrayLike
import numpy as np
from numpy.linalg import inv


class KalmanFilter:

    def __init__(self, params: KFModel):
        self.params = params
        """Other initializations can go here"""
        self.invR, self.invG, self.invQ, T, G= inv(self.params.R), inv(self.params.G), inv(self.params.Q), self.params.T, self.params.G
        self.A = np.array([
            [1,0,T,0],
            [0,1,0,T],
            [T*G[0,0], T*G[0,1], 1, 0],
            [T*G[1,0], T*G[1,1], 0, 1]
        ])
        self.J = -self.A.T@self.invQ
        self.C = np.array([[1, 0, 0, 0],[0, 1, 0, 0]])
        self.h = np.zeros((4,1))
        lam = self.params.Lambda
        self.invL = inv(np.array([[lam[0,0],lam[0,1],0,0],
                                [lam[1,0],lam[1,1],0,0],
                                [0,0,1e-9,0],
                                [0,0,0,1e-9]]))
        self.i = 0
        self.Jp = None
        self.hp = None


    def predict(self) -> ArrayLike:
        """
        This function, when implemented, should predict one step of the Kalman filter and return the predicted state vector
        """
        return self.hp


    def correct(self, meas) -> ArrayLike:
        """
        This function, when implemented, should implement one correction step of the Kalman filter and return the corrected state vector 
        """
        hi = self.C.T @ self.invR @ meas.reshape(2,1)
        hp_ = self.hp if self.i else np.zeros((4,1))
        Jp_ = self.invQ + self.Jp if self.i else self.invL

        add = self.invL if self.i==0 else self.invQ
        Jii = add + self.C.T @ self.invR @ self.C + self.A.T @ self.invQ @ self.A
        self.Jp = -self.J.T @ inv(Jii) @ self.J if self.i==0 else -self.J.T @ inv(Jii+self.Jp) @ self.J
        self.hp = -self.J.T @ inv(Jii) @ self.h if self.i==0 else -self.J.T @ inv(Jii + self.Jp) @ (self.hp + hi)
        self.i+=1

        h_ = hp_ + hi
        J_ = Jp_ + (self.C.T @ self.invR @ self.C)
        return (inv(J_) @ h_).reshape(4)

    def run_n_steps(self, N: int, measurements: ArrayLike) -> ArrayLike:
        """Given N (number of steps) and a 2xN measurement, runs N steps of the KF and returns a 4 x N trajectory prediction

        Args:
            N ([int]): Number of steps
            measurements ([ArrayLike]): Measurement array

        Returns:
            ArrayLike: Trajectory
        """
        est_trajectory = np.zeros((4, N))
        # print(est_trajectory.shape, measurements.shape)
        for n in range(N):
            _ = self.predict()
            # print(n)
            est_trajectory[:, n] = self.correct(measurements[:, n])

        return est_trajectory
