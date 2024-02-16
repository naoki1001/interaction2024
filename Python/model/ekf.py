import torch
from torch import nn

class ExtendedKalmanFilterForPoseEstimation(nn.Module):
    def __init__(self, dim_x=10, dim_z=3, dim_u=6, dt=0.005):
        super(ExtendedKalmanFilterForPoseEstimation, self).__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = torch.tensor([1.0, 0.0, 0.0, 0.0]) # state 
        self.P = torch.eye(dim_x) * (1.74E-2 * dt ** 2)  # uncertainty covariance
        self.Q = torch.eye(dim_x) * (1.74E-2 * dt ** 2)  # process uncertainty
        self.R = torch.eye(dim_z) * (1.0 * dt ** 2)  # state uncertainty
        self.F = torch.eye(dim_x)      # state transition matrix
        self.B = 0                     # control transition matrix

        # These matrices will be updated during each step
        self.y = torch.zeros(dim_z, 1) # residual
        self.K = torch.zeros(dim_x, dim_z) # Kalman gain
        self.S = torch.zeros(dim_z, dim_z) # system uncertainty
        self.z = torch.zeros(dim_z, 1) # Last measurement

        # Identity matrix
        self._I = torch.eye(dim_x)

    def forward(self, z, u):
        self.F = self.calc_F(self.x, u)

        if z is not None and z.numel() == 1 and self.dim_z == 1:
            z = z.view(-1, 1)

        F = self.F
        B = self.B
        P = self.P
        Q = self.Q
        x = self.x

        H = self.HJacobian(x)

        # Predict step
        self.predict_x(u)
        x = self.x
        P = torch.matmul(torch.matmul(F, P), F.t()) + Q

        # Save prior
        self.x_prior = x.clone()
        self.P_prior = P.clone()

        # Update step
        PHT = torch.matmul(P, H.t())
        self.S = torch.matmul(H, PHT) + self.R
        try:
            self.K = torch.matmul(PHT, torch.inverse(self.S))
        except:
            self.K = torch.matmul(PHT, torch.pinverse(self.S))

        self.y = z - self.Hx(x)
        print(self.K.size())
        print(self.y.size())
        self.x = x + torch.matmul(self.K, self.y)

        I_KH = self._I - torch.matmul(self.K, H)
        self.P = torch.matmul(torch.matmul(I_KH, P), I_KH.t()) + torch.matmul(torch.matmul(self.K, self.R), self.K.t())

        # Save measurement and posterior state
        self.z = z.clone()
        self.x_post = self.x.clone()
        self.P_post = self.P.clone()

        return self.x

    def predict_x(self, u=0):
        self.x = self.f(self.x, u)

    def HJacobian(self, x):
        """
        Jacobian of the observation function with respect to the quaternion.
        x: Quaternion [q_w, q_x, q_y, q_z]
        """
        q_w, q_x, q_y, q_z = x

        # Partial derivatives of theta and phi with respect to quaternion components
        H = torch.zeros((2, 4))
        H[0, 0] = 2 * q_y / (1 + (2 * (q_w*q_y + q_x*q_z))**2)
        H[0, 1] = 2 * q_z / (1 + (2 * (q_w*q_y + q_x*q_z))**2)
        H[0, 2] = 2 * q_w / (1 - (2 * (q_y**2 + q_z**2))**2)
        H[0, 3] = 2 * q_x / (1 - (2 * (q_y**2 + q_z**2))**2)
        H[1, 0] = 2 * q_x / (1 + (2 * (q_w*q_x + q_y*q_z))**2)
        H[1, 1] = 2 * q_w / (1 - (2 * (q_x**2 + q_z**2))**2)
        H[1, 2] = 2 * q_z / (1 + (2 * (q_w*q_x + q_y*q_z))**2)
        H[1, 3] = 2 * q_y / (1 - (2 * (q_x**2 + q_z**2))**2)

        return H

    def Hx(self, x):
        """
        Observation function that converts quaternion to pitch and roll angles.
        x: Quaternion [q_w, q_x, q_y, q_z]
        """
        # Calculate pitch and roll from the quaternion
        q_w, q_x, q_y, q_z = x
        theta = torch.atan2(2*(q_w*q_y + q_x*q_z), 1 - 2*(q_y**2 + q_z**2))
        phi = torch.atan2(2*(q_w*q_x + q_y*q_z), 1 - 2*(q_x**2 + q_z**2))
        
        return torch.tensor([theta, phi])

    def calc_F(self, x, u):
        """
        Compute the Jacobian matrix F_k for the state transition function.
        x_k: Current state quaternion [q_w, q_x, q_y, q_z]
        u_k: Control input [omega_x * delta_t, omega_y * delta_t, omega_z * delta_t]
        """
        dt = 0.005
        # Construct the F_k matrix
        F = torch.eye(4) + 0.5 * dt * torch.tensor([
            [0, -u[0], -u[1], -u[2]],
            [u[0], 0, u[2], -u[1]],
            [u[1], -u[2], 0, u[0]],
            [u[2], u[1], -u[0], 0]
        ], dtype=x.dtype)

        return F

    def f(self, x, u):
        """
        Updates the quaternion based on the angular velocity and time interval using PyTorch.
        x is the current quaternion, u is the control input.
        """
        u_k = torch.tensor([0, *u])
        # Compute the quaternion derivative
        q_dot = 0.5 * x#quaternion_multiplication(x, u_k)

        # Update the quaternion
        x_updated = x + q_dot

        # Normalize the updated quaternion
        x_normalized = x_updated / torch.norm(x_updated)

        return x_normalized


import numpy as np
import scipy.linalg as linalg
from copy import deepcopy

class ExtendedKalmanFilterForIMU:
    def __init__(self, dim_x=3, dim_z=2, dim_u=3, dt=0.005):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u
        self.x = np.zeros((dim_x, 1)) # state
        self.P = np.diag([1.74E-2*dt**2, 1.74E-2*dt**2, 1.74E-2*dt**2]) # uncertainty covariance
        self.B = 0                 # control transition matrix
        self.F = np.eye(dim_x)     # state transition matrix
        self.R = np.diag([1.0*dt**2, 1.0*dt**2])    # state uncertainty
        self.Q = np.diag([1.74E-2*dt**2, 1.74E-2*dt**2, 1.74E-2*dt**2]) # process uncertainty

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.y = np.zeros((dim_z, 1)) # residual
        self.K = np.zeros(self.x.shape) # kalman gain
        self.S = np.zeros((dim_z, dim_z))   # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))  # inverse system uncertainty
        self.z = np.zeros((dim_z, 1))

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

    def predict_update(self, z, u=0):
        self.F = self.calc_F(self.x, u)

        if np.isscalar(z) and self.dim_z == 1:
            z = np.asarray([z], float)

        F = self.F
        B = self.B
        P = self.P
        Q = self.Q
        R = self.R
        x = self.x

        H = self.HJacobian(x)

        # predict step
        # x = np.dot(F, x) + np.dot(B, u)
        self.predict_x(u=u)
        x = self.x
        P = F @ P @ F.T + Q

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

        # update step
        PHT = P @ H.T
        self.S = H @ PHT + R
        try:
            self.SI = linalg.inv(self.S)
        except:
            self.SI = linalg.pinv(self.S)
        self.K = PHT @ self.SI

        self.y = z - self.Hx(x)
        self.x = x + self.K @ self.y

        I_KH = self._I - self.K @ H
        # self.P = (I_KH @ P) @ I_KH.T + (self.K @ R) @ self.K.T
        self.P = I_KH @ P

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    
    def predict_x(self, u=0):
        self.x = self.f(self.x, u)
    
    def predict(self, u=0):
        self.F = self.calc_F(self.x, self.u)
        return super().predict(u)
    
    def f(self, x, u):
        u_x, u_y, u_z = u[0][0], u[1][0], u[2][0]
        c1, s1 = np.cos(x[0][0]), np.sin(x[0][0])
        c2, s2 = np.cos(x[1][0]), np.sin(x[1][0])
        c3, s3 = np.cos(x[2][0]), np.sin(x[2][0])
        x = np.array([
            [x[0][0] + u_x + u_y * s1 * s2 / c2 + u_z * c1 * s2 / c2],
            [x[1][0] + u_y * c1 - u_z * s1],
            [x[2][0] + u_y * s1 / c2 + u_z * c1 / c2]
        ])
        return x

    def calc_F(self, x, u):
        u_x, u_y, u_z = u[0][0], u[1][0], u[2][0]
        c1, s1 = np.cos(x[0][0]), np.sin(x[0][0])
        c2, s2 = np.cos(x[1][0]), np.sin(x[1][0])
        c3, s3 = np.cos(x[2][0]), np.sin(x[2][0])
        F = np.array([
            [1 + u_y * c1 * s2 / c2 - u_z * s1 * s2 / c2, u_y * s1 / c2 ** 2 + u_z * c1 / c2 ** 2, 0],
            [-u_y * s1 - u_z * c1, 1, 0],
            [u_y * c1 / c2 - u_z * s1 / c2, u_y * s1 * s2 / c2 ** 2 + u_z * c1 * s2 / c2 ** 2, 1]
        ])
        return F

    def HJacobian(self, x):
        H = np.eye(2, 3)
        return H
    
    def Hx(self, x):
        y = np.eye(2, 3) @ x
        return y