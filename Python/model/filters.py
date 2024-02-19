import torch
from torch import nn

import numpy as np
from scipy.spatial.transform import Rotation as R

from ekf import ExtendedKalmanFilterForIMU

def convert_rotation_matrix(matrix):
    opposite_matrix_matrix = matrix
    opposite_matrix_matrix[0][1] *= -1
    opposite_matrix_matrix[1][0] *= -1
    opposite_matrix_matrix[1][2] *= -1
    opposite_matrix_matrix[2][1] *= -1
    return opposite_matrix_matrix

def convert_quaternion(quaternion):
    opposite_quaternion = quaternion
    opposite_quaternion[1] *= -1
    opposite_quaternion[3] *= -1
    return opposite_quaternion

def convert_euler(euler):
    opposite_euler = euler
    opposite_euler[0] *= -1
    opposite_euler[2] *= -1
    return opposite_euler

def convert_axis_to_unity(xyz):
    xyz_converted = np.array([
        [-xyz[1][0]],
        [xyz[2][0]],
        [xyz[0][0]]
    ])
    return xyz_converted

def get_euler_from_acc(acc):
    euler = np.array([
        np.arctan2(acc[1], acc[2]),
        -np.arctan2(acc[0], np.sqrt(acc[1] ** 2 + acc[2] ** 2))
    ], dtype=np.float32)
    return euler

def calc_u(gyro, dt):
    gyro = np.array([
        [gyro[0]],
        [gyro[1]],
        [gyro[2]]
    ])
    u = gyro * dt
    return u

def calc_z(acc):
    z = np.array([
        [np.arctan2(acc[1],  acc[2])], 
        [-np.arctan2(acc[0],  np.sqrt(acc[1] ** 2 + acc[2] ** 2))]
    ])
    return z

class RawDataForPoseEstimation:
    def __init__(self, dim_p=3, dim_v=3, dim_q=4, dim_z=6):
        self.p = np.zeros((dim_p, 1))
        self.v = np.zeros((dim_v, 1))
        self.q = np.array([1.0, 0.0, 0.0, 0.0]) # state
        self.z = np.zeros((dim_z, 1)) # Last measurement
        self.g = np.array([0, 0, 9.80665]).reshape(-1, 1)
        self.ekf = ExtendedKalmanFilterForIMU()

    def forward(self, u, z=None, dt=0.005):
        acc = u[:3].reshape(-1, 1)
        gyro = u[3:]
        if z is None:
            r = R.from_quat(convert_quaternion(self.q))
            rotation_matrix = r.as_matrix()
            acc_g = convert_axis_to_unity(rotation_matrix @ acc - self.g)
            self.p = self.p + dt * self.v + 0.5 * (dt ** 2) * acc_g
            self.v = self.v + dt * acc_g
            self.q = convert_quaternion(R.from_euler('xyz',[*self.get_euler(u=u)], degrees=False).as_quat())
        else:
            rotation_matrix = R.from_quat(convert_quaternion(z[3:])).as_matrix()
            acc_g = convert_axis_to_unity(rotation_matrix @ acc - self.g)
            self.p = z[:3].reshape(-1, 1)
            self.v = self.v + dt * acc_g
            self.q = z[3:]
        return self.p.T[0], self.q
    
    def get_next_state(self, u, z=None, dt=0.005):
        acc = u[:3].reshape(-1, 1)
        gyro = u[3:]
        self.ekf.predict_update(z=calc_z(u[:3]), u=calc_u(gyro, dt))
        if z is None:
            rotation_matrix = R.from_quat(convert_quaternion(self.q)).as_matrix()
            acc_g = convert_axis_to_unity(rotation_matrix @ acc - self.g)
            print(f'acc_g:{acc_g.reshape(3)}')
            print(f'(r, p, y):{self.ekf.x.reshape(3)}')
            self.p = self.p + dt * self.v + 0.5 * (dt ** 2) * acc_g
            self.v = self.v + dt * acc_g
            #self.q = convert_quaternion(R.from_euler('xyz',[*self.get_euler(u=u)], degrees=False).as_quat())
            self.q = convert_quaternion(R.from_euler('xyz',[*self.ekf.x.reshape(3)], degrees=False).as_quat())
        else:
            rotation_matrix = R.from_quat(convert_quaternion(z[3:])).as_matrix()
            acc_g = convert_axis_to_unity(rotation_matrix @ acc - self.g)
            self.p = z[:3].reshape(-1, 1)
            self.v = self.v + dt * acc_g
            self.q = z[3:]
        # return self.p.T[0], self.v.T[0], self.q
        return self.ekf.x.reshape(3), acc_g.reshape(3), self.q
    
    def get_euler(self, u, dt=0.005):
        acc = u[:3]
        gyro = u[3:]
        r = R.from_quat(convert_quaternion(self.q))
        euler = (R.from_euler('xyz',[*(gyro * dt)], degrees=False) * r).as_euler('zxy')
        acc_euler = get_euler_from_acc(acc)
        new_euler = np.array([
            acc_euler[0],
            acc_euler[1],
            euler[2],
        ])
        return new_euler



class ExtendedKalmanFilterForPoseEstimation(nn.Module):
    def __init__(self, dim_x=10, dim_z=7, dim_u=6, dt=0.005):
        super(ExtendedKalmanFilterForPoseEstimation, self).__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = torch.zeros(dim_x, 1) # state 
        self.x[6, 0] = 1.0
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
        x: Position, velocity, Quaternion [p_x, p_y, p_z, v_x, v_y, v_z, q_w, q_x, q_y, q_z]
        """

        # Partial derivatives of theta and phi with respect to quaternion components
        H = torch.zeros((self.dim_z, self.dim_x))
        H[0:3, 0:3] = torch.eye((3, 3))
        H[3:7, 6:10] = torch.eye((3, 4))

        return H

    def Hx(self, x):
        """
        Observation function that converts state vector to position, and quaternoion.
        x: Position, velocity, Quaternion [p_x, p_y, p_z, v_x, v_y, v_z, q_w, q_x, q_y, q_z]
        x: Quaternion [q_w, q_x, q_y, q_z]
        """
        # Calculate pitch and roll from the quaternion
        H = torch.zeros((self.dim_z, self.dim_x))
        H[0:3, 0:3] = torch.eye((3, 3))
        H[3:7, 6:10] = torch.eye((3, 4))
        
        return H @ x

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
        x is the current state vector, u is the control input.
        """
        acc = u[:3]
        gyro = u[3:]
        x_predicted = x

        return x_predicted

class UnsentedKalmanFilterForPoseEstimation(nn.Module):
    def __init__(self, dim_x=10, dim_z=3, dim_u=6, dt=0.005):
        super(UnsentedKalmanFilterForPoseEstimation, self).__init__()
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

class ParticleFilterForPoseEstimation(nn.Module):
    def __init__(self, dim_x=10, dim_z=3, dim_u=6, dt=0.005):
        super(ParticleFilterForPoseEstimation, self).__init__()
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

class IMMForPoseEstimation(nn.Module):
    def __init__(self, filters, mu, M):
        if len(filters) < 2:
            raise ValueError('filters must contain at least two filters')

        self.filters = filters
        self.mu = torch.tensor(mu) / torch.sum(torch.tensor(mu))
        self.M = torch.tensor(M)

        x_shape = filters[0].x.shape
        for f in filters:
            if x_shape != f.x.shape:
                raise ValueError('All filters must have the same state dimension')

        self.x = torch.zeros_like(filters[0].x)
        self.P = torch.zeros_like(filters[0].P)
        self.N = len(filters)  # number of filters
        self.likelihood = torch.zeros(self.N)
        self.omega = torch.zeros((self.N, self.N))
        self._compute_mixing_probabilities()

        # initialize imm state estimate based on current filters
        self._compute_state_estimate()
        self.x_prior = self.x.clone()
        self.P_prior = self.P.clone()
        self.x_post = self.x.clone()
        self.P_post = self.P.clone()

    def forward(self, z, u=None):
        """
        Perform a forward pass by combining predict and update steps.

        Parameters
        ----------
        z : torch.Tensor
            Measurement for this update.

        u : torch.Tensor, optional
            Control vector. If not `None`, it is multiplied by B
            to create the control input into the system.
        """

        self.predict(u)
        self.update(z)
        return self.x_post, self.P_post

    def update(self, z):
        """
        Add a new measurement (z) to the Kalman filter. If z is None, nothing
        is changed.

        Parameters
        ----------
        z : torch.Tensor
            Measurement for this update.
        """

        # run update on each filter, and save the likelihood
        for i, f in enumerate(self.filters):
            f.update(z)
            self.likelihood[i] = f.likelihood

        # update mode probabilities from total probability * likelihood
        self.mu = self.cbar * self.likelihood
        self.mu /= torch.sum(self.mu)  # normalize

        self._compute_mixing_probabilities()

        # compute mixed IMM state and covariance and save posterior estimate
        self._compute_state_estimate()
        self.x_post = self.x.clone()
        self.P_post = self.P.clone()

    def predict(self, u=None):
        """
        Predict next state (prior) using the IMM state propagation
        equations.

        Parameters
        ----------
        u : torch.Tensor, optional
            Control vector. If not `None`, it is multiplied by B
            to create the control input into the system.
        """

        # compute mixed initial conditions
        xs, Ps = [], []
        for i, (f, w) in enumerate(zip(self.filters, self.omega.T)):
            x = torch.zeros_like(self.x)
            for kf, wj in zip(self.filters, w):
                x += kf.x * wj
            xs.append(x)

            P = torch.zeros_like(self.P)
            for kf, wj in zip(self.filters, w):
                y = kf.x - x
                P += wj * (torch.outer(y, y) + kf.P)
            Ps.append(P)

        # compute each filter's prior using the mixed initial conditions
        for i, f in enumerate(self.filters):
            # propagate using the mixed state estimate and covariance
            f.x = xs[i].clone()
            f.P = Ps[i].clone()
            f.predict(u)

        # compute mixed IMM state and covariance and save posterior estimate
        self._compute_state_estimate()
        self.x_prior = self.x.clone()
        self.P_prior = self.P.clone()

    def _compute_state_estimate(self):
        """
        Computes the IMM's mixed state estimate from each filter using
        the mode probability self.mu to weight the estimates.
        """
        self.x.fill_(0)
        for f, mu in zip(self.filters, self.mu):
            self.x += f.x * mu

        self.P.fill_(0)
        for f, mu in zip(self.filters, self.mu):
            y = f.x - self.x
            self.P += mu * (torch.outer(y, y) + f.P)

    def _compute_mixing_probabilities(self):
        """
        Compute the mixing probability for each filter.
        """

        self.cbar = torch.matmul(self.mu, self.M)
        for i in range(self.N):
            for j in range(self.N):
                self.omega[i, j] = (self.M[i, j]*self.mu[i]) / self.cbar[j]