import torch

def calc_u(gyro, dt):
    gyro = torch.tensor([
        [gyro[0]],
        [gyro[1]],
        [gyro[2]]
    ], dtype=torch.float32)
    u = gyro * dt
    return u

def calc_z(acc):
    acc = torch.tensor(acc)
    z = torch.tensor([
        [torch.atan(acc[1] / acc[2])],
        [-torch.atan(acc[0] / torch.sqrt(acc[1] ** 2 + acc[2] ** 2))]
    ], dtype=torch.float32)
    return z

def normalize_vector(v, return_mag=False):
    # Shape of v is assumed to be (batch, n, dim), where dim is the last dimension (e.g., 4 for quaternions)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v = v.to(device, non_blocking=True)
    batch, n, dim = v.shape
    v_mag = torch.sqrt(v.pow(2).sum(-1))  # Summing over the last dimension
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(device, non_blocking=True)))
    v_mag = v_mag.view(batch, n, 1).expand(batch, n, dim)
    v = v / v_mag
    if return_mag:
        return v, v_mag[..., 0]
    else:
        return v

def compute_rotation_matrix_from_quaternion(quaternion):
    # Shape of quaternion is assumed to be (batch, n, 4)
    batch, n, _ = quaternion.shape

    quat = normalize_vector(quaternion).contiguous()

    qw = quat[..., 0].contiguous().view(batch, n, 1)
    qx = quat[..., 1].contiguous().view(batch, n, 1)
    qy = quat[..., 2].contiguous().view(batch, n, 1)
    qz = quat[..., 3].contiguous().view(batch, n, 1)

    # Unit quaternion rotation matrices computation
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), -1)
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), -1)
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), -1)

    matrix = torch.cat((row0.unsqueeze(-2), row1.unsqueeze(-2), row2.unsqueeze(-2)), -2)

    return matrix

def rotation_matrix_to_6d(rotation_matrix):
    """
    Convert a rotation matrix to its 6D representation.
    Assumes the input rotation_matrix has shape (batch, n, 3, 3).
    """
    # Extract the first two columns of the rotation matrix
    b1 = rotation_matrix[..., 0]
    b2 = rotation_matrix[..., 1]

    # Concatenate b1 and b2 to form the 6D representation
    representation_6d = torch.cat((b1, b2), -1)

    return representation_6d


def sixd_to_rotation_matrix(sixd_representation):
    """
    Convert a 6D representation to a rotation matrix.
    """
    # Assuming the first two vectors (each 3D) of the 6D representation are orthogonal and normalized
    b1 = sixd_representation[:3]
    b2 = sixd_representation[3:6]

    # Compute the third column as the cross product of the first two
    b3 = torch.cross(b1, b2)

    # Form the rotation matrix
    rotation_matrix = torch.stack([b1, b2, b3], dim=1)

    return rotation_matrix

def rotation_matrix_to_quaternion(rotation_matrix):
    """
    Convert a rotation matrix to a quaternion.
    """
    # Calculate the quaternion components from the rotation matrix
    w = torch.sqrt(1.0 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]) / 2
    x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4 * w)
    y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4 * w)
    z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4 * w)

    return torch.tensor([w, x, y, z])
