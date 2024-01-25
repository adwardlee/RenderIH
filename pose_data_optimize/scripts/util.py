#encoding:utf-8
import math
import numpy as np

r_refine_rot = np.asarray([[ 0.5154645 ,  0.44757844, -0.73073243],
       [ 0.44752332, -0.86782812, -0.21586393],
       [-0.73076619, -0.21574961, -0.64763638]])

l_refine_rot = np.asarray([[ 0.55701023,  0.39777741, -0.7290492 ],
       [ 0.36920242, -0.90492556, -0.21165845],
       [-0.7439282 , -0.1512708 , -0.65091318]])


r_left_matrxi = np.asarray([[-1.71888690e-01,  9.84821885e-01, -2.40859543e-02],
       [-1.39159088e-01, -6.93268017e-05,  9.90270036e-01],
       [ 9.75237933e-01,  1.73567999e-01,  1.37058832e-01]])

def rot_x(angel, format='degree', dim = 3):
    if format is 'degree':
        angel = [angel * math.pi / 180.0]
    s = np.sin(angel)
    c = np.cos(angel)
    R = np.asarray([[1.0, 0.0, 0.0],
                   [0.0, c, -s],
                   [0.0, s, c]])
    out = np.eye(dim)
    out[0:3, 0:3] = R
    return out

def RightLeftAxisChange(mat):
    M = np.eye(mat.shape[0])
    M[1,1] = -1
    return M.dot(mat).dot(M)

def eulerAngles2rotationMat(theta, loc = [], format='degree', order = 'ZYX',axis='left'):
    """
    Calculates Rotation Matrix given euler angles.
    :param theta: 1-by-3 list [rx, ry, rz] angle in degree
    :return:
    RPY角，是ZYX欧拉角，依次 绕定轴XYZ转动[rx, ry, rz]
    """
    if format is 'degree':
        theta = [i * math.pi / 180.0 for i in theta]
    if axis == 'right':
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])

        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])

        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])
    else:
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), math.sin(theta[0])],
                        [0, -math.sin(theta[0]), math.cos(theta[0])]
                        ])

        R_y = np.array([[math.cos(theta[1]), 0, -math.sin(theta[1])],
                        [0, 1, 0],
                        [math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])

        R_z = np.array([[math.cos(theta[2]), math.sin(theta[2]), 0],
                        [-math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])
    # R = np.dot(R_z, np.dot(R_y, R_x))
    if order == 'ZYX':
        R = np.dot(R_x, np.dot(R_y, R_z))
    else:
        R = np.dot(R_z, np.dot(R_y, R_x))
    if loc.__len__() > 0:
        ans = np.eye(4)
        ans[0:3,0:3] = R
        ans[0:3, -1] = loc
    else:
        ans = R
    if axis == 'left':
        ans = RightLeftAxisChange(ans)
    return ans

def AxisConvertFrom3dmax2Axis(R):
    # R[:,1:3] = -R[:,1:3]
    R[1:3, :] = -R[1:3, :]
    return R
    # convert_matrix = np.eye(3)
    # convert_matrix[1,1] = -1
    # convert_matrix[2,2] = -1
    # return convert_matrix.dot(R).dot(convert_matrix)
def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z]) * 180/np.pi

def test_result(rot):
    # out = r_left_matrxi.dot(rot)
    # out = rot.dot(r_left_matrxi)
    out = rot.dot(AxisConvertFrom3dmax2Axis(r_left_matrxi))
    x,y,z = rotationMatrixToEulerAngles(out)
    print [x,-y,-z]
if __name__ == '__main__':
    lower_arm_r_euler_neuron = [-88.39, -128.277, 89.163]
    lower_arm_r_euler_unreal = [151.486, -33.188, 30.847]
    lower_arm_l_euler_neuron = [-88.457, 56.858, 90.584]
    lower_arm_l_euler_unreal = [-28.514, 33.188, -30.847]
    # lower_arm_r_euler_unreal = [-24.625, 32.761, -30.272]
    lower_arm_r_rot_neuron = eulerAngles2rotationMat(lower_arm_r_euler_neuron, [], 'degree', 'XYZ', axis='right')
    lower_arm_r_rot_unreal = eulerAngles2rotationMat(lower_arm_r_euler_unreal, [], 'degree', 'XYZ', axis='right')
    lower_arm_l_rot_neuron = eulerAngles2rotationMat(lower_arm_l_euler_neuron, [], 'degree', 'XYZ', axis='right')
    lower_arm_l_rot_unreal = eulerAngles2rotationMat(lower_arm_l_euler_unreal, [], 'degree', 'XYZ', axis='right')
    # r_rot = rot_x(55).dot(lower_arm_r_rot_neuron.dot(np.linalg.inv(lower_arm_r_rot_unreal)))
    # r_rot = np.linalg.inv(rot_x(55).dot(lower_arm_r_rot_neuron))
    # r_rot = AxisConvertFrom3dmax2Axis(rot_x(55).dot(lower_arm_r_rot_neuron)).dot(
    #     np.linalg.inv(AxisConvertFrom3dmax2Axis(lower_arm_r_rot_unreal))
    # )
    r_rot = np.linalg.inv(AxisConvertFrom3dmax2Axis(lower_arm_r_rot_unreal)).dot(AxisConvertFrom3dmax2Axis(rot_x(55 - 90).dot(lower_arm_r_rot_neuron)))
    l_rot = np.linalg.inv(AxisConvertFrom3dmax2Axis(lower_arm_l_rot_unreal)).dot(AxisConvertFrom3dmax2Axis(rot_x(55 - 90).dot(lower_arm_l_rot_neuron)))
    # l_rot = np.linalg.inv(lower_arm_l_rot_unreal).dot(rot_x(55).dot(lower_arm_l_rot_neuron))
    test_result(l_rot)
    print(r_rot)
    # R = eulerAngles2rotationMat([-13.45575, 10.385, -134.175], [], 'degree', axis='right')
    # print rotationMatrixToEulerAngles(R)