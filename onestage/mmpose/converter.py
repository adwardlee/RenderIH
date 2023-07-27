import numpy as np

__version__ = '1.0.0'

EQUAL_THRESHOLD=1e-3

def isOrthogonal(m):
    mt = m.T
    result = np.matmul(m,mt)
    idt = np.identity(3)
    result = np.mean(np.abs(result-idt))
    return result<EQUAL_THRESHOLD

def checkMatrix(m):
    if isinstance(m,list):
        m = np.array(m,dtype=np.float32)
    if m.ndim==1 and np.shape[0]==9:
        m = m.reshape(3,3)

    if m.ndim!=2 or m.shape[0]!=3 or m.shape[1]!=3:
        print("Matrix not support, only support (1*9 or 3*3) matrix")
        return None
    if not isOrthogonal(m):
        print("Matrix is not orthogonal")
        return None
    return m

def checkEuler(euler_angle):
    if isinstance(euler_angle,list):
        euler_angle = np.array(euler_angle,np.float32)
    if euler_angle.ndim!=1 or euler_angle.shape[0]!=3:
        print("Only support euler angle (len==3)")
        return None
    return euler_angle

def checkAxis(axis_angle,with_magnitude):
    if isinstance(axis_angle,list):
        axis_angle = np.array(axis_angle,dtype=np.float32)
    #print(axis_angle,with_magnitude)
    if axis_angle.ndim!=1 or (axis_angle.shape[0]!=3 and with_magnitude) or (axis_angle.shape[0]!=4 and not with_magnitude):
        print("Only support axis angle (len==3 with magnitude or len=4 without magnitude)")
        return None
    return axis_angle

def checkQuaternion(quaternion):
    if isinstance(quaternion,list):
        quaternion = np.array(quaternion,np.float32)
        quaternion = quaternion.reshape(-1)
    if quaternion.shape[0]!=4:
        print("quaternion len must be 4")
        return None
    n = np.linalg.norm(quaternion)
    if np.abs(n-1)>EQUAL_THRESHOLD:
        print("quaternion must normalized")
        return None
    w = np.sin(np.arccos(quaternion[0]))
    n = np.linalg.norm(quaternion[1:])
    #print(n/w)
    #print(np.abs(w)<EQUAL_THRESHOLD and np.abs(n)<EQUAL_THRESHOLD)
    if  (np.abs(w)<EQUAL_THRESHOLD and np.abs(n)>EQUAL_THRESHOLD) or  (np.abs(w)>EQUAL_THRESHOLD and np.abs(n/w-1)>EQUAL_THRESHOLD):
        print("quaternion must like [cos(theta/2),sin(theta/2)*ux,sin(theta/2)*uy,sin(theta/2)*uz] and u is normalized")

    return quaternion


def euler2quaternion(euler_angle,order='xyz'):
    euler_angle = checkEuler(euler_angle)
    if euler_angle is None:
        return None
    c = np.cos(euler_angle[0] / 2)
    d = np.cos(euler_angle[1]/ 2)
    e = np.cos(euler_angle[2] / 2)
    f = np.sin(euler_angle[0] / 2)
    g = np.sin(euler_angle[1]/ 2)
    h = np.sin(euler_angle[2]/ 2)
    order = order.upper()
    if "XYZ" == order:
        x = f * d * e + c * g * h
        y = c * g * e - f * d * h
        z = c * d * h + f * g * e
        w = c * d * e - f * g * h
    elif "YXZ" == order:
        x = f * d * e + c * g * h
        y = c * g * e - f * d * h
        z = c * d * h - f * g * e
        w = c * d * e + f * g * h
    elif "ZXY" == order:
        x = f * d * e - c * g * h
        y = c * g * e + f * d * h
        z = c * d * h + f * g * e
        w = c * d * e - f * g * h
    elif "ZYX" == order:
        x = f * d * e - c * g * h
        y = c * g * e + f * d * h
        z = c * d * h - f * g * e
        w = c * d * e + f * g * h
    elif "YZX" == order:
        x = f * d * e + c * g * h
        y = c * g * e + f * d * h
        z = c * d * h - f * g * e
        w = c * d * e - f * g * h
    elif "XZY" == order:
        x = f * d * e - c * g * h
        y = c * g * e - f * d * h
        z = c * d * h + f * g * e
        w = c * d * e + f * g * h

    quaternion = np.array([w,x,y,z],dtype=np.float32)
    return quaternion

def euler2axis(euler_angle,order='xyz',with_magnitude=False):
    euler_angle = checkEuler(euler_angle)
    if euler_angle is None:
        return None
    quaternion = euler2quaternion(euler_angle,order=order)
    axis = quaternion2axis(quaternion,with_magnitude=with_magnitude)
    return axis

def euler2matrix(euler_angle,order='xyz'):
    euler_angle = checkEuler(euler_angle)
    if euler_angle is None:
        return None
    b = np.zeros(9)
    c = euler_angle[0]
    d = euler_angle[1]
    e = euler_angle[2]
    f = np.cos(c)
    c = np.sin(c)
    g = np.cos(d)
    d = np.sin(d)
    h = np.cos(e)
    e = np.sin(e)
    order = order.upper()
    if "XYZ" == order:
        a = f * h
        k = f * e
        l = c * h
        m = c * e
        b[0] = g * h
        b[1] = -g * e
        b[2] = d
        b[3] = k + l * d
        b[4] = a - m * d
        b[5] = -c * g
        b[6] = m - a * d
        b[7] = l + k * d
        b[8] = f * g
    elif "YXZ" == order:
        a = g * h
        k = g * e
        l = d * h
        m = d * e
        b[0] = a + m * c
        b[1] = l * c - k
        b[2] = f * d
        b[3] = f * e
        b[4] = f * h
        b[5] = -c
        b[6] = k * c - l
        b[7] = m + a * c
        b[8] = f * g
    elif "ZXY" == order:
        a = g * h
        k = g * e
        l = d * h
        m = d * e
        b[0] = a - m * c
        b[1] = -f * e
        b[2] = l + k * c
        b[3] = k + l * c
        b[4] = f * h
        b[5] = m - a * c
        b[6] = -f * d
        b[7] = c
        b[8] = f * g
    elif "ZYX" == order:
        a = f * h
        k = f * e
        l = c * h
        m = c * e
        b[0] = g * h
        b[1] = l * d - k
        b[2] = a * d + m
        b[3] = g * e
        b[4] = m * d + a
        b[5] = k * d - l
        b[6] = -d
        b[7] = c * g
        b[8] = f * g
    elif "YZX" == order:
        a = f * g
        k = f * d
        l = c * g
        m = c * d
        b[0] = g * h
        b[1] = m - a * e
        b[2] = l * e + k
        b[3] = e
        b[4] = f * h
        b[5] = -c * h
        b[6] = -d * h
        b[7] = k * e + l
        b[8] = a - m * e
    elif "XZY" == order:
        a = f * g
        k = f * d
        l = c * g
        m = c * d
        b[0] = g * h
        b[1] = -e
        b[2] = d * h
        b[3] = a * e + m
        b[4] = f * h
        b[5] = k * e - l
        b[6] = l * e - k
        b[7] = c * h
        b[8] = m * e + a
    else:
        print("Not support the order",order)
        return None
    R = b.reshape(3,3)
    return R


def matrix2euler(m,order='xyz'):
    m = checkMatrix(m)
    if m is None:
        return None
    d = np.clip
    e = m.reshape(-1)
    a = e[0]
    f = e[1]
    g = e[2]
    h = e[3]
    k = e[4]
    l = e[5]
    m = e[6]
    n = e[7]
    e = e[8]
    order = order.upper()
    if "XYZ" == order:
        y = np.arcsin(d(g, -1, 1))
        if 0.99999 > np.abs(g):
            x = np.arctan2( - l, e)
            z = np.arctan2( - f, a)
        else:
            x = np.arctan2(n, k)
            z = 0
    elif "YXZ" == order:
        x = np.arcsin(- d(l, -1, 1))
        if 0.99999 > np.abs(l):
            y = np.arctan2(g, e)
            z = np.arctan2(h, k)
        else:
            y = np.arctan2( - m, a)
            z = 0
    elif "ZXY" ==order:
        x = np.arcsin(d(n, -1, 1))
        if 0.99999 > np.abs(n):
            y = np.arctan2( - m, e)
            z = np.arctan2( - f, k)
        else:
            y = 0
            z = np.arctan2(h, a)
    elif "ZYX" ==order:
        y = np.arcsin( - d(m, -1, 1))
        if 0.99999 > np.abs(m):
            x = np.arctan2(n, e)
            z = np.arctan2(h, a)
        else:
            x = 0
            z = np.arctan2( - f, k)
    elif "YZX" ==order:
        z = np.arcsin(d(h, -1, 1))
        if 0.99999 > np.abs(h):
            x = np.arctan2( - l, k)
            y = np.arctan2( - m, a)
        else:
            x = 0
            y = np.arctan2(g, e)
    elif "XZY" ==order:
        z = np.arcsin( - d(f, -1, 1))
        if 0.99999 > np.abs(f):
            x = np.arctan2(n, k)
            y = np.arctan2(g, a)
        else:
            x = np.arctan2( - l, e)
            y = 0
    else:
        print("not support order",order)
        return None
    return np.array([x,y,z],dtype=np.float32)


def matrix2quaternion(m):
    m = checkMatrix(m)
    if m is None:
        return None
    b = m.reshape(-1)
    c = b[0]
    a = b[1]
    d = b[2]
    e = b[3]
    f = b[4]
    g = b[5]
    h = b[6]
    k = b[7]
    b = b[8]
    l = c + f + b;
    if 0 < l:
        c = .5 / np.sqrt(l + 1)
        w = .25 / c
        x = (k - g) * c
        y = (d - h) * c
        z = (e - a) * c
    elif c > f and c > b:
        c = 2 * np.sqrt(1 + c - f - b)
        w = (k - g) / c
        x = .25 * c
        y = (a + e) / c
        z = (d + h) / c
    elif f > b:
        c = 2 * np.sqrt(1 + f - c - b)
        w = (d - h) / c
        x = (a + e) / c
        y = .25 * c
        z = (g + k) / c
    else:
        c = 2 * np.sqrt(1 + b - c - f)
        w = (e - a) / c
        x = (d + h) / c
        y = (g + k) / c
        z = .25 * c
    quaternion = np.array([w,x,y,z],dtype=np.float32)
    return quaternion

def matrix2axis(m,with_magnitude=False):
    m = checkMatrix(m)
    if m is None:
        return None
    quaternion = matrix2quaternion(m)
    axis = quaternion2axis(quaternion,with_magnitude=with_magnitude)
    return axis


def quaternion2axis(quaternion,with_magnitude=False):
    quaternion = checkQuaternion(quaternion)
    if quaternion is None:
        return None
    theta = np.arccos(quaternion[0])
    theta_s = np.sin(theta)
    if theta_s>EQUAL_THRESHOLD:
        u = [q/theta_s for q in quaternion[1:]]
    else:
        u = [1.0,0.0,0.0]
    theta *=2
    if with_magnitude:
        axis_angle = np.array(u,dtype=np.float32)*theta
    else:
        axis_angle = np.array(u+[theta],dtype=np.float32)
    return axis_angle

def quaternion2matrix(quaternion):
    quaternion = checkQuaternion(quaternion)
    if quaternion is None:
        return None
    a = quaternion[0]
    b = quaternion[1]
    c = quaternion[2]
    d = quaternion[3]
    m = np.array([[1-2*(c*c+d*d),2*b*c-2*a*d,2*a*c+2*b*d],
             [2*b*c+2*a*d,1-2*(b*b+d*d),2*c*d-2*a*b],
             [2*b*d-2*a*c,2*a*b+2*c*d,1-2*(b*b+c*c)]],dtype=np.float32)
    return m

def quaternion2euler(quaternion,order='xyz'):
    quaternion = checkQuaternion(quaternion)
    if quaternion is None:
        return None
    m = quaternion2matrix(quaternion)
    euler_angle = matrix2euler(m,order=order)
    return euler_angle

def axis2quaternion(axis_angle,with_magnitude=False):
    axis_angle = checkAxis(axis_angle,with_magnitude)
    if axis_angle is None:
        return None
    if with_magnitude:
        angle = np.linalg.norm(axis_angle)
        axis_angle = axis_angle/angle
    else:
        angle = axis_angle[3]
        axis_angle = axis_angle[:3]
        axis_angle /=np.linalg.norm(axis_angle)
    a = np.cos(angle/2)
    s = np.sin(angle/2)
    b,c,d = axis_angle*s
    quaternion = np.array([a,b,c,d],np.float32)
    return quaternion

def axis2matrix(axis_angle,with_magnitude=False):
    axis_angle = checkAxis(axis_angle,with_magnitude)
    if axis_angle is None:
        return None

    quaternion = axis2quaternion(axis_angle,with_magnitude=with_magnitude)
    m = quaternion2matrix(quaternion)
    return m

def axis2euler(axis_angle,with_magnitude=False,order='xyz'):
    axis_angle = checkAxis(axis_angle,with_magnitude)
    if axis_angle is None:
        return None

    quaternion = axis2quaternion(axis_angle,with_magnitude=with_magnitude)
    euler_angle = quaternion2euler(quaternion,order=order)
    return euler_angle