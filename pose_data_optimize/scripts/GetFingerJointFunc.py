import numpy as np
import matplotlib.pyplot as plt

last_finger_joints =    np.asarray([0, 5, 20, 45, 80, 85, 90]) / 180 * np.pi
scecond_finger_joints = np.asarray([0, 20, 45, 80, 90, 110, 120]) / 180 * np.pi
f1 = np.polyfit(last_finger_joints, scecond_finger_joints, 2)
p1 = np.poly1d(f1)
print('p1 is :\n',p1)
last_finger_joints = np.asarray([-20, -10, -5, 0, 5, 20, 45, 80, 85, 90]) / 180 * np.pi
z = p1(last_finger_joints)
#绘图
# plot1 = plt.plot(last_finger_joints, scecond_finger_joints, 'o',label='original values')
plot2 = plt.plot(last_finger_joints * 180/np.pi, z * 180/np.pi, 'r-',label='polyfit values')
plt.xlabel('x1')
plt.ylabel('y1')
plt.legend(loc=4) #指定legend的位置右下角
plt.title('polyfitting')
plt.show()