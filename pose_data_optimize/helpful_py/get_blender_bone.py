import numpy as np
from scipy.spatial.transform.rotation import Rotation
target_quat = np.asarray([ 0.9999, -0.0117,  0.0078, -0.0060])
r = Rotation.from_quat(target_quat)