import numpy as np
from scipy.spatial.transform import Rotation as R

def get_rotation_matrix(quat):
    # quat = [qx, qy, qz, qw]
    return R.from_quat(quat).as_matrix()

def get_transformation_matrix(file_line: str):
    parts = file_line.strip().split()
    if len(parts) != 9:
        raise ValueError(f"Unexpected column count ({len(parts)}): {file_line!r}")
    
    timestamp = parts[1]
    tx, ty, tz = map(float, parts[2:5])
    # print(f"Translation (World to Camera): tx={tx}, ty={ty}, tz={tz}")
    qx, qy, qz, qw = map(float, parts[5:9])
    # print(f"Quaternion (World to Camera): qx={qx}, qy={qy}, qz={qz}, qw={qw}")
    
    R_cw = get_rotation_matrix([qx, qy, qz, qw])
    # print(f"Rotation matrix (Camera to World):\n{R_cw}")
    
    R_wc = R_cw.T
    # print(f"Rotation matrix (World to Camera):\n{R_wc}")
    
    t_cw = np.array([tx, ty, tz], dtype=np.float64)
    # print(f"Translation vector (Camera to World): {t_cw}")
    t_wc = - (R_wc @ t_cw)
    # print(f"Translation vector (World to Camera): {t_wc}")
    
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_wc
    T[:3,  3] = t_wc
    # print(f"Transformation matrix (World to Camera):\n{T}")
    
    return timestamp, T

if __name__ == "__main__":
    get_transformation_matrix('3 1749580777788282624.png -0.0464696 0.0766286 0.0852775 -0.0164602 -0.0051737 -0.0070875 0.9998260')

