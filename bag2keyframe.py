import rosbag
import cv2
import os
import numpy as np
import bisect
from sensor_msgs.msg import CompressedImage

def load_keyframe_entries(txt_path):
    out = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            p = line.split()
            view_id = p[0]
            ts      = float(p[1])
            tx, ty, tz    = map(float, p[2:5])
            qx, qy, qz, qw= map(float, p[5:9])
            out.append((view_id, ts, tx, ty, tz, qx, qy, qz, qw))
    return out

def find_closest(ts, messages):
    times = [t for t, _ in messages]
    i = bisect.bisect_left(times, ts)
    candidates = []
    if i < len(messages): candidates.append(messages[i])
    if i > 0: candidates.append(messages[i - 1])
    return min(candidates, key=lambda x: abs(x[0] - ts))

def extract_keyframes_from_bag(bag_path, keyfile, set_num, output_dir,
                                topic_left='/zed_node/rgb/left_image_rect/compressed',
                                topic_right='/zed_node/rgb/right_image_rect/compressed'):
    # Load keyframe list
    keyframes = load_keyframe_entries(keyfile)
    kf_timestamps = [kf[1] for kf in keyframes]

    # Prepare output dirs
    out_L = os.path.join(output_dir, f'img_L_kp{set_num}')
    out_R = os.path.join(output_dir, f'img_R_kp{set_num}')
    os.makedirs(out_L, exist_ok=True)
    os.makedirs(out_R, exist_ok=True)

    # Store timestamps and messages
    imgs_L = []
    imgs_R = []

    print(f"Scanning bag for relevant image messages...")
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic_left, topic_right]):
            stamp = msg.header.stamp
            timestamp = stamp.secs + stamp.nsecs * 1e-9

            np_arr = np.frombuffer(msg.data, np.uint8)
            img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if topic == topic_left:
                imgs_L.append((timestamp, img_np))
            elif topic == topic_right:
                imgs_R.append((timestamp, img_np))

    print(f"Loaded {len(imgs_L)} left and {len(imgs_R)} right images from bag.")
    mapping_txt = os.path.join(output_dir, f'keyframe_images{set_num}.txt')

    with open(mapping_txt, 'w') as mf:
        for (view_id, ts, tx, ty, tz, qx, qy, qz, qw) in keyframes:
            best_L_ts, best_L_img = find_closest(ts, imgs_L)
            best_R_ts, best_R_img = find_closest(ts, imgs_R)

            fn_L = f"{int(best_L_ts * 1e9)}.png"
            fn_R = f"{int(best_R_ts * 1e9)}.png"

            cv2.imwrite(os.path.join(out_L, fn_L), best_L_img)
            cv2.imwrite(os.path.join(out_R, fn_R), best_R_img)

            mf.write(f"{view_id} {fn_L} {tx:.7f} {ty:.7f} {tz:.7f} "
                     f"{qx:.7f} {qy:.7f} {qz:.7f} {qw:.7f}\n")

            print(f"[{view_id}] saved keyframes: {fn_L}, {fn_R}")

    print(f"\n✅ Done. Keyframes saved to:\n  {out_L}\n  {out_R}")
    print(f"Mapping file: {mapping_txt}")


# Example usage
if __name__ == "__main__":
    bag_path   = '/home/pro/Desktop/tianqi_FYP/ROS/blue_zed2i_2025-06-12-17-04-01.bag'
    keyfile    = '/home/pro/Desktop/tianqi_FYP/ROS/KeyFrameTrajectory_TUM_Format11_views.txt'
    output_dir = '/home/pro/Desktop/tianqi_FYP/ROS'
    set_num = 11

    extract_keyframes_from_bag(bag_path, keyfile, set_num, output_dir)