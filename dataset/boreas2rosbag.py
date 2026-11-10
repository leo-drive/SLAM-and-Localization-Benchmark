#!/usr/bin/env python3
"""
Boreas-RT to ROSBag Converter (Unified Benchmark Edition)
"""

import os
import glob
import gc
import argparse
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

MIN_ROOF_DISTANCE_SQ = 3 ** 2

def extract_extrinsics(sequence_path):
    """Computes T_lidar_dmu using Applanix as the intermediary frame."""
    t_al_file = os.path.join(sequence_path, 'calib', 'T_applanix_lidar.txt')
    t_ad_file = os.path.join(sequence_path, 'calib', 'T_applanix_dmu.txt')
    
    if not (os.path.exists(t_al_file) and os.path.exists(t_ad_file)):
        print(f"[WARN] Calibration files missing in {sequence_path}/calib/. Using Identity.")
        return np.eye(4)
        
    T_a_l = np.loadtxt(t_al_file)
    T_a_d = np.loadtxt(t_ad_file)
    
    # T_lidar_dmu = inverse(T_applanix_lidar) * T_applanix_dmu
    T_lidar_dmu = np.linalg.inv(T_a_l) @ T_a_d
    print(f"\n[INFO] Computed T_lidar_dmu for SLAM YAML Configs:\n{T_lidar_dmu}\n")
    return T_lidar_dmu

def create_pointcloud2(typestore, header, points):
    PointCloud2 = typestore.types['sensor_msgs/msg/PointCloud2']
    PointField = typestore.types['sensor_msgs/msg/PointField']
    
    fields = [
        PointField(name='x', offset=0, datatype=7, count=1),           
        PointField(name='y', offset=4, datatype=7, count=1),
        PointField(name='z', offset=8, datatype=7, count=1),
        PointField(name='intensity', offset=12, datatype=7, count=1),
        PointField(name='ring', offset=16, datatype=4, count=1),       
        PointField(name='time', offset=18, datatype=7, count=1)
    ]
    
    cloud_dtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('intensity', '<f4'), ('ring', '<u2'), ('time', '<f4')
    ])
    
    structured_points = np.empty(points.shape[0], dtype=cloud_dtype)
    structured_points['x'] = points[:, 0]
    structured_points['y'] = points[:, 1]
    structured_points['z'] = points[:, 2]
    structured_points['intensity'] = points[:, 3]
    structured_points['ring'] = np.clip(np.nan_to_num(points[:, 4]), 0, 127).astype(np.uint16)
    structured_points['time'] = points[:, 5]
    
    buffer = structured_points.tobytes()
    return PointCloud2(
        header=header, height=1, width=points.shape[0], fields=fields,
        is_bigendian=False, point_step=22, row_step=22 * points.shape[0],
        data=np.array(memoryview(buffer), dtype=np.uint8), is_dense=True
    )

def index_dataset(seq_dir):
    """Indexes files and records to determine the exact total length for the progress bar."""
    lidar_dir = os.path.join(seq_dir, 'lidar')
    imu_file = os.path.join(seq_dir, 'imu', 'dmu_imu_infilled.csv')
    
    lidar_files = []
    if os.path.exists(lidar_dir):
        for f in glob.glob(os.path.join(lidar_dir, '*.bin')):
            lidar_files.append((int(os.path.basename(f).replace('.bin', '')) * 1000, 'lidar', f))
    lidar_files.sort(key=lambda x: x[0])

    imu_records = []
    if os.path.exists(imu_file):
        imu_raw = np.loadtxt(imu_file, delimiter=',', skiprows=1)
        for row in imu_raw:
            imu_records.append((int(row[0]), 'imu', row[1:7]))
            
    return lidar_files, imu_records

def stream_events(lidar_files, imu_records):
    """Yields interleaved events based on pre-indexed lists."""
    l_idx, i_idx = 0, 0
    num_l, num_i = len(lidar_files), len(imu_records)

    while l_idx < num_l or i_idx < num_i:
        if i_idx >= num_i or (l_idx < num_l and lidar_files[l_idx][0] <= imu_records[i_idx][0]):
            yield lidar_files[l_idx]
            l_idx += 1
        else:
            yield imu_records[i_idx]
            i_idx += 1

def write_ros1_bag(dataset_dir, output_bag):
    extract_extrinsics(dataset_dir)
    
    print("[INFO] Indexing dataset to calculate progress total...")
    lidar_files, imu_records = index_dataset(dataset_dir)
    total_events = len(lidar_files) + len(imu_records)
    print(f"[INFO] Found {len(lidar_files)} LiDAR scans and {len(imu_records)} IMU records.")
    
    typestore = get_typestore(Stores.ROS1_NOETIC)
    Header = typestore.types['std_msgs/msg/Header']
    Time = typestore.types['builtin_interfaces/msg/Time']  
    Imu = typestore.types['sensor_msgs/msg/Imu']
    Vector3 = typestore.types['geometry_msgs/msg/Vector3']
    Quaternion = typestore.types['geometry_msgs/msg/Quaternion']

    with Writer(output_bag) as writer:
        imu_conn = writer.add_connection('/dmu/imu', Imu.__msgtype__, typestore=typestore)
        pc_conn = writer.add_connection('/velodyne/points', typestore.types['sensor_msgs/msg/PointCloud2'].__msgtype__, typestore=typestore)
        
        # tqdm now has the total count and will display a percentage bar with ETA
        event_stream = stream_events(lidar_files, imu_records)
        for idx, (ts_ns, sensor_type, payload) in enumerate(tqdm(event_stream, total=total_events, desc="Converting Bag", unit="msgs")):
            sec, nanosec = int(ts_ns // 10**9), int(ts_ns % 10**9)
            
            if sensor_type == 'lidar':
                raw_points = np.fromfile(payload, dtype=np.float32).reshape(-1, 6)
                dist_sq = raw_points[:, 0]**2 + raw_points[:, 1]**2 + raw_points[:, 2]**2
                valid_pts = raw_points[dist_sq > MIN_ROOF_DISTANCE_SQ]
                
                point_times = np.nan_to_num(valid_pts[:, 5])
                valid_pts[:, 5] = point_times - point_times.min()
                
                header = Header(seq=idx, stamp=Time(sec=sec, nanosec=nanosec), frame_id='lidar_link')
                pc_msg = create_pointcloud2(typestore, header, valid_pts)
                writer.write(pc_conn, ts_ns, typestore.serialize_ros1(pc_msg, pc_msg.__msgtype__))
                
            elif sensor_type == 'imu':
                wx, wy, wz, ax, ay, az = payload
                pitch = np.arctan2(ax, np.sqrt(ay**2 + az**2))
                roll = np.arctan2(-ay, az)
                qx, qy, qz, qw = R.from_euler('xyz', [roll, pitch, 0.0]).as_quat()
                
                header = Header(seq=idx, stamp=Time(sec=sec, nanosec=nanosec), frame_id='imu_link')
                imu_msg = Imu(
                    header=header,
                    orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
                    orientation_covariance=np.zeros(9, dtype=np.float64),
                    angular_velocity=Vector3(x=wx, y=wy, z=wz),
                    angular_velocity_covariance=np.zeros(9, dtype=np.float64),
                    linear_acceleration=Vector3(x=ax, y=ay, z=az),
                    linear_acceleration_covariance=np.zeros(9, dtype=np.float64)
                )
                writer.write(imu_conn, ts_ns, typestore.serialize_ros1(imu_msg, Imu.__msgtype__))
            
            if idx % 1000 == 0: gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert Boreas-RT sequence to a benchmark-ready ROS 1 Bag.")
    parser.add_argument("dataset_dir", type=str, help="Path to the extracted Boreas sequence directory.")
    parser.add_argument("--output", type=str, default="boreas_benchmark.bag", help="Path to the output .bag file.")
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_dir):
        raise FileNotFoundError(f"Provided dataset directory does not exist: {args.dataset_dir}")
        
    write_ros1_bag(args.dataset_dir, args.output)
