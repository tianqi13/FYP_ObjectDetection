import open3d as o3d
import numpy as np 

pcd = o3d.io.read_point_cloud("point_clouds/all_pc.ply")
print(f"Point cloud before voxel sampling: {len(pcd.points)}")
voxel_size = 0.05
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
print(f"Point cloud after voxel sampling: {len(pcd.points)}")

o3d.io.write_point_cloud("point_clouds/voxel_all0.05.ply", pcd)