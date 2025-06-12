import open3d as o3d
import glob
import os

# Set voxel size
voxel_size = 0.001

# Find all .ply files matching the pattern
ply_files = glob.glob("combined_pc*.ply")

print(f"Found {len(ply_files)} point clouds to process.")

for ply_path in ply_files:
    print(f"Processing: {ply_path}")
    
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_path)
    
    # Downsample
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Create output filename
    base, ext = os.path.splitext(ply_path)
    out_path = f"{base}_downsampled.ply"
    
    # Save downsampled point cloud
    o3d.io.write_point_cloud(out_path, down_pcd)
    print(f"Saved downsampled point cloud to: {out_path}")

print("✅ All point clouds processed.")
