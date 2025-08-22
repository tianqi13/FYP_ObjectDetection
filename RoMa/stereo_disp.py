import os, sys
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn.functional as F
import open3d as o3d
import numpy as np
import cv2
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from romatch import roma_outdoor
from romatch.utils.utils import tensor_to_pil

# ─────────────── device ────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")

class RoMaMatcher:
    def __init__(self, coarse_res=420, upsample_res=(720, 1280)):
        """
        Initialize the RoMa matcher with specified resolutions.
        :param coarse_res: Coarse resolution for the model.
        :param upsample_res: Resolution to which the output will be upsampled.
        """
        HERE = os.path.dirname(__file__)
        self.intrinsic_file = os.path.join(HERE, 'K.txt')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        
        self.model = roma_outdoor(device=device, coarse_res=coarse_res, upsample_res=upsample_res)
        self.H, self.W = self.model.get_output_resolution()
        self.device = device
    
    def grid_to_pixel(self, grid_xy, W, H):
        """
        Convert a grid_sample grid (align_corners=False) to pixel coords.
        grid_xy : [H,W,2] in [-1,1]
        return  : (x_pix, y_pix)  each [H,W]
        """
        xi  = grid_xy[..., 0]
        eta = grid_xy[..., 1]
        x_pix = (xi + 1.0) * W * 0.5 + 0.5      
        y_pix = (eta + 1.0) * H * 0.5 + 0.5
        return x_pix, y_pix

    def get_disparity(self, im_A_path, im_B_path, cert_thr=0.5):
        warp, certainty = self.model.match(im_A_path, im_B_path, device=self.device)
        
        # left-image pixels: warp[: , :W , 2:]  → coords in right image
        grid_lr = warp[:, :self.W, 2:]          # [H,W,2] in [-1,1]

        # (1) convert to absolute pixel coords in the right view
        xB, yB = self.grid_to_pixel(grid_lr, self.W, self.H)

        # (2) pixel grid of the left view
        u = torch.arange(self.W, device=device).view(1, self.W).expand(self.H, self.W).float()
        v = torch.arange(self.H, device=device).view(self.H, 1).expand(self.H, self.W).float()

        # (3) disparity:   d = u − xB    (positive when point shifts right→left)
        disp = u - xB                        # [H,W]  float32  

        cert_left = certainty[:, :self.W]                   # ←  keep only the left view
        mask_cert = cert_left > cert_thr
        mask_epi  = (torch.abs(v - yB) < 5.0)
        valid     = mask_cert & mask_epi
        disp  = torch.where(valid, disp, 0.0)
        disp = disp.cpu().numpy()

        return disp
    
    def visualise_disparity(self, pred_disp, output_dir):
        pred_disp = (pred_disp - pred_disp.min()) / (pred_disp.max() - pred_disp.min())
        pred_disp = (pred_disp * 255).astype(np.uint8)
        pred_disp = cv2.applyColorMap(pred_disp, cv2.COLORMAP_INFERNO)
        cv2.imwrite(f'{output_dir}/vis.png', pred_disp)
    
    def depth2xyzmap(self, depth:np.ndarray, K, uvs:np.ndarray=None, zmin=0.1):
        invalid_mask = (depth<zmin)
        H,W = depth.shape[:2]
        if uvs is None:
            vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
            vs = vs.reshape(-1)
            us = us.reshape(-1)
        else:
            uvs = uvs.round().astype(int)
            us = uvs[:,0]
            vs = uvs[:,1]
        zs = depth[vs,us]
        xs = (us-K[0,2])*zs/K[0,0]
        ys = (vs-K[1,2])*zs/K[1,1]
        pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
        xyz_map = np.zeros((H,W,3), dtype=np.float32)
        xyz_map[vs,us] = pts
        if invalid_mask.any():
            xyz_map[invalid_mask] = 0
        return xyz_map

    def toOpen3dCloud(self, points,colors=None,normals=None):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if colors is not None:
            if colors.max()>1:
                colors = colors/255.0
                cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        if normals is not None:
            cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
        return cloud
    
    def remove_invisible(self, disp):
        yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
        us_right = xx - disp
        invalid = us_right < 0
        disp[invalid] = np.inf
        return disp

    def get_point_cloud(self, disp, frame0_ori, output_dir, scale=1):
        with open(self.intrinsic_file, 'r') as f:
            lines = f.readlines()
            K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
            baseline = float(lines[1])
        K[:2] *= scale
        
        valid = disp > 0
        if not np.any(valid):
            raise RuntimeError("No valid disparity pixels > 0!")
        depth = np.zeros_like(disp, dtype=np.float32)
        depth[valid] = baseline * K[0, 0] / disp[valid]        
        
        xyz_map = self.depth2xyzmap(depth, K)
        points = xyz_map.reshape(-1, 3)
        
        # toOpen3dCloud expects colors in RGB format
        frame0_ori = cv2.cvtColor(frame0_ori, cv2.COLOR_BGR2RGB)
        colors = frame0_ori.reshape(-1, 3)  
        
        
        keep_mask = (points[:,2] > 0) & (points[:,2] <= 10)
        pts_filt   = points[keep_mask]     
        cols_filt  = colors[keep_mask]   
        
        ones = np.ones((pts_filt.shape[0], 1))
        points_hom = np.hstack([pts_filt, ones])  # shape (N, 4)

        points_world_hom = (T_mat @ points_hom.T).T
        points_world = points_world_hom[:, :3]
        
        pcd = self.toOpen3dCloud(points_world, cols_filt)
        
        o3d.io.write_point_cloud(output_dir+'/cloudRoMa.ply', pcd)
    
    def get_a_point_cloud(self, disp, frame0_ori, T_mat, scale=1):
        with open(self.intrinsic_file, 'r') as f:
            lines = f.readlines()
            K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
            baseline = float(lines[1])
        K[:2] *= scale
        
        valid = disp > 0
        if not np.any(valid):
            raise RuntimeError("No valid disparity pixels > 0!")
        depth = np.zeros_like(disp, dtype=np.float32)
        depth[valid] = baseline * K[0, 0] / disp[valid]        
        
        xyz_map = self.depth2xyzmap(depth, K)
        points = xyz_map.reshape(-1, 3)
        
        # toOpen3dCloud expects colors in RGB format
        frame0_ori = cv2.cvtColor(frame0_ori, cv2.COLOR_BGR2RGB)
        colors = frame0_ori.reshape(-1, 3)  
        
        
        keep_mask = (points[:,2] > 0) & (points[:,2] <= 10)
        pts_filt   = points[keep_mask]     
        cols_filt  = colors[keep_mask]   
        
        ones = np.ones((pts_filt.shape[0], 1))
        points_hom = np.hstack([pts_filt, ones])  # shape (N, 4)

        points_world_hom = (T_mat @ points_hom.T).T
        points_world = points_world_hom[:, :3]
        
        pcd = self.toOpen3dCloud(points_world, cols_filt)
        
        return pcd

# ─────────────── main ────────────────
if __name__ == "__main__":
    img_L_path = "img_L_valve.png"
    img_R_path = "img_R_valve.png"
    HERE = os.path.dirname(__file__)
    output_dir = HERE

    matcher = RoMaMatcher(coarse_res=420, upsample_res=(720, 1280))
    pred_disp = matcher.get_disparity(img_L_path, img_R_path)

    img_L = cv2.imread(img_L_path)

    matcher.visualise_disparity(pred_disp, output_dir)
    matcher.get_point_cloud(disp=pred_disp, frame0_ori=img_L, output_dir=output_dir, scale=1.0)
    
