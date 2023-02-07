import open3d as o3d
import numpy as np
import torch

class ICP():
    
    def __init__(self):
        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        self.yaw_init = np.radians(30)

    def compute_delta_states(self, batch_pairs, action):
        
        rgb_raw = batch_pairs["rgb"].cpu().numpy()
        prev_rgb = o3d.geometry.Image(rgb_raw[0,:,:,:3])
        cur_rgb = o3d.geometry.Image(rgb_raw[0,:,:,3:])


        depth_raw = batch_pairs["depth"].cpu().numpy()
        prev_depth = o3d.geometry.Image(depth_raw[0,:,:,0].astype(np.float32))
        cur_depth = o3d.geometry.Image(depth_raw[0,:,:,1].astype(np.float32))

        # 1 = [0., 0., -0.25] [0., 0., 0., 1.] 
        # 2 = [0., 0., 0.] [0., 0.258819, 0., 0.965925]
        # 3 = [0., 0., 0.] [0., -0.258819, 0., 0.965925]

        prev_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(prev_rgb, prev_depth, convert_rgb_to_intensity=True)
        cur_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(cur_rgb, cur_depth, convert_rgb_to_intensity=True)

        source = o3d.geometry.PointCloud.create_from_rgbd_image(prev_rgbd, self.pinhole_camera_intrinsic)
        target = o3d.geometry.PointCloud.create_from_rgbd_image(cur_rgbd, self.pinhole_camera_intrinsic)

        if action == 1:
            trans_init = np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,-0.25e-4],[0,0,0,1]])
        elif action == 2:
            trans_init = np.asarray([[np.cos(self.yaw_init),0,np.sin(self.yaw_init),0],[0,1,0,0],[-np.sin(self.yaw_init),0,np.cos(self.yaw_init),0],[0,0,0,1]])
        elif action == 3:
            trans_init = np.asarray([[np.cos(-self.yaw_init),0,np.sin(-self.yaw_init),0],[0,1,0,0],[-np.sin(-self.yaw_init),0,np.cos(-self.yaw_init),0],[0,0,0,1]])

        threshold = 1e-2
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_rmse=1.e-10))

        tmp = reg_p2p.transformation
        # y rotation
        if action == 1:
            beta = np.arccos(tmp[0,0])
        elif action == 2:
            beta = np.arccos(tmp[0,0])
        elif action == 3:
            beta = -np.arccos(tmp[0,0])
            
        trans = np.array([tmp[0,3], tmp[1,3], tmp[2,3]]) * 1e+4
        
        
        # x, z, yaw
        return [trans[0], trans[2], beta]