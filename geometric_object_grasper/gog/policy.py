import open3d as o3d
import numpy as np
import torch

from gog import cameras
from gog.utils import pybullet_utils
from gog.utils.o3d_tools import PointCloud, VoxelGrid, get_reconstructed_surface
from gog.grasp_sampler import GraspSampler
from gog.shape_completion import ShapeCompletionNetwork
from gog.grasp_optimizer import SplitPSO


class GraspingPolicy:
    def __init__(self, robot_hand, params):
        self.robot_hand = robot_hand
        # self.config = cameras.RealSense.CONFIG[0]
        self.config = cameras.RealSense.CONFIG
        # self.bounds = np.array([[-0.25, 0.25], [-0.25, 0.25], [0.0, 1]])  # workspace limits
        self.bounds = np.array([[-0.5, 0.5], [-0.5, 0.5], [0, 1]])  # workspace limits

        self.grasp_sampler = GraspSampler(robot=robot_hand, params=params['grasp_sampling'])

        self.vae = ShapeCompletionNetwork(input_dim=[5, 32, 32, 32], latent_dim=512).to('cuda')
        self.vae.load_state_dict(torch.load(params['vae']['model_weights']))
        print('Shape completion model loaded....!')
        self.vae.eval()

        self.optimizer = SplitPSO(robot_hand, params['power_grasping']['optimizer'])

    def seed(self, seed):
        self.grasp_sampler.seed(seed)
        self.optimizer.seed(seed)

    def state_representation(self, obs, plot=True):

        intrinsics = np.array(self.config[0]['intrinsics']).reshape(3, 3)

        point_cloud_0 = PointCloud.from_depth(obs['depth'][0], intrinsics)
        point_cloud_1 = PointCloud.from_depth(obs['depth'][1], intrinsics)
        point_cloud_2 = PointCloud.from_depth(obs['depth'][2], intrinsics)
        point_cloud_3 = PointCloud.from_depth(obs['depth'][3], intrinsics)
        point_cloud_4 = PointCloud.from_depth(obs['depth'][4], intrinsics)
        point_cloud_5 = PointCloud.from_depth(obs['depth'][5], intrinsics)
        point_cloud_6 = PointCloud.from_depth(obs['depth'][6], intrinsics)
        point_cloud_7 = PointCloud.from_depth(obs['depth'][7], intrinsics)
        point_cloud_8 = PointCloud.from_depth(obs['depth'][8], intrinsics)
        point_cloud_9 = PointCloud.from_depth(obs['depth'][9], intrinsics)
        point_cloud_10 = PointCloud.from_depth(obs['depth'][10], intrinsics)
        point_cloud_11 = PointCloud.from_depth(obs['depth'][11], intrinsics)
        point_cloud_12 = PointCloud.from_depth(obs['depth'][12], intrinsics)
        point_cloud_13 = PointCloud.from_depth(obs['depth'][13], intrinsics)
        

        point_cloud_0.estimate_normals()
        point_cloud_1.estimate_normals()
        point_cloud_2.estimate_normals()
        point_cloud_3.estimate_normals()
        point_cloud_4.estimate_normals()
        point_cloud_5.estimate_normals()
        point_cloud_6.estimate_normals()
        point_cloud_7.estimate_normals()
        point_cloud_8.estimate_normals()
        point_cloud_9.estimate_normals()
        point_cloud_10.estimate_normals()
        point_cloud_11.estimate_normals()
        point_cloud_12.estimate_normals()
        point_cloud_13.estimate_normals()

        # Transform point cloud w.r.t. world frame
        transform_0 = pybullet_utils.get_camera_pose(self.config[0]['pos'],
                                                   self.config[0]['target_pos'],
                                                   self.config[0]['up_vector'])
        point_cloud_0 = point_cloud_0.transform(transform_0)

        transform_1 = pybullet_utils.get_camera_pose(self.config[1]['pos'],
                                                   self.config[1]['target_pos'],
                                                   self.config[1]['up_vector'])
        point_cloud_1 = point_cloud_1.transform(transform_1)

        transform_2 = pybullet_utils.get_camera_pose(self.config[2]['pos'],
                                                   self.config[2]['target_pos'],
                                                   self.config[2]['up_vector'])
        point_cloud_2 = point_cloud_2.transform(transform_2)

        transform_3 = pybullet_utils.get_camera_pose(self.config[3]['pos'],
                                                   self.config[3]['target_pos'],
                                                   self.config[3]['up_vector'])
        point_cloud_3 = point_cloud_3.transform(transform_3)

        transform_4 = pybullet_utils.get_camera_pose(self.config[4]['pos'],
                                                   self.config[4]['target_pos'],
                                                   self.config[4]['up_vector'])
        point_cloud_4 = point_cloud_4.transform(transform_4)
   
        transform_5 = pybullet_utils.get_camera_pose(self.config[5]['pos'],
                                                   self.config[5]['target_pos'],
                                                   self.config[5]['up_vector'])
        point_cloud_5 = point_cloud_5.transform(transform_5)

        transform_6 = pybullet_utils.get_camera_pose(self.config[6]['pos'],
                                                   self.config[6]['target_pos'],
                                                   self.config[6]['up_vector'])
        point_cloud_6 = point_cloud_6.transform(transform_6)

        transform_7 = pybullet_utils.get_camera_pose(self.config[7]['pos'],
                                                   self.config[7]['target_pos'],
                                                   self.config[7]['up_vector'])
        point_cloud_7 = point_cloud_7.transform(transform_7)

        transform_8 = pybullet_utils.get_camera_pose(self.config[8]['pos'],
                                                   self.config[8]['target_pos'],
                                                   self.config[8]['up_vector'])
        point_cloud_8 = point_cloud_8.transform(transform_8)

        transform_9 = pybullet_utils.get_camera_pose(self.config[9]['pos'],
                                                   self.config[9]['target_pos'],
                                                   self.config[9]['up_vector'])
        point_cloud_9 = point_cloud_9.transform(transform_9)

        transform_10 = pybullet_utils.get_camera_pose(self.config[10]['pos'],
                                                   self.config[10]['target_pos'],
                                                   self.config[10]['up_vector'])
        point_cloud_10 = point_cloud_10.transform(transform_10)

        transform_11 = pybullet_utils.get_camera_pose(self.config[11]['pos'],
                                                   self.config[11]['target_pos'],
                                                   self.config[11]['up_vector'])
        point_cloud_11 = point_cloud_11.transform(transform_11)

        transform_12 = pybullet_utils.get_camera_pose(self.config[12]['pos'],
                                                   self.config[12]['target_pos'],
                                                   self.config[12]['up_vector'])
        point_cloud_12 = point_cloud_12.transform(transform_12)

        transform_13 = pybullet_utils.get_camera_pose(self.config[13]['pos'],
                                                   self.config[13]['target_pos'],
                                                   self.config[13]['up_vector'])
        point_cloud_13 = point_cloud_13.transform(transform_13)

        # Keep only the points that lie inside the robot's workspace
        crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([self.bounds[0, 0],
                                                                           self.bounds[1, 0],
                                                                           self.bounds[2, 0]]),
                                                       max_bound=np.array([self.bounds[0, 1],
                                                                           self.bounds[1, 1],
                                                                           self.bounds[2, 1]]))
        point_cloud_0 = point_cloud_0.crop(crop_box)
        point_cloud_1 = point_cloud_1.crop(crop_box)
        point_cloud_2 = point_cloud_2.crop(crop_box)
        point_cloud_3 = point_cloud_3.crop(crop_box)
        point_cloud_4 = point_cloud_4.crop(crop_box)
        point_cloud_5 = point_cloud_5.crop(crop_box)
        point_cloud_6 = point_cloud_6.crop(crop_box)
        point_cloud_7 = point_cloud_7.crop(crop_box)
        point_cloud_8 = point_cloud_8.crop(crop_box)
        point_cloud_9 = point_cloud_9.crop(crop_box)
        point_cloud_10 = point_cloud_10.crop(crop_box)
        point_cloud_11 = point_cloud_11.crop(crop_box)
        point_cloud_12 = point_cloud_12.crop(crop_box)
        point_cloud_13 = point_cloud_13.crop(crop_box)

        # Orient normals to align with camera direction
        point_cloud_0.orient_normals_to_align_with_direction(-transform_0[0:3, 2])
        point_cloud_1.orient_normals_to_align_with_direction(-transform_1[0:3, 2])
        point_cloud_2.orient_normals_to_align_with_direction(-transform_2[0:3, 2])
        point_cloud_3.orient_normals_to_align_with_direction(-transform_3[0:3, 2])
        point_cloud_4.orient_normals_to_align_with_direction(-transform_4[0:3, 2])
        point_cloud_5.orient_normals_to_align_with_direction(-transform_5[0:3, 2])
        point_cloud_6.orient_normals_to_align_with_direction(-transform_6[0:3, 2])
        point_cloud_7.orient_normals_to_align_with_direction(-transform_7[0:3, 2])
        point_cloud_8.orient_normals_to_align_with_direction(-transform_8[0:3, 2])
        point_cloud_9.orient_normals_to_align_with_direction(-transform_9[0:3, 2])
        point_cloud_10.orient_normals_to_align_with_direction(-transform_10[0:3, 2])
        point_cloud_11.orient_normals_to_align_with_direction(-transform_11[0:3, 2])
        point_cloud_12.orient_normals_to_align_with_direction(-transform_12[0:3, 2])
        point_cloud_13.orient_normals_to_align_with_direction(-transform_13[0:3, 2])
        

        point_cloud = point_cloud_0 + point_cloud_1 + point_cloud_2 + point_cloud_3 + point_cloud_4 + point_cloud_5 + point_cloud_6 + point_cloud_7 + point_cloud_8 + point_cloud_9 + point_cloud_10 + point_cloud_11 + point_cloud_12 + point_cloud_13     

        # Save data
        cam_point_clouds = [point_cloud_0, point_cloud_1, point_cloud_2, point_cloud_3, point_cloud_4, point_cloud_5, point_cloud_6, point_cloud_7, point_cloud_8, point_cloud_9, point_cloud_10, point_cloud_11, point_cloud_12, point_cloud_13]
        cam_frames = [transform_0, transform_1, transform_2, transform_3, transform_4]

        point_cloud.paint_uniform_color([0.0, 0.5, 0.5])
        if plot:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            o3d.visualization.draw_geometries([point_cloud, mesh_frame])
        return point_cloud, cam_point_clouds, cam_frames

    def get_candidates(self, point_cloud, plot):
        grasp_candidates = self.grasp_sampler.sample(point_cloud, plot)
        return grasp_candidates

    def reconstruct(self, candidate, plot=False):
        def normalize(grid):
            diagonal = np.sqrt(2) * grid.shape[1]

            normalized_grid = grid.copy()
            normalized_grid[0] /= diagonal
            normalized_grid[1] = normalized_grid[1]
            normalized_grid[2:] = np.arccos(grid[2:]) / np.pi
            return normalized_grid
        points = np.asarray(candidate['enclosing_pts'].points)
        normals = np.asarray(candidate['enclosing_pts'].normals)

        voxel_grid = VoxelGrid(resolution=32)
        partial_grid = voxel_grid.voxelize(points, normals)
        partial_grid = normalize(partial_grid)

        x = torch.FloatTensor(partial_grid).to('cuda')
        x = x.unsqueeze(0)
        pred_sdf, pred_normals, _, _ = self.vae(x)

        pts, normals = get_reconstructed_surface(pred_sdf=pred_sdf.squeeze().detach().cpu().numpy(),
                                                 pred_normals=pred_normals.squeeze().detach().cpu().numpy(),
                                                 x=x.squeeze().detach().cpu().numpy(),
                                                 leaf_size=voxel_grid.leaf_size,
                                                 min_pt=voxel_grid.min_pt)

        rec_point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        rec_point_cloud.normals = o3d.utility.Vector3dVector(normals)

        if plot:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            rec_point_cloud.paint_uniform_color([0, 0, 1])
            o3d.visualization.draw_geometries([rec_point_cloud, mesh_frame])

        return rec_point_cloud

    def optimize(self, init_preshape, point_cloud, plot):
        opt_preshape = self.optimizer.optimize(init_preshape, point_cloud, plot)
        return opt_preshape