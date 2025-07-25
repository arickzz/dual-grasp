import open3d as o3d
import numpy as np

from gog.utils.orientation import angle_axis2rot
from gog.robot_hand import CShapeHand
import time

class FrameEstimator:
    def __init__(self, pts, normals):
        self.pts = np.asarray(pts)
        self.normals = np.asarray(normals)

    def estimate(self, pt):
        """
        Estimates a local frame placed on point pt

        Parameters
        ----------
        pt: np.array()
            The input point (1x3 matrix)

        Returns
        -------
        np.array() :
            4x4 matrix
        """
        curvature_axis, normal, binormal = self.find_principal_curvature()
        frame = np.eye(4)
        frame[0:3, 0] = binormal
        frame[0:3, 1] = curvature_axis
        frame[0:3, 2] = normal
        frame[0:3, 3] = pt
        return frame

    def find_principal_curvature(self):
        """
        Estimate the principal curvature
        """
        # 1. Calculate "curvature axis" and "normal axis" (corresponds to min and max principal curvature axis).
        correlation_matrix = np.matmul(self.normals.transpose(), self.normals)
        eigen_values, eigen_vectors = np.linalg.eig(correlation_matrix)
        min_id = np.argmin(eigen_values)
        curvature_axis = eigen_vectors[:, min_id]

        max_id = np.argmax(eigen_values)
        normal = eigen_vectors[:, max_id]

        # Ensure that the estimated normal is pointing in the same direction as the existing normals.
        avg_normal = np.sum(self.normals, 0)
        avg_normal /= np.linalg.norm(avg_normal)
        if np.dot(avg_normal, normal) < 0:
            normal *= -1.0

        binormal = np.cross(curvature_axis, normal)
        return curvature_axis, normal, binormal


class GraspSampler:
    def __init__(self, robot, params):
        self.params = params
        self.num_samples = params['samples']
        self.rotations = params['rotations']
        self.robot = robot

        self.c_shapes = []
        self.bhand_angles = params['c_shape_hand']['bhand_angles']
        nr_of_configurations = len(params['c_shape_hand']['inner_radius'])
        for i in range(nr_of_configurations):
            self.c_shapes.append(CShapeHand(inner_radius=params['c_shape_hand']['inner_radius'][i],
                                            outer_radius=params['c_shape_hand']['outer_radius'][i],
                                            gripper_width=params['c_shape_hand']['gripper_width'],
                                            phi=params['c_shape_hand']['phi'][i]))

        self.rng = np.random.RandomState()

    def seed(self, seed):
        self.rng.seed(seed)

    def sample(self, point_cloud, plot):                                       
        """
        Generate collision-free grasp candidates by approximating the
        robot hand with a c-shaped cylinder.

        Parameters
        ----------
        point_cloud: open3d.geometry.PointCloud
            The input point cloud (transformed w.r.t. the table).
        plot: boolean
            If True plots the generated grasp candidates.

        Returns
        -------
        A list of the grasp candidates. Each element of the list is a
        dictionary of {'pose', 'finger_joints', 'enclosing_pts}.
        """

        # Sample only from the points on the left/right side of the robot
        y = np.asarray(point_cloud.points)[:, 1]
        sample_ids_y = np.where((y > 0.0))[0]    # left hand, y<0; right hand, y>0
        above_side_pts = point_cloud.select_by_index(sample_ids_y)

        # Downsample point cloud (no need for dense point cloud)
        col_point_cloud = above_side_pts.voxel_down_sample(voxel_size=0.005)

        # Create a kd-tree for nearest neighbor search
        pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)

        rotation_angle = np.pi / self.rotations

        num_iterations = 0
        grasp_candidates = []
        candidates_rand_points = []
        while len(grasp_candidates) < 1 and num_iterations < 500:  #self.num_samples:

            num_iterations += 1

            # Sample a point randomly from input point cloud
            random_id = self.rng.choice(sample_ids_y)

            # Find nearest neighbors
            [_, idx, _] = pcd_tree.search_hybrid_vector_3d(query=point_cloud.points[random_id],
                                                           radius=0.02, max_nn=1000)
            nn_pcd = point_cloud.select_by_index(indices=idx)

            # Estimate a local frame
            frame_estimator = FrameEstimator(pts=nn_pcd.points, normals=nn_pcd.normals)
            local_frame = frame_estimator.estimate(pt=point_cloud.points[random_id])

            for i in range(self.rotations):
                # Rotate around the z axis of the local frame
                rot_around_z = angle_axis2rot(angle=i*rotation_angle, axis=local_frame[0:3, 2])
                candidate = local_frame.copy()

                candidate[0:3, 0:3] = np.matmul(rot_around_z, local_frame[0:3, 0:3])
            
                # Deslocar a c-shape no sentido contrário ao da normal ao ponto aleatório
                # candidate[3][0] -= point_cloud.normals[random_id][0]
                # candidate[3][1] -= point_cloud.normals[random_id][1]
                # candidate[3][2] -= point_cloud.normals[random_id][2]

                # Rotate 180 degrees around x axis
                hand_frame = candidate.copy()
                hand_frame[0:3, 1] = -candidate[0:3, 1]
                hand_frame[0:3, 2] = -candidate[0:3, 2]


                # Check for collision with robot hand
                for j in range(len(self.c_shapes)):
                    c_shape = self.c_shapes[j]


                    if c_shape.is_collision_free(grasp_candidate=hand_frame,
                                                pts=np.asarray(col_point_cloud.points)):

                        hand_frame = c_shape.push_forward(hand_frame, np.asarray(point_cloud.points))
                        enclosing_ids = c_shape.get_points_in_closing_region(grasp_candidate=hand_frame,
                                                                            pts=np.asarray(col_point_cloud.points))
                        
                        if len(enclosing_ids) < 200:
                           continue

                        enclosing_pts = col_point_cloud.select_by_index(enclosing_ids) #.transform(np.linalg.inv(hand_frame))

                        frame = np.matmul(hand_frame, self.robot.ref_frame)
                        #Barret Hand
                        # grasp_candidates.append({'pose': frame,
                        #                          'finger_joints': np.array([0.0, self.bhand_angles[j],
                        #                                                     self.bhand_angles[j],
                        #                                                     self.bhand_angles[j]]),
                        #                          'enclosing_pts': enclosing_pts})
                        #RH8D hand
                        grasp_candidates.append({'pose': hand_frame,
                                                 'local_frame': local_frame,
                                                 'finger_joints': np.array([0.0, self.bhand_angles[j],
                                                                            self.bhand_angles[j],
                                                                            self.bhand_angles[j], 
                                                                            self.bhand_angles[j],
                                                                            self.bhand_angles[j]]),
                                                 'enclosing_pts': enclosing_pts})
                    
                        print('Grasp added. {} enclosing points.'.format(len(enclosing_pts.points)))
                        
                        if plot:
                            
                            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
                        
                            #Barret Hand
                            # self.robot.update_links(joint_values=[0.0, self.bhand_angles[j],
                            #                                       self.bhand_angles[j],
                            #                                       self.bhand_angles[j]],
                            #                         palm_pose=hand_frame)

                            enclosing_pts_vis = o3d.geometry.PointCloud()
                            enclosing_pts_vis.points = o3d.utility.Vector3dVector(enclosing_pts.points)
                            enclosing_pts_vis.paint_uniform_color([0, 1, 0.0])

                            #RH8D hand
                            self.robot.update_links(joint_values=[0.0,  self.bhand_angles[j],
                                                                        self.bhand_angles[j],
                                                                        self.bhand_angles[j], 
                                                                        self.bhand_angles[j],
                                                                        self.bhand_angles[j]],
                                                    palm_pose=hand_frame)
                            visuals = []
                            visuals = self.robot.get_visual_map(boxes=False, frames=False)
                            visuals.append(point_cloud)
                            visuals.append(c_shape.sample_surface().transform(hand_frame))

                            visuals.append(enclosing_pts_vis)


                            # visuals.append(col_point_cloud)
                                                
                            o3d.visualization.draw_geometries(visuals)

                        num_iterations = 0

                        break
            
                        

        return grasp_candidates



# class GraspSampler:
#     def __init__(self, robot, params):
#         self.num_samples = params['samples']
#         self.rotations = params['rotations']
#         self.robot = robot
#
#         self.c_shapes = []
#         self.bhand_angles = params['c_shape_hand']['bhand_angles']
#         nr_of_configurations = len(params['c_shape_hand']['inner_radius'])
#         for i in range(nr_of_configurations):
#             self.c_shapes.append(CShapeHand(inner_radius=params['c_shape_hand']['inner_radius'][i],
#                                             outer_radius=params['c_shape_hand']['outer_radius'][i],
#                                             gripper_width=params['c_shape_hand']['gripper_width'],
#                                             phi=params['c_shape_hand']['phi'][i]))
#
#         self.rng = np.random.RandomState()
#
#     def seed(self, seed):
#         self.rng.seed(seed)
#
#     def sample(self, point_cloud, full_point_cloud=None, plot=False):
#         """
#         Generate collision-free grasp candidates
#
#         Parameters
#         ----------
#         point_cloud: open3d.geometry.PointCloud
#             The input point cloud (transformed w.r.t. the table).
#         plot: boolean
#             If True plots the generated grasp candidates.
#
#         Returns
#         -------
#         A list of the grasp candidates. Each element of the list is a dictionary of {'pose', 'joint_values'}.
#         """
#
#         # Sample only from the points above the table # Todo: Do it outside the sampling operation
#         z = np.asarray(point_cloud.points)[:, 2]
#         ids = np.where(z > 0.01)[0]
#         point_cloud.select_by_index(ids)
#         self.rng.shuffle(ids)
#
#         # ids = np.arange(0, len(np.asarray(point_cloud.points)))
#         # ids = ids[::4]
#
#         # Create a kd-tree for nearest neighbor search
#         pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)
#
#         rotation_angle = np.pi / self.rotations
#
#         grasp_candidates = []
#         while len(grasp_candidates) < self.num_samples:
#             print('\r' + str(len(grasp_candidates)), end='', flush=True)
#
#             # Sample a point randomly from input point cloud
#             random_id = self.rng.choice(ids)
#
#             # Find nearest neighbors
#             [_, idx, _] = pcd_tree.search_hybrid_vector_3d(query=point_cloud.points[random_id],
#                                                            radius=0.02, max_nn=1000)
#             nn_pcd = point_cloud.select_by_index(indices=idx)
#
#             # Estimate a local frame
#             frame_estimator = FrameEstimator(pts=nn_pcd.points, normals=nn_pcd.normals)
#             local_frame = frame_estimator.estimate(pt=point_cloud.points[random_id])
#
#             for i in range(self.rotations):
#                 # Rotate around the z axis of the local frame
#                 rot_around_z = angle_axis2rot(angle=i*rotation_angle, axis=local_frame[0:3, 2])
#                 candidate = local_frame.copy()
#                 candidate[0:3, 0:3] = np.matmul(rot_around_z, local_frame[0:3, 0:3])
#
#                 # Rotate 180 degrees around x axis
#                 hand_frame = candidate.copy()
#                 hand_frame[0:3, 1] = -candidate[0:3, 1]
#                 hand_frame[0:3, 2] = -candidate[0:3, 2]
#
#                 # Check for collision with robot hand
#                 for j in range(len(self.c_shapes)):
#                     c_shape = self.c_shapes[j]
#                     if c_shape.is_collision_free(grasp_candidate=hand_frame,
#                                                  pts=np.asarray(point_cloud.points)):
#
#                         hand_frame = c_shape.push_forward(hand_frame, np.asarray(point_cloud.points))
#                         enclosing_ids = c_shape.get_points_in_closing_region(grasp_candidate=hand_frame,
#                                                                              pts=np.asarray(point_cloud.points))
#
#                         if full_point_cloud is None:
#                             frame = np.matmul(hand_frame, self.robot.ref_frame)
#                             grasp_candidates.append({'pose': frame,
#                                                      'joint_vals': np.array([0.0, self.bhand_angles[j],
#                                                                              self.bhand_angles[j],
#                                                                              self.bhand_angles[j]]),
#                                                      'enclosing_pts': point_cloud.select_by_index(enclosing_ids).transform(np.linalg.inv(hand_frame))})
#
#                         else:
#                             enclosing_full_ids = c_shape.get_points_in_closing_region(grasp_candidate=hand_frame,
#                                                                                       pts=np.asarray(full_point_cloud.points))
#                             frame = np.matmul(hand_frame, self.robot.ref_frame)
#                             grasp_candidates.append({'pose': frame,
#                                                      'joint_vals': np.array([0.0, self.bhand_angles[j],
#                                                                              self.bhand_angles[j],
#                                                                              self.bhand_angles[j]]),
#                                                      'enclosing_pts': [point_cloud.select_by_index(enclosing_ids).transform(np.linalg.inv(hand_frame)),
#                                                                        full_point_cloud.select_by_index(enclosing_full_ids).transform(np.linalg.inv(hand_frame))]})
#
#                         if plot:
#                             mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
#                             mesh_frame = mesh_frame.transform(frame)
#
#                             self.robot.update_links(joint_values=[0.0, self.bhand_angles[j],
#                                                                   self.bhand_angles[j],
#                                                                   self.bhand_angles[j]],
#                                                     palm_pose=hand_frame)
#                             visuals = self.robot.get_visual_map(boxes=True)
#                             visuals.append(point_cloud)
#                             visuals.append(c_shape.sample_surface().transform(hand_frame))
#                             visuals.append(mesh_frame)
#                             o3d.visualization.draw_geometries(visuals)
#
#                         break
#
#         return grasp_candidates
