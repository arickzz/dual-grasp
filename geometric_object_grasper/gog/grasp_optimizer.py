import numpy as np
import random
import open3d as o3d
from sklearn.neighbors import KDTree
import time
from collections import defaultdict


# from gog.utils.orientation import Quaternion, angle_axis2rot
# from gog.robot_hand import RobotHand
from geometric_object_grasper.gog.utils.orientation import Quaternion, angle_axis2rot
from geometric_object_grasper.gog.robot_hand import RobotHand

def to_matrix(vec):
    pose = np.eye(4)
    pose[0:3, 0:3] = Quaternion().from_roll_pitch_yaw(vec[3:]).rotation_matrix()
    pose[0:3, 3] = vec[0:3]
    return pose

def count_non_zero(vector):
    count = 0
    for value in vector:
        if value != 0:
            count += 1
    return count

def top_five_shape_metric_points(hand, metric_vector):
    
    data = {'link': hand['links'],
            'points': hand['points'],
            'metric_vector': metric_vector,
            'normals': hand['normals'],
            'is_top_five': [0]*len(hand['points'])}

    links = defaultdict(list)

    for i in range(len(data['link'])):
        link = data['link'][i]
        point = data['points'][i]
        metric = data['metric_vector'][i]
        normals = data['normals'][i]
        links[link].append((point, normals, metric))

    final_data = []
    for key in links.keys():
        counter = 0
        for k in range(len(links['{}'.format(key)])):
            if links['{}'.format(key)][k][2] != 0:
                counter += 1

        if counter == 0:
            continue
        elif counter>0 and counter<=5:
            for k in range(len(links['{}'.format(key)])):
                if links['{}'.format(key)][k][2] != 0:
                    final_data.append(links['{}'.format(key)][k])
        else:
            sorted_points = sorted(links['{}'.format(key)], key=lambda x: x[2])
            # top_five_points = points[:5] for link, points in sorted_links.items()
            for j in range(5):
                final_data.append(sorted_points[j])

    return final_data

class QualityMetric:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class ShapeComplementarityMetric(QualityMetric):
    def __init__(self):
        self.w = 0.5
        self.a = 0.5

    # def compute_shape_complementarity(self, hand, target, radius=0.01):

    #     # Find nearest neighbors for each hand contact
    #     kdtree = KDTree(target['points'])
    #     dists, nn_ids = kdtree.query(hand['points'], return_distance=True)
    #     # if len(nn_ids) == 0:
    #     #     print("Numero de vizinhos = 0")

    #     metric_vector = [0]*len(hand['points'])
    #     shape_metric = 0.0

    #     for i in range(len(hand['points'])):

    #         # Remove duplicates or contacts with nn distance greater than a threshold
    #         duplicates = np.argwhere(nn_ids == nn_ids[i])[:, 0]
    #         if np.min(dists[duplicates]) != dists[i] or dists[i] > radius:
    #             metric_vector[i] = 0
    #             continue

    #         # Distance error
    #         e_p = np.linalg.norm(hand['points'][i] - target['points'][nn_ids[i][0]])

    #         # Alignment error
    #         c_n = -hand['normals'][i] # for TRO is -hand['normals']
    #         e_n = c_n.dot(target['normals'][nn_ids[i][0]])

    #         metric_vector[i] = e_p + self.w * e_n
    #         shape_metric += metric_vector[i]


    #     # contact_points = top_five_shape_metric_points(hand, metric_vector)
    #     # contact_normals = [0]*len(contact_points)

    #     # if len(contact_points) > 0:
    #     #     for k in range(len(contact_points)):
    #     #         aux = contact_points[k]
    #     #         shape_metric += contact_points[k][2]
    #     #         contact_points[k] = aux[0]
    #     #         contact_normals[k] = aux[1]

    #         # print("Shape Complementarity = {}".format(-shape_metric / len(hand['points']))) 
    #         # return -shape_metric / len(contact_points), contact_points, contact_normals
    #     return -shape_metric / len(hand['points'])
    
        # else:
            # print("# contact points = 0")
            # raise ValueError()

    def compute_shape_complementarity(self, hand, target, radius=0.01):
        # Find nearest neighbors for each hand contact
        kdtree = KDTree(target['points'])
        dists, nn_ids = kdtree.query(hand['points'], return_distance=True)
        
        ids_valid_hand_points = []
        ids_valid_object_points = []
        count = 0   # count points without correspondence in the object
        shape_metric = 0.0
        for i in range(hand['points'].shape[0]):
            
            # Remove duplicates or contacts with nn distance greater than a threshold
            duplicates = np.argwhere(nn_ids == nn_ids[i])[:, 0]
            if np.min(dists[duplicates]) != dists[i] or dists[i] > radius:
                count += 1
                continue
            
            ids_valid_hand_points.append(i)
            ids_valid_object_points.append(nn_ids[i])

            # Distance error
            e_p = np.linalg.norm(hand['points'][i] - target['points'][nn_ids[i][0]])

            # Alignment error
            c_n = -hand['normals'][i] # for TRO is -hand['normals']
            e_n = c_n.dot(target['normals'][nn_ids[i][0]])

            shape_metric += e_p + self.w * e_n
            # print("Shape Complementarity - Dist: {}\tNormals: {}\tMetric: {}".format(e_p, e_n, -shape_metric/len(hand['points'])))
        # print("Shape Complementarity = {}".format(-shape_metric / len(hand['points']))) 
        
        # print("Total: {}\tValid: {}\tNo correspondence: {}".format(len(hand['points']), len(hand['points'])-count, count))

        if count >= len(hand['points']) or shape_metric <= 0:
            return 0, len(hand['points'])-count, ids_valid_hand_points, ids_valid_object_points
        else:
            return shape_metric / len(hand['points']), len(hand['points'])-count, ids_valid_hand_points, ids_valid_object_points
            


    @staticmethod
    def compute_collision_penalty(hand, collisions):
        if collisions.shape[0] == 0:
            return 0.0

        # Find nearest neighbors for each collision
        kdtree = KDTree(hand['points'])
        dists, nn_ids = kdtree.query(collisions)

        e_col = 0.0
        for i in range(collisions.shape[0]):
            e_col += np.linalg.norm(collisions[i] - hand['points'][nn_ids[i][0]])
        
        # print("Collision penalty: {}".format(e_col))
        return -e_col
 
    def __call__(self, hand, target, collisions):
        return self.compute_shape_complementarity(hand, target) #, self.compute_shape_complementarity(hand, target)[0] + self.a * self.compute_collision_penalty(hand, collisions)


class Optimizer:
    def __init__(self, params, name):
        self.params = params
        self.name = name

    def optimize(self):
        raise NotImplementedError


class Swarm(list):
    def __init__(self, positions, limits=None):
        super(Swarm, self).__init__()
        for i in range(positions.shape[0]):
            particle = {'pos': positions[i],
                        'prev_best_pos': positions[i],
                        'vel': 0.0,
                        'score': 0.0,
                        'prev_best_score': 0.0}
            self.append(particle)

        self.limits = limits

        self.global_best_score = 1e+308
        self.global_best_pos = self[0]['pos']

    def update(self, w, c1, c2):
        for particle in self:

            r1 = random.uniform(0, 1)
            r2 = random.uniform(0, 1)
            particle['vel'] = w * particle['vel'] + c1 * r1 * (particle['prev_best_pos'] - particle['pos']) + \
                              c2 * r2 * (self.global_best_pos - particle['pos'])
            particle['pos'] += particle['vel']

            # Bound constraints are enforced by projecting to the limits
            if self.limits is not None:
                for i in range(particle['pos'].shape[0]):
                    particle['pos'][i] = max(particle['pos'][i], self.limits[i, 0])
                    particle['pos'][i] = min(particle['pos'][i], self.limits[i, 1])

    def update_global_best(self, particle):
        # Update each particle's optimal position
        if particle['score'] > particle['prev_best_score']:
            particle['prev_best_pos'] = particle['pos']
            particle['prev_best_score'] = particle['score']

        # Update swarm's best global position
        if particle['score'] > self.global_best_score:
            self.global_best_score = particle['score']
            self.global_best_pos = particle['pos'].copy()


class SplitPSO(Optimizer):
    def __init__(self, robot, params, name='split-pso'):
        super(SplitPSO, self).__init__(name, params)
        self.robot = robot
        self.params = params
        # parameters not needed in the optimizer
        aux1 = 0
        aux2 = []
        aux3 = []

        if self.params['metric'] == 'shape_complementarity':
            self.quality_metric = ShapeComplementarityMetric()


        self.rng = np.random.RandomState()

    def seed(self, seed):
        self.rng.seed(seed)

    def generate_random_angles(self, init_joints, joints_range=None):
        #barret hand
        # joints_range = np.array([[0.0, 10.0 * np.pi / 180],
        #                          [0.0, 50.0 * np.pi / 180],
        #                          [0.0, 50.0 * np.pi / 180],
        #                          [0.0, 50.0 * np.pi / 180]])
        #RH8D Hand
        joints_range = np.array([[-1.57, 1.57], 
                                 [0.0, 1.57],
                                 [0.0, 1.57],
                                 [0.0, 1.57],
                                 [0.0, 1.57],
                                 [0.0, 1.57]])

        nr_joints = len(init_joints)

        joints_particles = np.zeros((self.params['particles'] + 1, nr_joints))
        joints_particles[0] = init_joints

        for i in range(1, self.params['particles']):
            for j in range(nr_joints):
                joints_particles[i, j] = init_joints[j] + self.rng.uniform(joints_range[j, 0], joints_range[j, 1])

        return joints_particles

    def generate_random_poses(self, init_pose, t_range=[-0.02, 0.02], ea_range=[-20, 20]):

        pose_particles = np.zeros((self.params['particles'] + 1, 6))
        pose_particles[0, 0:3] = init_pose[0:3, 3]
        pose_particles[0, 3:] = Quaternion().from_rotation_matrix(init_pose[0:3, 0:3]).roll_pitch_yaw()

        for i in range(1, self.params['particles']):
            # Randomize position
            t = init_pose[0:3, 3] + self.rng.uniform(low=t_range[0], high=t_range[1], size=(3,))

            # Randomize orientation
            random_axis = np.random.rand(3, )
            random_axis /= np.linalg.norm(random_axis)
            random_rot = angle_axis2rot(np.random.uniform(ea_range[0], ea_range[1]) * np.pi / 180, random_axis)
            random_rot = np.matmul(random_rot, init_pose[0:3, 0:3])

            pose_particles[i, 0:3] = t
            pose_particles[i, 3:] = Quaternion().from_rotation_matrix(random_rot).roll_pitch_yaw()

        return pose_particles

    def create_swarm(self, init_preshape):
        # Generate two swarms, one by randomizing the pose and one by randomizing the joints
        pose_swarm = Swarm(self.generate_random_poses(init_preshape['pose']))

        # ToDo: hardcoded limits
        # Barret Hand
        # joints_swarm = Swarm(self.generate_random_angles(init_preshape['finger_joints']), 
        #                      limits=np.array([[0, 30 * np.pi / 180], [0, 140 * np.pi / 180],
        #                                       [0, 140 * np.pi / 180], [0, 140 * np.pi / 180]]))
        # RH8D Hand
        joints_swarm = Swarm(self.generate_random_angles(init_preshape['finger_joints']),
                             limits=np.array([[-1.57, 1.57], [0, 1.57],
                                              [0, 1.57], [0, 1.57], 
                                              [0, 1.57],[0, 1.57]]))

        return pose_swarm, joints_swarm

    def plot_grasp(self, point_cloud, preshape, contact_pts=[], contact_normals=[]):
        self.robot.update_links(joint_values=preshape['joints'],
                                palm_pose=preshape['pose'])

        hand_contacts = self.robot.get_contacts()
        object_pts = {'points': np.asarray(point_cloud.points),
                      'normals': np.asarray(point_cloud.normals)}
        
        contact_pts_vis = o3d.geometry.PointCloud()
        contact_pts_vis.points = o3d.utility.Vector3dVector(contact_pts)
        contact_pts_vis.normals = o3d.utility.Vector3dVector(contact_normals)
        contact_pts_vis.paint_uniform_color([0.0, 0.5, 0.0])


        visuals = self.robot.get_visual_map(meshes=True, boxes=False, frames=False)
        visuals.append(point_cloud)
        visuals.append(contact_pts_vis)
        
        o3d.visualization.draw_geometries(visuals)
        # o3d.visualization.draw_geometries([point_cloud, contact_pts_vis])


    def optimize(self, init_preshape, point_cloud, plot):
        if plot:
           self.plot_grasp(point_cloud, preshape={'joints': init_preshape['finger_joints'],
                                                  'pose': init_preshape['pose']})

        finger_joints = init_preshape['finger_joints']
        opt_pose = init_preshape['pose']

        # Generate swarms
        # pose_swarm, joints_swarm = self.create_swarm(init_preshape)

        object_pts = {'points': np.asarray(point_cloud.points),
                      'normals': np.asarray(point_cloud.normals)}
        

        # for i in range(self.params['max_iterations']):
        # #for i in range(1):
        #     # Optimize palm pose
        #     for particle in pose_swarm:
        #         # Update robot hand links
        #         pose = to_matrix(particle['pos'])
        #         self.robot.update_links(joint_values=joints_swarm.global_best_pos, palm_pose=pose)

        #         # Estimate each particle's fitness value(quality metric)
        #         particle['score'] = self.quality_metric(hand=self.robot.get_contacts(), target=object_pts,collisions=self.robot.get_collision_pts(point_cloud))[1]

        #         # Update personal best and global best
        #         pose_swarm.update_global_best(particle)

        #     # Update velocity and position
        #     pose_swarm.update(w=self.params['w'], c1=self.params['c1'], c2=self.params['c2'])
            

        #     # Optimize joints
        #     for particle in joints_swarm:
        #         # Update robot hand links
        #         self.robot.update_links(joint_values=particle['pos'],
        #                                 palm_pose=to_matrix(pose_swarm.global_best_pos))

        #         # Estimate each particle's fitness value(quality metric)
        #         particle['score'] = self.quality_metric(hand=self.robot.get_contacts(),
        #                                                 target=object_pts,
        #                                                 collisions=self.robot.get_collision_pts(point_cloud))[1]

        #         joints_swarm.update_global_best(particle)

        #     # Update velocity and position
        #     joints_swarm.update(w=self.params['w'], c1=self.params['c1'], c2=self.params['c2'])

            
            # print('\r' + str(joints_swarm.global_best_score), end='', flush=True)
        # print('Joints global best: {}'.format(i, joints_swarm.global_best_score))
        # print('Pose global best {}'.format(i, pose_swarm.global_best_score))

        # self.plot_grasp(point_cloud, preshape={'joints': joints_swarm.global_best_pos,
        #                                         'pose': to_matrix(pose_swarm.global_best_pos)})
        
        # opt_pose = to_matrix(pose_swarm.global_best_pos)
        # finger_joints = joints_swarm.global_best_pos


        # BARRET Hand
        # max_bounds = [1.8, 1.8, 1.8]
        # for j in range(3): 
        #     while True:
        #         finger_joints[j+1] += 0.1 # close the link
        #         if finger_joints[j+1] > max_bounds[j]:
        #             break
        #         self.robot.update_links(joint_values=finger_joints, palm_pose=opt_pose)
            
        #         if len(self.robot.get_collision_pts(point_cloud)) > 0: # check for colisions
        #             finger_joints[j + 1] -= 0.1
        #             break

        # RH8D Hand
        links = [['Index_Proximal', 'Index_Middle', 'Index_Distal'],
                 ['Small_Proximal', 'Small_Middle', 'Small_Distal'],
                 ['Middle_Proximal', 'Middle_Middle', 'Middle_Distal'],
                 ['Ring_Proximal', 'Ring_Middle', 'Ring_Distal'],
                 ['Thumb_Proximal', 'Thumb_Methacarpal', 'Thumb_Distal']]
        joint_contacts = []
        collision_pts = 0
        collision_pts_new = 0
        max_bounds = [1.57, 1.57, 1.57, 1.57, 1.57]
        collision_pts_finger = [0, 0, 0, 0, 0]
        for j in range(5):
            while True:
                finger_joints[j+1] += 0.05 # close the joint

                
                if finger_joints[j+1] > max_bounds[j]:
                    finger_joints[j+1] = 0.05
                    break
                self.robot.update_links(joint_values=finger_joints, palm_pose=opt_pose)

                collision_pts_new = len(self.robot.get_collision_pts(point_cloud))
            
                if collision_pts_new > collision_pts:
                    #finger_joints[j + 1] -= 0.05
                    collision_pts_finger[j] = collision_pts_new-collision_pts
                    collision_pts = collision_pts_new
                    break
                
        # Save collision points and respective normals
        # contact_pts_link = np.zeros((2, len(self.robot.links)))
        collision_pts, contact_pts_link = self.robot.get_collision_pts_links(point_cloud)
        collision_normals = self.robot.get_collision_normals(point_cloud)

        # preshape={'joints': finger_joints, 'pose': opt_pose}
        # self.plot_grasp(point_cloud, preshape, collision_pts)

        for j in range(5):
            finger_joints[j + 1] -= 0.05

        grasp_center = self.robot.update_links(joint_values=finger_joints, palm_pose=opt_pose)
        # score = self.quality_metric(hand=self.robot.get_contacts(), target=object_pts,
        #                             collisions=self.robot.get_collision_pts(point_cloud))
        
        hand_contacts = self.robot.get_contacts(cts_link=True)
        
        if plot:
            self.plot_grasp(point_cloud, preshape={'joints': finger_joints,'pose': opt_pose}, contact_pts=collision_pts, contact_normals=collision_normals)
        
        return {# 'finger_joints': joints_swarm.global_best_pos,
                # 'pose': to_matrix(pose_swarm.global_best_pos),
                'finger_joints': finger_joints,
                'pose': opt_pose,
                #'score': score,
                'collision_pts': collision_pts,
                'collision_normals': collision_normals,
                'hand_contacts': hand_contacts,
                'grasp_center': grasp_center,
                'contact_pts_link': contact_pts_link}


class PSO(Optimizer):
    def __init__(self, robot, params, name='pso'):
        super(PSO, self).__init__(name, params)
        self.robot = robot
        self.params = params

    def optimize(self, init_preshape, point_cloud):
        pass