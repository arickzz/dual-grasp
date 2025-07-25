import open3d as o3d
import argparse


def test_c_shape():
    import open3d as o3d
    import numpy as np
    from gog.robot_hand import BarrettHand, CShapeHand, RH8DRHand, RH8DLHand

    #Barret Hand
    # default_params = {
    #     'gripper_width': 0.07,
    #     'inner_radius': [0.05, 0.06, 0.07, 0.08],
    #     'outer_radius': [0.09, 0.1, 0.11, 0.12],
    #     'phi': [np.pi / 3, np.pi / 4, np.pi / 6, 0],
    #     'bhand_angles': [1.2, 1, 0.8, 0.65]
    # }

    # RH8DR
    default_params = {
        'gripper_width': 0.09,
        'inner_radius': [0.025, 0.035, 0.038, 0.05],
        'outer_radius': [0.067, 0.077, 0.084, 0.095],
        'phi': [0, 0, 0, 0],
        'bhand_angles': [0.4, 0.245, 0.160, 0]
    }

    #robot_hand = BarrettHand(urdf_file='../assets/robot_hands/barrett/bh_282.urdf')
    # robot_hand = RH8DRHand(urdf_file='../assets/robot_hands/RH8DR/RH8DR.urdf', contacts_file_name='contacts_right_hand_10')
    robot_hand = RH8DLHand(urdf_file='../assets/robot_hands/RH8DL/RH8DL.urdf', contacts_file_name='contacts_left_hand')

    for i in range(4):
        init_angle = default_params['bhand_angles'][i]
        #RH8D
        grasp_center = robot_hand.update_links(joint_values=[0.0, init_angle, init_angle, init_angle, init_angle, init_angle])
        # print(grasp_center)
        #BarretHand
        #robot_hand.update_links(joint_values=[0.0, init_angle, init_angle, init_angle])
        
        visuals = robot_hand.get_visual_map()
        
        c_shape_hand = CShapeHand(inner_radius=default_params['inner_radius'][i],
                                  outer_radius=default_params['outer_radius'][i],
                                  gripper_width=default_params['gripper_width'],
                                  phi=default_params['phi'][i])
        point_cloud = c_shape_hand.sample_surface()
        # visuals.append(point_cloud)

        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
        # visuals.append(frame)
        o3d.visualization.draw_geometries(visuals)


# def write_grasps_file_binary(grasps_dict, grasp_order):

#     grasps_file = open('/home/vislab/DA2/Grasps_Register/binary/{}.bin'.format(grasp_order), 'wb') ## Binary

#     # begin diccionary 
#     grasps_file.write(b"{")

#     # pose

#     pose_binary = bytes(f'"pose": [[{grasps_dict['pose'][0][0]}, {grasps_dict['pose'][0][1]}, {grasps_dict['pose'][0][2]}, {grasps_dict['pose'][0][3]}], 
#                                    [{grasps_dict['pose'][1][0]}, {grasps_dict['pose'][1][1]}, {grasps_dict['pose'][1][2]}, {grasps_dict['pose'][1][3]}], 
#                                    [{grasps_dict['pose'][2][0]}, {grasps_dict['pose'][2][1]}, {grasps_dict['pose'][2][2]}, {grasps_dict['pose'][2][3]}], 
#                                    [{grasps_dict['pose'][3][0]}, {grasps_dict['pose'][3][1]}, {grasps_dict['pose'][3][2]}, {grasps_dict['pose'][3][3]}]]', 'utf-8')
#     grasps_file.write(pose_binary)

#     # finger joints
#     grasps_file.write('"finger_joints": [{}, {}, {}, {}, {}, {}],'.format(grasps_dict['finger_joints'][0],
#                                                                             grasps_dict['finger_joints'][1],
#                                                                             grasps_dict['finger_joints'][2],
#                                                                             grasps_dict['finger_joints'][3],
#                                                                             grasps_dict['finger_joints'][4],
#                                                                             grasps_dict['finger_joints'][5]))

#     # score
#     grasps_file.write('"score": {},'.format(grasps_dict['score']))

#     # enclosing points
#     grasps_file.write('"enclosing_points": [')

#     for j in range(len(grasps_dict['enclosing_pts'])):
#         grasps_file.write("[{}, {}, {}]".format(grasps_dict['enclosing_pts'][j][0],grasps_dict['enclosing_pts'][j][1],grasps_dict['enclosing_pts'][j][2]))

#         if j < len(grasps_dict['enclosing_pts'])-1:
#             grasps_file.write(", ")

#     grasps_file.write("],")


#     # contact points
#     grasps_file.write('"contact_points": [')

#     for j in range(len(grasps_dict['collision_pts'])):
#         grasps_file.write("[{}, {}, {}]".format(grasps_dict['collision_pts'][j][0],grasps_dict['collision_pts'][j][1],grasps_dict['collision_pts'][j][2]))

#         if j < len(grasps_dict['collision_pts'])-1:
#             grasps_file.write(", ")

#     grasps_file.write("],")

#     # contact normals
#     grasps_file.write('"contact_normals": [')

#     for j in range(len(grasps_dict['collision_normals'])):
#         grasps_file.write("[{}, {}, {}]".format(grasps_dict['collision_normals'][j][0],grasps_dict['collision_normals'][j][1],grasps_dict['collision_normals'][j][2]))

#         if j < len(grasps_dict['collision_normals'])-1:
#             grasps_file.write(", ")

#     grasps_file.write("]}")

def write_grasps_file(grasps_dict, grasp_order, obj_name):

    grasps_file = open('/home/vislab/DA2/Grasps_Register/R/{}/{}.txt'.format(obj_name, grasp_order), 'w') ## Binary

    # begin diccionary 
    grasps_file.write("{")

    # pose
    grasps_file.write('"pose": [[{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]],'.format(grasps_dict['pose'][0][0],grasps_dict['pose'][0][1],grasps_dict['pose'][0][2],grasps_dict['pose'][0][3],
                                                                                                                grasps_dict['pose'][1][0],grasps_dict['pose'][1][1],grasps_dict['pose'][1][2],grasps_dict['pose'][1][3],
                                                                                                                grasps_dict['pose'][2][0],grasps_dict['pose'][2][1],grasps_dict['pose'][2][2],grasps_dict['pose'][2][3],
                                                                                                                grasps_dict['pose'][3][0],grasps_dict['pose'][3][1],grasps_dict['pose'][3][2],grasps_dict['pose'][3][3]))

    # cam frame
    # grasps_file.write('"cam_frame": [[{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]],'.format(grasps_dict['cam_frame'][0][0],grasps_dict['cam_frame'][0][1],grasps_dict['cam_frame'][0][2],grasps_dict['cam_frame'][0][3],
    #                                                                                                                     grasps_dict['cam_frame'][1][0],grasps_dict['cam_frame'][1][1],grasps_dict['cam_frame'][1][2],grasps_dict['cam_frame'][1][3],
    #                                                                                                                     grasps_dict['cam_frame'][2][0],grasps_dict['cam_frame'][2][1],grasps_dict['cam_frame'][2][2],grasps_dict['cam_frame'][2][3],
    #                                                                                                                     grasps_dict['cam_frame'][3][0],grasps_dict['cam_frame'][3][1],grasps_dict['cam_frame'][3][2],grasps_dict['cam_frame'][3][3]))
    

    # local frame
    # grasps_file.write('"local_frame": [[{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]],'.format(grasps_dict['local_frame'][0][0],grasps_dict['local_frame'][0][1],grasps_dict['local_frame'][0][2],grasps_dict['local_frame'][0][3],
    #                                                                                                                     grasps_dict['local_frame'][1][0],grasps_dict['local_frame'][1][1],grasps_dict['local_frame'][1][2],grasps_dict['local_frame'][1][3],
    #                                                                                                                     grasps_dict['local_frame'][2][0],grasps_dict['local_frame'][2][1],grasps_dict['local_frame'][2][2],grasps_dict['local_frame'][2][3],
    #                                                                                                                     grasps_dict['local_frame'][3][0],grasps_dict['local_frame'][3][1],grasps_dict['local_frame'][3][2],grasps_dict['local_frame'][3][3]))
    
    # pre pose
    grasps_file.write('"pre_pose": [[{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]],'.format(grasps_dict['pre_pose'][0][0],grasps_dict['pre_pose'][0][1],grasps_dict['pre_pose'][0][2],grasps_dict['pre_pose'][0][3],
                                                                                                                grasps_dict['pre_pose'][1][0],grasps_dict['pre_pose'][1][1],grasps_dict['pre_pose'][1][2],grasps_dict['pre_pose'][1][3],
                                                                                                                grasps_dict['pre_pose'][2][0],grasps_dict['pre_pose'][2][1],grasps_dict['pre_pose'][2][2],grasps_dict['pre_pose'][2][3],
                                                                                                                grasps_dict['pre_pose'][3][0],grasps_dict['pre_pose'][3][1],grasps_dict['pre_pose'][3][2],grasps_dict['pre_pose'][3][3]))


    # finger joints
    grasps_file.write('"finger_joints": [{}, {}, {}, {}, {}, {}],'.format(grasps_dict['finger_joints'][0],
                                                                            grasps_dict['finger_joints'][1],
                                                                            grasps_dict['finger_joints'][2],
                                                                            grasps_dict['finger_joints'][3],
                                                                            grasps_dict['finger_joints'][4],
                                                                            grasps_dict['finger_joints'][5]))
    
    # pre finger joints
    grasps_file.write('"pre_finger_joints": [{}, {}, {}, {}, {}, {}],'.format(grasps_dict['pre_finger_joints'][0],
                                                                            grasps_dict['pre_finger_joints'][1],
                                                                            grasps_dict['pre_finger_joints'][2],
                                                                            grasps_dict['pre_finger_joints'][3],
                                                                            grasps_dict['pre_finger_joints'][4],
                                                                            grasps_dict['pre_finger_joints'][5]))

    # grasp center pos
    grasps_file.write('"grasp_center": [{}, {}, {}],'.format(grasps_dict['grasp_center'][0],
                                                             grasps_dict['grasp_center'][1],
                                                             grasps_dict['grasp_center'][2]))

    # score
    # grasps_file.write('"score": {},'.format(grasps_dict['score']))

    # contact points by link R
    grasps_file.write('"contacts_finger": [{}, {}, {}, {}, {}, {}],'.format(grasps_dict['contact_pts_link'][0][1],  # palm
                                                                           grasps_dict['contact_pts_link'][10][1]+grasps_dict['contact_pts_link'][11][1]+grasps_dict['contact_pts_link'][12][1]+grasps_dict['contact_pts_link'][13][1], # thumb
                                                                           grasps_dict['contact_pts_link'][1][1]+grasps_dict['contact_pts_link'][2][1]+grasps_dict['contact_pts_link'][3][1], # index
                                                                           grasps_dict['contact_pts_link'][14][1]+grasps_dict['contact_pts_link'][15][1]+grasps_dict['contact_pts_link'][16][1], # middle
                                                                           grasps_dict['contact_pts_link'][7][1]+grasps_dict['contact_pts_link'][8][1]+grasps_dict['contact_pts_link'][9][1], # ring
                                                                           grasps_dict['contact_pts_link'][4][1]+grasps_dict['contact_pts_link'][5][1]+grasps_dict['contact_pts_link'][6][1])) # small

    # contact points by link L
    # grasps_file.write('"contacts_finger": [{}, {}, {}, {}, {}, {}],'.format(grasps_dict['contact_pts_link'][0][1],  # palm
    #                                                                        grasps_dict['contact_pts_link'][4][1]+grasps_dict['contact_pts_link'][5][1]+grasps_dict['contact_pts_link'][6][1]+grasps_dict['contact_pts_link'][7][1], # thumb
    #                                                                        grasps_dict['contact_pts_link'][1][1]+grasps_dict['contact_pts_link'][2][1]+grasps_dict['contact_pts_link'][3][1], # index
    #                                                                        grasps_dict['contact_pts_link'][11][1]+grasps_dict['contact_pts_link'][12][1]+grasps_dict['contact_pts_link'][13][1], # middle
    #                                                                        grasps_dict['contact_pts_link'][14][1]+grasps_dict['contact_pts_link'][15][1]+grasps_dict['contact_pts_link'][16][1], # ring
    #                                                                        grasps_dict['contact_pts_link'][8][1]+grasps_dict['contact_pts_link'][9][1]+grasps_dict['contact_pts_link'][10][1])) # small 


    # enclosing points
    grasps_file.write('"enclosing_points": [')

    for j in range(len(grasps_dict['enclosing_pts'])):
        grasps_file.write("[{}, {}, {}]".format(grasps_dict['enclosing_pts'][j][0],grasps_dict['enclosing_pts'][j][1],grasps_dict['enclosing_pts'][j][2]))

        if j < len(grasps_dict['enclosing_pts'])-1:
            grasps_file.write(", ")

    grasps_file.write("],")


    # hand contact points
    grasps_file.write('"hand_contacts": {')
    counter = 0

    for key in grasps_dict['hand_contacts'].keys():
        grasps_file.write('"{}": ['.format(key))
        for j in range(len(grasps_dict['hand_contacts']["{}".format(key)])):  
                grasps_file.write("[{}, {}, {}]".format(grasps_dict['hand_contacts']["{}".format(key)][j][0],grasps_dict['hand_contacts']["{}".format(key)][j][1],grasps_dict['hand_contacts']["{}".format(key)][j][2]))

                if j < len(grasps_dict['hand_contacts']["{}".format(key)])-1:
                    grasps_file.write(", ")

        grasps_file.write("]")
        if counter < len(grasps_dict['hand_contacts'].keys())-1:
                    grasps_file.write(", ")
        counter += 1

    grasps_file.write("},")

    # hand contact normals
    grasps_file.write('"hand_normals": {')
    counter = 0

    for key in grasps_dict['hand_normals'].keys():
        grasps_file.write('"{}": ['.format(key))
        for j in range(len(grasps_dict['hand_normals']["{}".format(key)])):  
                grasps_file.write("[{}, {}, {}]".format(grasps_dict['hand_normals']["{}".format(key)][j][0],grasps_dict['hand_normals']["{}".format(key)][j][1],grasps_dict['hand_normals']["{}".format(key)][j][2]))

                if j < len(grasps_dict['hand_normals']["{}".format(key)])-1:
                    grasps_file.write(", ")

        grasps_file.write("]")
        if counter < len(grasps_dict['hand_normals'].keys())-1:
                    grasps_file.write(", ")
        counter += 1

    grasps_file.write("},")


    # contact points
    grasps_file.write('"contact_points": [')

    for j in range(len(grasps_dict['collision_pts'])):
        grasps_file.write("[{}, {}, {}]".format(grasps_dict['collision_pts'][j][0],grasps_dict['collision_pts'][j][1],grasps_dict['collision_pts'][j][2]))

        if j < len(grasps_dict['collision_pts'])-1:
            grasps_file.write(", ")

    grasps_file.write("],")

    # contact normals
    grasps_file.write('"contact_normals": [')

    for j in range(len(grasps_dict['collision_normals'])):
        grasps_file.write("[{}, {}, {}]".format(grasps_dict['collision_normals'][j][0],grasps_dict['collision_normals'][j][1],grasps_dict['collision_normals'][j][2]))

        if j < len(grasps_dict['collision_normals'])-1:
            grasps_file.write(", ")

    grasps_file.write("]}")


def make_dest_dir(env):
    import os
    from pathlib import Path

    # Specify the parent directory path and the directory name
    parent_directory = "/home/vislab/DA2/Grasps_Register/R"
    
    for obj in env.obj_name:    

        # Create the full path to the new directory
        new_directory_path = os.path.join(parent_directory, obj)
        
        # Check if the directory exists
        if not(Path(new_directory_path).exists()):
            # Create the directory
            os.makedirs(new_directory_path)


def get_cam_frame(cam_point_clouds, cam_frames, rand_point):
    for i in range(len(cam_point_clouds)):
        if rand_point in cam_point_clouds[i].points:
            return cam_frames[i]


def test_floating_gripper(args):
    from gog.robot_hand import BarrettHand
    from gog.robot_hand import RH8DLHand
    from gog.robot_hand import CShapeHand
    from gog.robot_hand import RH8DRHand
    from gog.environment import Environment
    from gog.policy import GraspingPolicy
    from gog.utils.orientation import Quaternion
    import yaml
    import numpy as np

    with open('../yaml/params.yml', 'r') as stream:
        params = yaml.safe_load(stream)
    # robot_hand = BarrettHand(urdf_file='../assets/robot_hands/barrett/bh_282.urdf')
    # robot_hand = RH8DLHand(urdf_file='../assets/robot_hands/RH8DL/RH8DL.urdf', contacts_file_name='contacts_left_hand')
    robot_hand = RH8DRHand(urdf_file='../assets/robot_hands/RH8DR/RH8DR.urdf', contacts_file_name='contacts_right_hand_10')

    env = Environment(assets_root='../assets', params=params['env'])
    policy = GraspingPolicy(robot_hand=robot_hand, params=params)
    policy.seed(args.seed)

    make_dest_dir(env)

    rng = np.random.RandomState()
    rng.seed(args.seed)
   
    n_episodes = len(env.obj_name)

    for j in range(n_episodes):
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        print('Episode:{}, seed:{}'.format(j, episode_seed))
        env.seed(episode_seed)  

        total_grasp_order = 0

        obs = env.reset(j)

        while True:
            print('-----------------------')

            state, cam_point_clouds, cam_frames = policy.state_representation(obs, plot=False)

            # if len(np.asarray(state.points)) < 500:
            #     break

            # Sample candidates
            candidates = policy.get_candidates(state, plot=False)
            print('{} grasp candidates found...!'.format(len(candidates)))

            cam_frame=[]

            actions = []
            preshapes = []
            for i in range(len(candidates)): 
                text = str(i+1) + '.'
                print(text, end='\r')

                candidate = candidates[i]
                # Local reconstruction
                # rec_point_cloud = policy.reconstruct(candidate, plot=False)
                # rec_point_cloud = rec_point_cloud.transform(candidate['pose'])

                # text += 'Local region reconstructed.'
                # print(text, end='\r')

                # Grasp optimization
                init_preshape = {'pose': candidate['pose'], 'finger_joints': candidate['finger_joints']}
                opt_preshape = policy.optimize(init_preshape, state, plot=True)

                # text += 'Grasp optimized e: ' + str(np.abs(opt_preshape['score']))
                # print(text)
                print("Grasp optimized")

                total_grasp_order += 1

                # cam_frame = get_cam_frame(cam_point_clouds, cam_frames, candidates_rand_points[i])

                # Grasp registration
                grasp = {'pose': opt_preshape['pose'], 'pre_pose': candidate['pose'], 'local_frame': candidate['local_frame'], 
                         'finger_joints': opt_preshape['finger_joints'], 'pre_finger_joints': candidate['finger_joints'], 'grasp_center': opt_preshape['grasp_center'],
                         # 'score': np.abs(opt_preshape['score']), 
                         'contact_pts_link': opt_preshape['contact_pts_link'], 'enclosing_pts': np.asarray(candidate['enclosing_pts'].points), 
                         'hand_contacts': opt_preshape['hand_contacts']['points'], 'hand_normals': opt_preshape['hand_contacts']['normals'],
                         'collision_pts': opt_preshape['collision_pts'], 'collision_normals': opt_preshape['collision_normals']}
            
                write_grasps_file(grasp, total_grasp_order, obs['name'])

                # robot_hand.update_links(joint_values=opt_preshape['finger_joints'],
                #                         palm_pose=opt_preshape['pose'])

                # hand_contacts = robot_hand.get_contacts()
                # pcd_contacts = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(hand_contacts['points']))
                # pcd_contacts.paint_uniform_color([1, 0, 0])
                # o3d.visualization.draw_geometries([pcd_contacts, state])

                # obs = env.step(action)

            break
         


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_episodes', default='2', type=int, help='')
    parser.add_argument('--plot', action='store_true', default=False, help='')
    parser.add_argument('--seed', default=2, type=int, help='')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test_floating_gripper(args)

    # test_c_shape()