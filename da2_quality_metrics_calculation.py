# Initial Dataset attributes: pose, cam_frame, local_frame, finger_joints, grasp_center, score, enclosing_points, contact_points, contact_normals
#
#    Final Dataset | pose_R, finger_joints_R, score_R, enclosing_points_R, contact_points_R, contact_normals_R
#     attributes   | pose_L, finger_joints_L, score_L, enclosing_points_L, contact_points_L, contact_normals_L

from dexnet.grasping import PointGraspMetrics3D
import os
from autolab_core import YamlConfig
import numpy as np
import open3d as o3d
from geometric_object_grasper.gog.robot_hand import RH8DLHand, RH8DRHand
from geometric_object_grasper.gog.environment_original import Environment
from geometric_object_grasper.gog.utils.orientation import Quaternion, rot_z
from geometric_object_grasper.gog.grasp_optimizer import ShapeComplementarityMetric
import h5py
import yaml
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_point_cloud_size(point_cloud):

    # Calculate the extent in each dimension
    min_bound = point_cloud.get_min_bound()
    max_bound = point_cloud.get_max_bound()

    # Calculate the size in each dimension
    size_z = max_bound[2] - min_bound[2]

    return min_bound[2]


def dual_grasp_vis_test(grasp_R, grasp_L, obj_path, env):

    obs = env.reset(obj_path=obj_path)


    action0_L = {'pos': np.array([grasp_L['pre_pose'][0][3], grasp_L['pre_pose'][1][3], grasp_L['pre_pose'][2][3]]),
               'quat': Quaternion().from_rotation_matrix(np.array(grasp_L['pre_pose'])[0:3, 0:3]),
               'finger_joints': np.array(grasp_L['pre_finger_joints'])}
    action1_L = {'pos': np.array([grasp_L['pose_pre_transf'][0][3], grasp_L['pose_pre_transf'][1][3], grasp_L['pose_pre_transf'][2][3]]),
               'quat': Quaternion().from_rotation_matrix(np.array(grasp_L['pose_pre_transf'])[0:3, 0:3]),
               'finger_joints': np.array(grasp_L['finger_joints'])}
    action_L = [action0_L, action1_L]


    action0_R = {'pos': np.array([grasp_R['pre_pose'][0][3], grasp_R['pre_pose'][1][3], grasp_R['pre_pose'][2][3]]),
               'quat': Quaternion().from_rotation_matrix(np.array(grasp_R['pre_pose'])[0:3, 0:3]),
               'finger_joints': np.array(grasp_R['pre_finger_joints'])}
    action1_R = {'pos': np.array([grasp_R['pose_pre_transf'][0][3], grasp_R['pose_pre_transf'][1][3], grasp_R['pose_pre_transf'][2][3]]),
               'quat': Quaternion().from_rotation_matrix(np.array(grasp_L['pose_pre_transf'])[0:3, 0:3]),
               'finger_joints': np.array(grasp_R['finger_joints'])}
    action_R = [action0_R, action1_R]


    obs = env.pregrasp_step(action0_L, hand='L')
    obs = env.pregrasp_step(action0_R, hand='R')

    obs = env.grasp_step(action1_L, hand='L')
    obs = env.grasp_step(action1_R, hand='R')

    obs = env.drop_recover_step(action1_L, hand='L')
    obs = env.drop_recover_step(action1_R, hand='R')


def grasps_visualization(grasp_R, grasp_L, point_cloud, obj_T, valid_contact_points=[], valid_contact_normals=[], torque_vector=[]):

    robot_R = RH8DRHand(urdf_file='/home/vislab/DA2/geometric_object_grasper/assets/robot_hands/RH8DR/RH8DR.urdf', contacts_file_name='contacts_right_hand_10')
    robot_L = RH8DLHand(urdf_file='/home/vislab/DA2/geometric_object_grasper/assets/robot_hands/RH8DL/RH8DL.urdf', contacts_file_name='contacts_left_hand')

    mesh = o3d.io.read_triangle_mesh(point_cloud)
    mesh = rotate_point_cloud(mesh)
    mesh.paint_uniform_color([0.0, 0.5, 0.5])
    mesh_size_z = get_point_cloud_size(mesh)

    robot_R.update_links(joint_values= grasp_R['finger_joints'], palm_pose=grasp_R['pose'])

    robot_L.update_links(joint_values= grasp_L['finger_joints'], palm_pose=grasp_L['pose'])
    
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.07)

    hand_R_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    hand_R_frame = hand_R_frame.transform(np.array(grasp_R['pose']))

    hand_L_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
    hand_L_frame = hand_L_frame.transform(np.array(grasp_L['pose']))

    ref_L_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
    ref_L_frame = ref_L_frame.transform(np.matmul(np.array(grasp_L['pose']), np.linalg.inv(np.array(robot_L.ref_frame))))

    # rand_point_L_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    # rand_point_L_frame = rand_point_L_frame.transform(np.linalg.inv(np.array(grasp_L['local_frame'])))

    # obj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.07)
    # obj_frame = obj_frame.transform(obj_T)
    
    contact_pt_R = o3d.geometry.PointCloud()
    contact_pt_R.points = o3d.utility.Vector3dVector(grasp_R['contact_points'])
    contact_pt_R.normals = o3d.utility.Vector3dVector(grasp_R['contact_normals'])
    contact_pt_R.paint_uniform_color([0.5, 0.0, 0.0])

    contact_pt_L = o3d.geometry.PointCloud()
    contact_pt_L.points = o3d.utility.Vector3dVector(grasp_L['contact_points'])
    contact_pt_L.normals = o3d.utility.Vector3dVector(grasp_L['contact_normals'])
    contact_pt_L.paint_uniform_color([0.5, 0.0, 0.0])

    hand_contact_R = o3d.geometry.PointCloud()
    hand_contact_R.points = o3d.utility.Vector3dVector(grasp_R['hand_contacts']['points'])
    hand_contact_R.normals = o3d.utility.Vector3dVector(grasp_R['hand_contacts']['normals'])
    hand_contact_R.paint_uniform_color([0.0, 0.7, 0.0])

    hand_contact_L = o3d.geometry.PointCloud()
    hand_contact_L.points = o3d.utility.Vector3dVector(grasp_L['hand_contacts']['points'])
    hand_contact_L.normals = o3d.utility.Vector3dVector(grasp_L['hand_contacts']['normals'])
    hand_contact_L.paint_uniform_color([0.0, 0.7, 0.0])

    valid_contact_pts = o3d.geometry.PointCloud()
    valid_contact_pts.points = o3d.utility.Vector3dVector(valid_contact_points)
    valid_contact_pts.normals = o3d.utility.Vector3dVector(valid_contact_normals)
    valid_contact_pts.paint_uniform_color([0.0, 0.5, 0.0])

    enclosing_pt_cloud_R = o3d.geometry.PointCloud()
    enclosing_pt_cloud_R.points = o3d.utility.Vector3dVector(grasp_R['enclosing_points'])
    enclosing_pt_cloud_R.paint_uniform_color([0.5, 0.0, 0.0])

    enclosing_pt_cloud_L = o3d.geometry.PointCloud()
    enclosing_pt_cloud_L.points = o3d.utility.Vector3dVector(grasp_L['enclosing_points'])
    enclosing_pt_cloud_L.paint_uniform_color([0.5, 0.0, 0.0])

    grasp_center_R = o3d.geometry.TriangleMesh.create_sphere(radius=0.07)
    grasp_center_R.translate(grasp_R['grasp_center'])
    grasp_center_R.paint_uniform_color([0.0, 0.5, 0.0])

    grasp_center_L = o3d.geometry.TriangleMesh.create_sphere(radius=0.07)
    grasp_center_L.translate(grasp_L['grasp_center'])
    grasp_center_L.paint_uniform_color([0.0, 0.5, 0.0])

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.005,
        cone_radius=0.01,
        cylinder_height=0.1,
        cone_height=0.02
    )
    arrow.paint_uniform_color([1, 0.0, 0.0])

    z_axis = np.array([0, 0, 1])
    v = np.cross(z_axis, torque_vector[:,0:3][0])
    c = np.dot(z_axis, torque_vector[:,0:3][0])
    if np.linalg.norm(v) < 1e-6:  # If vectors are aligned, no rotation is needed
        R = np.eye(3)
    else:
        v_skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + v_skew + v_skew @ v_skew * ((1 - c) / (np.linalg.norm(v) ** 2))

    # Apply the rotation and translate the arrow to the origin
    arrow.rotate(R, center=(0, 0, 0))
    arrow.translate([0,0,0])


    # Rotate arrow to align with the vector direction
    # arrow_direction = torque_vector[:,0:3][0] / np.linalg.norm(torque_vector[:,0:3][0])
    # print(f"Arrow direction: {arrow_direction}")
    # print(f"Torque vector: {torque_vector[:,0:3][0]}")
    # arrow.rotate(arrow.get_rotation_matrix_from_xyz((0, 0, np.arccos(arrow_direction[2]))), center=(0, 0, 0))
    # arrow.translate([0,0,0])

    visuals= robot_L.get_visual_map(boxes=False, frames=False, contacts=False)
    visuals.extend(robot_R.get_visual_map(boxes=False, frames=False, contacts=False))
    
    visuals.append(base_frame)

    # visuals.append(grasp_center_R)
    # visuals.append(grasp_center_L)
    
    # visuals.append(mesh)

    # visuals.append(object_pts_vis)

    # visuals.append(contact_pt_R)
    # visuals.append(contact_pt_L)

    # visuals.append(hand_contact_L)
    # visuals.append(hand_contact_R)

    # visuals.append(grasp_center_R)
    # visuals.append(grasp_center_L)

    # visuals.append(valid_contact_pts)

    # visuals.append(enclosing_pt_cloud_R)
    # visuals.append(enclosing_pt_cloud_L)
    visuals.append(arrow)

    # Visualize multiple geometries
    o3d.visualization.draw_geometries(visuals, point_show_normal=True)
    # o3d.visualization.draw_geometries([mesh, contact_pt_R, contact_pt_L]) 


def check_z_axes_alignment(grasp_R, grasp_L, tolerance=0.35):

    # Extract the third column (z-axis) from the rotation submatrix
    z_axis_R = np.array([grasp_R['pose'][0][2], grasp_R['pose'][1][2], grasp_R['pose'][2][2]])
    z_axis_L = np.array([grasp_L['pose'][0][2], grasp_L['pose'][1][2], grasp_L['pose'][2][2]])

    # Compute the dot product between the two z-axis vectors
    dot_product = np.dot(z_axis_R, z_axis_L)

    # Check if the dot product is close to 1 (indicating alignment)
    if np.abs(dot_product-(-1)) < tolerance:
        return 1
    else:
        return 0    


def verify_dist_between_grasps(grasp_R, grasp_L):

    min_dist = 0.14 # 0.7 is the radium of a esphere that covers one hand
    max_dist = 2

    grasp_R_center = np.array([grasp_R['grasp_center'][0], grasp_R['grasp_center'][1], grasp_R['grasp_center'][2]])
    grasp_L_center = np.array([grasp_L['grasp_center'][0], grasp_L['grasp_center'][1], grasp_L['grasp_center'][2]])

    center_dist = np.linalg.norm(grasp_R_center - grasp_L_center)

    if max_dist > center_dist > min_dist:
        return 1
    else:
        return 0


def list_of_objects(data_folder):

    # Get a list of all entries (files and directories) in the directory
    entries = os.listdir(data_folder)
    # Filter out only the directories
    objects = [entry for entry in entries if os.path.isdir(os.path.join(data_folder, entry))]

    return objects


def read_init_dataset(data_folder):
    # importing the module 
    import json

    objects = list_of_objects(data_folder)
    
    data_set = {}
    for object in objects:
        js = {}
        i = 1
        while(1):
            
            try:
                # reading the data from the file
                with open(os.path.join(data_folder,'{}/{}.txt'.format(object, i)), 'r') as f: 
                    
                    # js['{}'.format(i)] = json.loads(f.read())

                    data = f.read() 
                
                    # reconstructing the data as a dictionary 
                    js_aux = json.loads(data)

                    js['{}'.format(i)] = js_aux

                    i += 1
                    
            except FileNotFoundError:
                break
        
        data_set['{}'.format(object)] = js

    return data_set, objects


def compute_object_frame(obj):
# input: path to object
    
    plot = False

    # Compute object centroid
    mesh = o3d.io.read_triangle_mesh(obj)
    centroid = np.asarray(mesh.get_center())

    frame_z = np.array([0, 0, 1])

    frame_x = (np.transpose((1/np.sqrt(pow(frame_z[1],2)+pow(frame_z[2],2)))*np.array([frame_z[2], -frame_z[1], 0]))).flatten()

    frame_y = np.cross(frame_z, frame_x)

    T = np.eye(4)
    T = [[frame_x[0], frame_y[0], frame_z[0], centroid[0]],
         [frame_x[1], frame_y[1], frame_z[1], centroid[1]],
         [frame_x[2], frame_y[2], frame_z[2], centroid[2]],
         [0, 0, 0, 1]]

    
    if plot:
        centroid_vis = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        centroid_vis.compute_vertex_normals()
        centroid_vis.paint_uniform_color([1, 0, 0])  # Set color to red
        centroid_vis.translate(centroid)

        T_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        T_vis = T_vis.transform(T)

        # Visualize the mesh and the centroid
        o3d.visualization.draw_geometries([mesh, T_vis])

    return T


def write_final_data_set(enclosing_points, grasp_R_pose, grasp_L_pose, finger_joints_R, finger_joints_L, score_R, score_L, dext, force_closure, Torque, obj_name, grasp_counter, valid_contact_points, valid_contact_normals, torque_vector):
    
    scale = 1

    # transformation of enclosing_points
    num_max_enclosing_pts = 0
    for j in range(grasp_counter):
        if len(enclosing_points[j]) > num_max_enclosing_pts:
                num_max_enclosing_pts = len(enclosing_points[j])
    
    enclosing_pts = np.zeros((grasp_counter, num_max_enclosing_pts, 3))

    for j in range(grasp_counter):
        enclosing_pts[j][:enclosing_points[j].shape[0], :] = enclosing_points[j]

    # transformation of valid contact points
    num_max_valid_contact_points = 0
    for j in range(grasp_counter):
        if len(valid_contact_points[j][0]) > num_max_valid_contact_points:
                num_max_valid_contact_points = len(valid_contact_points[j][0])
    
    valid_contact_pts = np.zeros((grasp_counter, num_max_valid_contact_points, 3))

    for j in range(grasp_counter):
        valid_contact_pts[j][:valid_contact_points[j][0].shape[0], :] = valid_contact_points[j][0]

    #transformation of valid contact normals    
    num_max_valid_contact_normals = 0
    for j in range(grasp_counter):
        if len(valid_contact_normals[j][0]) > num_max_valid_contact_normals:
                num_max_valid_contact_normals = len(valid_contact_normals[j][0])
    
    valid_contact_nls = np.zeros((grasp_counter, num_max_valid_contact_normals, 3))

    for j in range(grasp_counter):
        valid_contact_nls[j][:valid_contact_normals[j][0].shape[0], :] = valid_contact_normals[j][0]

    data = h5py.File("/home/vislab/DA2/final_data_set" + '/' + obj_name.split('.')[0] + '.h5', 'w')
    temp1 = data.create_group("grasps")
    temp1["transforms"] = [(grasp_R_pose[i], grasp_L_pose[i]) for i in range(grasp_counter)]
    temp1["finger_joints"] = [(finger_joints_R[i], finger_joints_L[i]) for i in range(grasp_counter)]
    temp1["grasp_points"] = [np.array(enclosing_pts[i]) for i in range(grasp_counter)]
    temp1["valid_grasp_points"] = [np.array(valid_contact_pts[i]) for i in range(grasp_counter)]
    temp1["valid_grasp_normals"] = [np.array(valid_contact_nls[i]) for i in range(grasp_counter)]
    temp1["qualities/shape_score"] = [(score_R[i], score_L[i]) for i in range(grasp_counter)]
    temp1["qualities/Force_closure"] = np.array(force_closure).reshape(len(force_closure), -1) if len(force_closure)!=0 else []
    temp1["qualities/Dexterity"] = np.array(dext).reshape(len(dext), -1) if len(dext)!=0 else []
    temp1["qualities/Torque_optimization"] = np.array(Torque).reshape(len(Torque), -1) if len(Torque)!=0 else []
    temp1["Torque_vector"] = [(torque_vector[i]) for i in range(grasp_counter)]
    temp2 = data.create_group("object")
    temp2["file"] = obj_name.split('/')[-1]
    temp2["scale"] = scale


def read_h5_file():

    # Open the HDF5 file
    with h5py.File('/home/vislab/DA2/final_data_set/1a96d308eef57195efaa61516f88b67_1.h5', 'r') as file:
        
        # Print the keys at the root level of the file
        print("Root keys:")
        print(list(file.keys()))
        
        # You can navigate through the file's structure to see its contents
        # For example, if there's a group named 'group1' inside the root:
        if 'grasps' in file:
            print("\nContents of grasps:")
            for key in file['grasps'].keys():
                print(key)
                # print("{}: {}".format(key,file['grasps'][key][:]))
                if key == "qualities":
                    for key_sub in file['grasps']['qualities'].keys():
                        print("\t{}".format(key_sub))

        if 'object' in file:
            print("\nContents of object:")
            for key in file['object'].keys():
                print(key)
                # print("{}: {}".format(key,file['object'][key][:]))

def hand_points_data_transf(hand_points, hand_normals):

    points_aux = []
    normals_aux = []
    link_aux = []

    for key in hand_points.keys():
        for j in range(len(hand_points["{}".format(key)])):  
            points_aux.append(hand_points["{}".format(key)][j])
            normals_aux.append(hand_normals["{}".format(key)][j])
            link_aux.append(key)
    
    return {'points': np.asarray(points_aux), 
            'normals': np.asarray(normals_aux),
            'links': link_aux}


def data_transform(grasp_R, grasp_L, point_cloud, obj_T):

    # input object mesh
    mesh = o3d.io.read_triangle_mesh(point_cloud)
    mesh_size_z = get_point_cloud_size(mesh)

    # Right hand
    for i in range(len(data_set_R['{}'.format(obj)])):

        # pose R
        grasp_R_frame = np.array(grasp_R['{}'.format(i+1)]['pose'])
        grasp_R['{}'.format(i+1)]['pose'] = grasp_R_frame
        grasp_R['{}'.format(i+1)]['pose'][2][3] += mesh_size_z

        # enclosing R
        enclosing_pt_cloud_R = o3d.geometry.PointCloud()
        enclosing_pt_cloud_R.points = o3d.utility.Vector3dVector(grasp_R['{}'.format(i+1)]['enclosing_points'])
        enclosing_pt_cloud_R.paint_uniform_color([0.5, 0.0, 0.0])
        grasp_R['{}'.format(i+1)]['enclosing_points'] = np.asarray(enclosing_pt_cloud_R.points)
        for j in range(len(grasp_R['{}'.format(i+1)]['enclosing_points'])):
            grasp_R['{}'.format(i+1)]['enclosing_points'][j][2] += mesh_size_z 

        # contact points and normals R
        contact_pt_R = o3d.geometry.PointCloud()
        contact_pt_R.points = o3d.utility.Vector3dVector(grasp_R['{}'.format(i+1)]['contact_points'])
        contact_pt_R.normals = o3d.utility.Vector3dVector(grasp_R['{}'.format(i+1)]['contact_normals'])
        grasp_R['{}'.format(i+1)]['contact_points'] = np.asarray(contact_pt_R.points)
        grasp_R['{}'.format(i+1)]['contact_normals'] = np.asarray(contact_pt_R.normals)
        for j in range(len(grasp_R['{}'.format(i+1)]['contact_points'])):
            grasp_R['{}'.format(i+1)]['contact_points'][j][2] += mesh_size_z

        # hand points and normals R
        grasp_R['{}'.format(i+1)]['hand_contacts'] = hand_points_data_transf(grasp_R['{}'.format(i+1)]['hand_contacts'], grasp_R['{}'.format(i+1)]['hand_normals'])
        grasp_R['{}'.format(i+1)]['hand_normals'] = []

        hand_contacts_R = o3d.geometry.PointCloud()
        hand_contacts_R.points = o3d.utility.Vector3dVector(grasp_R['{}'.format(i+1)]['hand_contacts']['points'])
        hand_contacts_R.normals = o3d.utility.Vector3dVector(grasp_R['{}'.format(i+1)]['hand_contacts']['normals'])
        grasp_R['{}'.format(i+1)]['hand_contacts']['points'] = np.asarray(hand_contacts_R.points)
        grasp_R['{}'.format(i+1)]['hand_contacts']['normals'] = np.asarray(hand_contacts_R.normals)

        for j in range(len(grasp_R['{}'.format(i+1)]['hand_contacts']['points'])):
            grasp_R['{}'.format(i+1)]['hand_contacts']['points'][j][2] += mesh_size_z

        # grasp center R
        grasp_R['{}'.format(i+1)]['grasp_center'][2] += mesh_size_z


    #Left hand
    for i in range(len(data_set_L['{}'.format(obj)])):

        # pose L
        grasp_L_frame = np.array(grasp_L['{}'.format(i+1)]['pose'])
        grasp_L['{}'.format(i+1)]['pose'] = grasp_L_frame
        grasp_L['{}'.format(i+1)]['pose'][2][3] += mesh_size_z

        # enclosing L
        enclosing_pt_cloud_L = o3d.geometry.PointCloud()
        enclosing_pt_cloud_L.points = o3d.utility.Vector3dVector(grasp_L['{}'.format(i+1)]['enclosing_points'])
        enclosing_pt_cloud_L.paint_uniform_color([0.5, 0.0, 0.0])
        grasp_L['{}'.format(i+1)]['enclosing_points'] = np.asarray(enclosing_pt_cloud_L.points) 
        for j in range(len(grasp_L['{}'.format(i+1)]['enclosing_points'])):
            grasp_L['{}'.format(i+1)]['enclosing_points'][j][2] += mesh_size_z

        # contact points and normals L
        contact_pt_L = o3d.geometry.PointCloud()
        contact_pt_L.points = o3d.utility.Vector3dVector(grasp_L['{}'.format(i+1)]['contact_points'])
        contact_pt_L.normals = o3d.utility.Vector3dVector(grasp_L['{}'.format(i+1)]['contact_normals'])
        grasp_L['{}'.format(i+1)]['contact_points'] = np.asarray(contact_pt_L.points)
        grasp_L['{}'.format(i+1)]['contact_normals'] = np.asarray(contact_pt_L.normals)
        for j in range(len(grasp_L['{}'.format(i+1)]['contact_points'])):
            grasp_L['{}'.format(i+1)]['contact_points'][j][2] += mesh_size_z

        # hand points and normals R
        grasp_L['{}'.format(i+1)]['hand_contacts'] = hand_points_data_transf(grasp_L['{}'.format(i+1)]['hand_contacts'], grasp_L['{}'.format(i+1)]['hand_normals'])
        grasp_L['{}'.format(i+1)]['hand_normals'] = []

        hand_contacts_L = o3d.geometry.PointCloud()
        hand_contacts_L.points = o3d.utility.Vector3dVector(grasp_L['{}'.format(i+1)]['hand_contacts']['points'])
        hand_contacts_L.normals = o3d.utility.Vector3dVector(grasp_L['{}'.format(i+1)]['hand_contacts']['normals'])
        grasp_L['{}'.format(i+1)]['hand_contacts']['points'] = np.asarray(hand_contacts_L.points)
        grasp_L['{}'.format(i+1)]['hand_contacts']['normals'] = np.asarray(hand_contacts_L.normals)

        for j in range(len(grasp_L['{}'.format(i+1)]['hand_contacts']['points'])):
            grasp_L['{}'.format(i+1)]['hand_contacts']['points'][j][2] += mesh_size_z

        # grasp center L
        grasp_L['{}'.format(i+1)]['grasp_center'][2] += mesh_size_z

    return grasp_R, grasp_L


def print_final_data_set_obj(enclosing_pts, grasp_R_pose, grasp_L_pose, dext, force_closure, Torque, obj_name, grasp_counter):

    print("--- Object: {} ---".format(obj_name))
    print("Number of grasps: {}".format(grasp_counter+1))

    for i in range(grasp_counter):

        print("Enclosing points: {}".format(enclosing_pts[i]))
        print("Right grasp pose: {}".format(grasp_R_pose[i]))
        print("Left grasp pose: {}".format(grasp_L_pose[i]))
        print("Dexterity: {}".format(dext[i]))
        print("Force closure: {}".format(force_closure[i]))
        print("Torque: {}".format(Torque[i]))


def regist_contacts_finger(obj_name, hands_num, grasp_final_ident, contacts_R, contacts_L, score_R, score_L, force_closure, dex, torque, valid_hand_contacts_R, valid_hand_contacts_L, dex_R, dex_L, torque_R, torque_L):   

    contacts_finger_file = open('contacts_finger', 'w')

    for i in range(len(obj_name)):

        # contacts_finger_file.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(force_closure[i][0],
        #                                                                                     contacts_R[i][0], contacts_R[i][1], contacts_R[i][2], contacts_R[i][3], contacts_R[i][4], contacts_R[i][5],
        #                                                                                     contacts_L[i][0], contacts_L[i][1], contacts_L[i][2], contacts_L[i][3], contacts_L[i][4], contacts_L[i][5],
        #                                                                                     contacts_R[i][0]+contacts_R[i][1]+contacts_R[i][2]+contacts_R[i][3]+contacts_R[i][4]+contacts_R[i][5],
        #                                                                                     contacts_L[i][0]+contacts_L[i][1]+contacts_L[i][2]+contacts_L[i][3]+contacts_L[i][4]+contacts_L[i][5],
        #                                                                                     contacts_R[i][0]+contacts_R[i][1]+contacts_R[i][2]+contacts_R[i][3]+contacts_R[i][4]+contacts_R[i][5]+contacts_L[i][0]+contacts_L[i][1]+contacts_L[i][2]+contacts_L[i][3]+contacts_L[i][4]+contacts_L[i][5],
        #                                                                                     score_R[i], score_L[i], score_R[i]+score_L[i], dex[i][0]))

        contacts_finger_file.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(obj_name[i], grasp_final_ident[i], hands_num[i][0], hands_num[i][1], force_closure[i][0], score_R[i], score_L[i], dex[i][0], dex_R[i][0], dex_L[i][0], torque[i][0], torque_R[i][0], torque_L[i][0], valid_hand_contacts_R[i], valid_hand_contacts_L[i]))


def write_statistic_doc(statistics, histogram=False):

    statistics_file = open('Final_Statistics', 'w')

    statistics_file.write("\nThis data set was created with {} objects.\nIn {} of those objects, the algorithm could not find any grasp.\n\n".format(statistics['num_objects'], statistics['num_objects_not_used']))

    statistics_file.write('{} grasps were discared due to irregular distance between grasps.\n'.format(statistics['num_grasps_unvalid_dist']))

    statistics_file.write('{} grasps were discared because they had no valid contact points.\n'.format(statistics['num_grasps_unvalid_contact_pts']))

    statistics_file.write("\nNumber of grasps\n\n")

    statistics_file.write("\tTotal:  {}\n".format(sum(statistics['grasp_counter'])))
    
    # average number of grasps by object
    statistics_file.write("\tAverage by object:  {}\n".format(sum(statistics['grasp_counter'])/len(statistics['grasp_counter'])))

    # max and min number of grasps
    statistics_file.write("\tMax and Min in an object:  {}  |  {}\n".format(max(statistics['grasp_counter']), 
                                                                            min(statistics['grasp_counter'])))

    # Evalutation metrics
    statistics_file.write("\nEvaluation metrics (Average | Max | Min)\n\n")

    shape_complementarity = [arr for arr in statistics['shape_complementarity'] if len(arr) > 0]
    shape_complementarity = np.concatenate(shape_complementarity)
    statistics_file.write("\tShape Complementarity:  {}  |  {}  |  {}\n".format(sum(shape_complementarity)/len(shape_complementarity),
                                                                         max(shape_complementarity),
                                                                         min(shape_complementarity)))

    force_closure = [arr for arr in statistics['force_closure'] if len(arr) > 0]
    force_closure = np.concatenate(force_closure)
    statistics_file.write("\tForce Closure:  {}  |  {}  |  {}\n".format(sum(force_closure)/len(force_closure),
                                                                         max(force_closure),
                                                                         min(force_closure)))
    
    dexterity = [arr for arr in statistics['dexterity'] if len(arr) > 0]
    dexterity = np.concatenate(dexterity) 
    statistics_file.write("\tDexterity:  {}  |  {}  |  {}\n".format(sum(dexterity)/len(dexterity),
                                                                 max(dexterity),
                                                                 min(dexterity)))
    
    torque_optimization = [arr for arr in statistics['torque_optimization'] if len(arr) > 0]
    torque_optimization = np.concatenate(torque_optimization)
    statistics_file.write("\tForce Optimization:  {}  |  {}  |  {}\n".format(sum(torque_optimization)/len(torque_optimization),
                                                                            max(torque_optimization),
                                                                            min(torque_optimization)))

    # Enclosing Points
    statistics_file.write("\nEnclosing Points\n\n")
    
    max_encl = 0
    min_encl = 9999999
    num_encl = 0
    sum_encl = 0
    for i in range(len(statistics['num_enclosing_pts'])):
        if len(statistics['num_enclosing_pts'][i]) != 0:
            sum_encl += sum(statistics['num_enclosing_pts'][i])
            num_encl += len(statistics['num_enclosing_pts'][i])
            if max(statistics['num_enclosing_pts'][i]) > max_encl:
                max_encl = max(statistics['num_enclosing_pts'][i])
            if min(statistics['num_enclosing_pts'][i]) < min_encl:
                min_encl = min(statistics['num_enclosing_pts'][i])

    statistics_file.write("\tAverage by object:  {}\n".format(sum_encl/num_encl))
    # max and min number of grasps
    statistics_file.write("\tMax and Min in an object:  {}  |  {}\n".format(max_encl, min_encl))

    # Contact Points
    statistics_file.write("\nContact Points\n\n")
    max_cont = 0
    min_cont = 9999999
    num_cont = 0
    sum_cont = 0
    for i in range(len(statistics['num_contact_pts'])):
        if len(statistics['num_contact_pts'][i]) != 0:
            sum_cont += sum(statistics['num_contact_pts'][i])
            num_cont += len(statistics['num_contact_pts'][i])
            if max(statistics['num_contact_pts'][i]) > max_cont:
                max_cont = max(statistics['num_contact_pts'][i])
            if min(statistics['num_contact_pts'][i]) < min_cont:
                min_cont = min(statistics['num_contact_pts'][i])

    statistics_file.write("\tAverage by object:  {}\n".format(sum_cont/num_cont))
    # max and min number of grasps
    statistics_file.write("\tMax and Min in an object:  {}  |  {}\n".format(max_cont, min_cont))

    if histogram == True:
        plot_3d_histogram(force_closure, shape_complementarity, dexterity)


def plot_3d_histogram(force_closure, shape_complementarity, dexterity):
    
    # Generating random data for two variables
    x = force_closure.reshape((len(force_closure,))) # First variable
    # y = shape_complementarity.reshape((len(shape_complementarity,)))  # Second variable correlated with the first
    y = dexterity.reshape((len(dexterity,)))

    # Create 2D histogram data
    hist, xedges, yedges = np.histogram2d(x, y, bins=40 )

    # Create the 3D figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Construct arrays for the 3D bar positions
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)

    # The heights of the bars represent the frequency
    dx = 25
    dy = 0.0005
    dz = hist.ravel()

    # Plot 3D bars
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

    # Set the x-axis limit to 200
    ax.set_xlim(0, 2500)

    # Labels
    ax.set_xlabel('Force Closure')
    ax.set_ylabel('Shape Complementarity')
    ax.set_zlabel('Frequency')
    ax.set_title('3D Histogram of Force Closure vs Shape Complementarity')

    plt.show()


def rotate_point_cloud(point_cloud):

    # Calculate the extent in each dimension
    min_bound = point_cloud.get_min_bound()
    max_bound = point_cloud.get_max_bound()

    # Calculate the size in each dimension
    size_x = max_bound[0] - min_bound[0]
    size_y = max_bound[1] - min_bound[1]

    if size_x > size_y:
        return point_cloud.rotate(rot_z(np.pi/2))
    else:
        return point_cloud


def shape_complementarity_metric(grasp_R, grasp_L, point_cloud, shape_metric, obj_T):

    # mesh = o3d.io.read_triangle_mesh(point_cloud)
    # mesh = rotate_point_cloud(mesh)
    # mesh.compute_vertex_normals()
    contact_pts_R = {'points': grasp_R['contact_points'],
                  'normals': grasp_R['contact_normals']}
    contact_pts_L = {'points': grasp_L['contact_points'],
                  'normals': grasp_L['contact_normals']}

    # Right Hand
    shape_score_R, num_valid_hand_contacts_R, ids_valid_hand_points_R, ids_valid_object_points_R = shape_metric(grasp_R['hand_contacts'], target=contact_pts_R, collisions=grasp_R['contact_points'])          

    # Left Hand
    shape_score_L, num_valid_hand_contacts_L, ids_valid_hand_points_L, ids_valid_object_points_L = shape_metric(grasp_L['hand_contacts'], target=contact_pts_L, collisions=grasp_L['contact_points'])

    # grasps_visualization(data_set_R['{}'.format(obj)]['{}'.format(i+1)], data_set_L['{}'.format(obj)]['{}'.format(j+1)],
    #                                              os.path.join(obj_folder,'{}'.format(obj)), obj_T, object_pts)

    return shape_score_R, shape_score_L, num_valid_hand_contacts_R, num_valid_hand_contacts_L, ids_valid_hand_points_R, ids_valid_hand_points_L, ids_valid_object_points_R, ids_valid_object_points_L


def compute_valid_contact_points(grasp_R, grasp_L, ids_valid_object_points_R, ids_valid_object_points_L, ids_valid_hand_points_R, ids_valid_hand_points_L):

    # compute right hand contact points
    valid_contact_points_R = []
    valid_contact_normals_R = []
    for i in range(len(ids_valid_object_points_R)):
        valid_contact_points_R.append(grasp_R['contact_points'][np.asarray(ids_valid_object_points_R[i])][0])
        valid_contact_normals_R.append(grasp_R['hand_contacts']['normals'][ids_valid_hand_points_R[i]])

    # compute left hand contact points
    valid_contact_points_L = []
    valid_contact_normals_L = []
    for i in range(len(ids_valid_object_points_L)):
        valid_contact_points_L.append(grasp_L['contact_points'][np.asarray(ids_valid_object_points_L[i])][0])
        valid_contact_normals_L.append(grasp_L['hand_contacts']['normals'][ids_valid_hand_points_L[i]])

    # Join left and right hand
    valid_grasp_points = np.vstack((np.asarray(valid_contact_points_R), np.asarray(valid_contact_points_L)))
    valid_grasp_normals = np.vstack((np.asarray(valid_contact_normals_R), np.asarray(valid_contact_normals_L)))

    valid_grasp_points = valid_grasp_points[np.newaxis, :, :]
    valid_grasp_normals = valid_grasp_normals[np.newaxis, :, :]

    valid_contact_points_R = np.asarray(valid_contact_points_R)[np.newaxis, :, :]
    valid_contact_points_L = np.asarray(valid_contact_points_L)[np.newaxis, :, :]

    return valid_grasp_points, valid_grasp_normals, valid_contact_points_R, valid_contact_points_L 


if __name__ == "__main__":

    # used paths
    config_filename ="/home/vislab/DA2/api_config.yaml"
    obj_folder = "/home/vislab/DA2/geometric_object_grasper/assets/objects/unseen"
    data_folder_R = '/home/vislab/DA2/Grasps_Register/R/'
    data_folder_L = '/home/vislab/DA2/Grasps_Register/L/'

    # turn relative paths absolute
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)

    # with open('/home/vislab/DA2/geometric_object_grasper/yaml/params.yml', 'r') as stream:
    #     params_env = yaml.safe_load(stream)
    # env = Environment(assets_root='/home/vislab/DA2/geometric_object_grasper/assets', params=params_env['env'])

    shape_metric = ShapeComplementarityMetric()

    config = YamlConfig(config_filename)
    params={'friction_coef': config['sampling_friction_coef']}

    data_set_R, objects_R = read_init_dataset(data_folder_R)
    data_set_L, objects_L = read_init_dataset(data_folder_L)

    statistics = {}
    statistics['num_objects'] = len(max(objects_R, objects_L))
    statistics['num_objects_not_used'] = 0
    statistics['num_grasps_unvalid_dist'] = 0
    statistics['num_grasps_unvalid_contact_pts'] = 0
    statistics['grasp_counter'] = [] # number of grasps by object
    statistics['shape_complementarity'] = []
    statistics['force_closure'] = [] # force closure by grasp
    statistics['dexterity'] = [] # dexterity by grasp
    statistics['torque_optimization'] = [] # torque optimization by grasp
    statistics['num_enclosing_pts'] = [] # number of enclosing points by object
    statistics['num_contact_pts'] = [] # number of contact points by object
    

    contact_points_finger_R = []
    contact_points_finger_L = []
    score_R_2 = []
    score_L_2 = []
    f_2 = []
    d_2 = []
    d_R_2 = []
    d_L_2 = []
    t_2 = []
    t_L_2 = []
    t_R_2 = []
    obj_name = []
    hands_num = []
    grasp_final_ident = []
    num_valid_hand_contacts_R = []
    num_valid_hand_contacts_L = []
    
    # nr objects
    for k in range(len(data_set_R)):

        # check if the object exists in both R and L data sets and if the algorithm found any grasp for both hands
        if objects_R[k] in data_set_R and objects_R[k] in data_set_L and data_set_R['{}'.format(objects_R[k])]!={} and data_set_L['{}'.format(objects_R[k])]!={} :
        
            obj = objects_R[k]

            obj_T = compute_object_frame(os.path.join(obj_folder,'{}'.format(obj)))
            
            # Applying necessary transformation to data  
            data_set_R['{}'.format(obj)], data_set_L['{}'.format(obj)] = data_transform(data_set_R['{}'.format(obj)], data_set_L['{}'.format(obj)],
                                                                                        os.path.join(obj_folder,'{}'.format(obj)), obj_T)

            # Number of grasp pairs by object
            grasp_counter = 0

            aux_num_enclosing_pts = []
            aux_num_contact_pts = []
            shape_metric_stat = []
            score_R = []
            score_L = []
            a = []
            f = []
            d_array = []
            t_array = []
            torque_vector_array = []
            enclosing_points = []
            grasp_R_poses = []
            grasp_L_poses = []
            finger_joints_R = []
            finger_joints_L = []

            aux_valid_contact_points = []
            aux_valid_contact_normals = []

            # nr of R grasps
            for i in range(len(data_set_R['{}'.format(obj)])):
                # nr of L grasps
                for j in range(len(data_set_L['{}'.format(obj)])): 

                    if verify_dist_between_grasps(data_set_R['{}'.format(obj)]['{}'.format(i+1)], data_set_L['{}'.format(obj)]['{}'.format(j+1)]):

                        # Shape Complementarity
                        try:
                            shape_complementarity_R, shape_complementarity_L, num_valid_hand_contacts_R_aux, num_valid_hand_contacts_L_aux, ids_valid_hand_points_R, ids_valid_hand_points_L, ids_valid_object_points_R, ids_valid_object_points_L = shape_complementarity_metric(data_set_R['{}'.format(obj)]['{}'.format(i+1)], data_set_L['{}'.format(obj)]['{}'.format(j+1)], os.path.join(obj_folder,'{}'.format(obj)), shape_metric, obj_T)
                        except:
                            print("Exception computing Shape Complementarity Metric")
                            continue
                        
                        if len(ids_valid_object_points_R) == 0 or len(ids_valid_object_points_L) == 0 :
                            # print('Grasp declined: no valid contact points')
                            # grasps_visualization(data_set_R['{}'.format(obj)]['{}'.format(i+1)], data_set_L['{}'.format(obj)]['{}'.format(j+1)], os.path.join(obj_folder,'{}'.format(obj)), obj_T)
                            statistics['num_grasps_unvalid_contact_pts'] += 1
                            continue

                        valid_contact_points, valid_contact_normals, valid_contact_points_R, valid_contact_points_L = compute_valid_contact_points(data_set_R['{}'.format(obj)]['{}'.format(i+1)], data_set_L['{}'.format(obj)]['{}'.format(j+1)], ids_valid_object_points_R, ids_valid_object_points_L, ids_valid_hand_points_R, ids_valid_hand_points_L)

                        # Force closure
                        try:
                            a_aux, f_aux= PointGraspMetrics3D.Dual_force_closure_batch(valid_contact_points, -valid_contact_normals, params=params)
                        except:
                            print("Exception computing Force Closure")
                            continue
                        a.append(a_aux)
                        # print("force: {}".format(f_aux))

                        # Dexterity
                        try:
                            d_array_aux = PointGraspMetrics3D.Dexterity(valid_contact_points)

                            d_array_R_aux = PointGraspMetrics3D.Dexterity(valid_contact_points_R)
                            d_array_L_aux = PointGraspMetrics3D.Dexterity(valid_contact_points_L)
                        except:
                            print("Exception computing Dexterity")
                            continue
                        # print("d_array: {}".format(d_array_aux))

                        # Torque optimization
                        try:
                            t_array_aux, torque_vector = PointGraspMetrics3D.Torque_optimization(valid_contact_points)

                            t_array_R_aux, torque_vector_R = PointGraspMetrics3D.Torque_optimization(valid_contact_points_R)
                            t_array_L_aux, torque_vector_L = PointGraspMetrics3D.Torque_optimization(valid_contact_points_L)
                        except:
                            print("Exception computing Torque Optimization")
                            continue
                        # print("t_array: {}".format(t_array_aux))


                        # to record

                        shape_metric_stat.append((shape_complementarity_R+shape_complementarity_L)/2)
                        num_valid_hand_contacts_R.append(num_valid_hand_contacts_R_aux)
                        num_valid_hand_contacts_L.append(num_valid_hand_contacts_L_aux)

                        aux_valid_contact_points.append(valid_contact_points)
                        aux_valid_contact_normals.append(valid_contact_normals)

                        score_R_2.append(np.asarray(shape_complementarity_R))
                        score_L_2.append(np.asarray(shape_complementarity_L))
                        f_2.append(f_aux)
                        d_2.append(d_array_aux)
                        d_R_2.append(d_array_R_aux)
                        d_L_2.append(d_array_L_aux)
                        t_2.append(t_array_aux)
                        t_R_2.append(t_array_R_aux)
                        t_L_2.append(t_array_L_aux)
                        f.append(f_aux)
                        d_array.append(d_array_aux)
                        t_array.append(t_array_aux)
                        torque_vector_array.append(torque_vector[:,0:3][0])
                        

                        enclosing_points.append(np.vstack((np.asarray(data_set_R['{}'.format(obj)]['{}'.format(i+1)]['enclosing_points']), np.asarray(data_set_L['{}'.format(obj)]['{}'.format(j+1)]['enclosing_points']))))
                        grasp_R_poses.append(np.asarray(data_set_R['{}'.format(obj)]['{}'.format(i+1)]['pose']))
                        grasp_L_poses.append(np.asarray(data_set_L['{}'.format(obj)]['{}'.format(j+1)]['pose']))

                        finger_joints_R.append(np.asarray(data_set_R['{}'.format(obj)]['{}'.format(i+1)]['finger_joints']))
                        finger_joints_L.append(np.asarray(data_set_L['{}'.format(obj)]['{}'.format(j+1)]['finger_joints']))

                        contact_points_finger_R.append(np.asarray(data_set_R['{}'.format(obj)]['{}'.format(i+1)]['contacts_finger']))
                        contact_points_finger_L.append(np.asarray(data_set_L['{}'.format(obj)]['{}'.format(j+1)]['contacts_finger']))
                        
                        score_R.append(np.asarray(shape_complementarity_R))
                        score_L.append(np.asarray(shape_complementarity_L))

                        aux_num_enclosing_pts.append(len(enclosing_points[grasp_counter]))
                        aux_num_contact_pts.append(len(valid_contact_points[0]))

                        obj_name.append(obj)
                        hands_num.append(np.asarray([i+1, j+1]))
                        grasp_final_ident.append(grasp_counter)

                        # Visualization of grasp performance
                        # grasps_visualization(data_set_R['{}'.format(obj)]['{}'.format(i+1)], data_set_L['{}'.format(obj)]['{}'.format(j+1)], os.path.join(obj_folder,'{}'.format(obj)), obj_T, valid_contact_points[0], valid_contact_normals[0], torque_vector)

                        # dual_grasp_vis_test(data_set_R['{}'.format(obj)]['{}'.format(i+1)], data_set_L['{}'.format(obj)]['{}'.format(j+1)], os.path.join(obj_folder,'{}'.format(obj)), env)

                        grasp_counter += 1
                    
                    else:
                        # print("Grasps distance is invalid.")
                        statistics['num_grasps_unvalid_dist'] += 1
                        # grasps_visualization(data_set_R['{}'.format(obj)]['{}'.format(i+1)], data_set_L['{}'.format(obj)]['{}'.format(j+1)], os.path.join(obj_folder,'{}'.format(obj)), obj_T)

            # statistics
            statistics['grasp_counter'].append(grasp_counter)
            statistics['shape_complementarity'].append(shape_metric_stat)
            statistics['force_closure'].append(f)
            statistics['dexterity'].append(d_array)
            statistics['torque_optimization'].append(t_array)
            statistics['num_enclosing_pts'].append(aux_num_enclosing_pts)
            statistics['num_contact_pts'].append(aux_num_contact_pts)

            # produce .h5 file related to the object k
            write_final_data_set(enclosing_points, grasp_R_poses, grasp_L_poses, finger_joints_R, finger_joints_L, score_R, score_L, d_array, f, t_array, obj, grasp_counter, aux_valid_contact_points, aux_valid_contact_normals, torque_vector_array)

            # print information related to the object k
            # print_final_data_set_obj(enclosing_points, grasp_R_poses, grasp_L_poses, d_array, f, t_array, obj, grasp_counter)

        else:
            statistics['num_objects_not_used'] += 1

    # produce a file with the grasp identification and metrics. It can also write the number of contact points by finger
    regist_contacts_finger(obj_name, hands_num, grasp_final_ident, contact_points_finger_R, contact_points_finger_L, score_R_2, score_L_2, f_2, d_2, t_2, num_valid_hand_contacts_R, num_valid_hand_contacts_L, d_R_2, d_L_2, t_R_2, t_L_2)

    # produce document with final dataset statistic information
    write_statistic_doc(statistics, histogram=True)




    
   