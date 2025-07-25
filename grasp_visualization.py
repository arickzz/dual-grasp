# from dexnet.grasping import PointGraspMetrics3D
import os
from autolab_core import YamlConfig
import numpy as np
import open3d as o3d
from geometric_object_grasper.gog.robot_hand import RH8DLHand
from geometric_object_grasper.gog.robot_hand import RH8DRHand
# from geometric_object_grasper.gog.environment_original import Environment
from geometric_object_grasper.gog.utils.orientation import Quaternion, rot_z
# from geometric_object_grasper.gog.grasp_optimizer import ShapeComplementarityMetric
from da2_quality_metrics_calculation import compute_object_frame
import h5py
import yaml
#global_path='/home/vislab/'
global_path='/home/plinio/'

def read_h5_file(obj_name, grasp_final_ident):

    # Open the HDF5 file
    with h5py.File(global_path+'DA2/final_data_set/{}.h5'.format(obj_name), 'r') as file:
        
        valid_contact_points = file['grasps']['valid_grasp_points'][int(grasp_final_ident)]
        valid_contact_normals = file['grasps']['valid_grasp_normals'][int(grasp_final_ident)]
        grasp_pose = file['grasps']['transforms'][int(grasp_final_ident)]
        finger_joints = file['grasps']['finger_joints'][int(grasp_final_ident)]
        shape_score = file['grasps']['qualities/shape_score'][int(grasp_final_ident)]
        force_closure = file['grasps']['qualities/Force_closure'][int(grasp_final_ident)]
        dexterity = file['grasps']['qualities/Dexterity'][int(grasp_final_ident)]
        torque_optimization = file['grasps']['qualities/Torque_optimization'][int(grasp_final_ident)]
        torque_vector = file['grasps']['Torque_vector'][int(grasp_final_ident)]

    grasp = {'grasp_R': {'pose': grasp_pose[0], 'finger_joints': finger_joints[0]},
             'grasp_L': {'pose': grasp_pose[1], 'finger_joints': finger_joints[1]},
             'metrics': {'shape_score': shape_score, 'force_closure': force_closure,
                         'dexterity': dexterity, 'torque_optimization': torque_optimization},
             'valid_contacts': {'points': valid_contact_points, 'normals': valid_contact_normals},
             'torque_vector': {'vector': torque_vector}}        

    return grasp

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


def grasps_visualization(grasp_R, grasp_L, valid_contacts, torque_vector, point_cloud, obj_T):

    robot_R = RH8DRHand(urdf_file=global_path+'DA2/geometric_object_grasper/assets/robot_hands/RH8DR/RH8DR.urdf', contacts_file_name='contacts_right_hand_10')
    robot_L = RH8DLHand(urdf_file=global_path+'DA2/geometric_object_grasper/assets/robot_hands/RH8DL/RH8DL.urdf', contacts_file_name='contacts_left_hand')

    mesh = o3d.io.read_triangle_mesh(point_cloud)
    mesh.paint_uniform_color([0.0, 0.5, 0.5])
    mesh.transform(obj_T)
    mesh = rotate_point_cloud(mesh)

    robot_R.update_links(joint_values= grasp_R['finger_joints'], palm_pose=grasp_R['pose'])

    robot_L.update_links(joint_values= grasp_L['finger_joints'], palm_pose=grasp_L['pose'])

    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.07)

    hand_R_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    hand_R_frame = hand_R_frame.transform(np.array(grasp_R['pose']))
    
    hand_L_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
    hand_L_frame = hand_L_frame.transform(np.array(grasp_L['pose']))

    ref_L_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
    ref_L_frame = ref_L_frame.transform(np.matmul(np.array(grasp_L['pose']), np.linalg.inv(np.array(robot_L.ref_frame))))   

    valid_contact_pts = o3d.geometry.PointCloud()
    valid_contact_pts.points = o3d.utility.Vector3dVector(valid_contacts['points'])
    valid_contact_pts.normals = o3d.utility.Vector3dVector(valid_contacts['normals'])
    valid_contact_pts.normalize_normals()
    valid_contact_pts.paint_uniform_color([0.0, 0.5, 0.0])


    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.003,
        cone_radius=0.01,
        cylinder_height=0.1,
        cone_height=0.02
    )
    arrow.paint_uniform_color([1, 1, 0.0])

    z_axis = np.array([0, 0, 1])
    v = np.cross(z_axis, torque_vector['vector'])
    c = np.dot(z_axis, torque_vector['vector'])
    if np.linalg.norm(v) < 1e-6:  # If vectors are aligned, no rotation is needed
        R = np.eye(3)
    else:
        v_skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + v_skew + v_skew @ v_skew * ((1 - c) / (np.linalg.norm(v) ** 2))

    # Apply the rotation and translate the arrow to the origin
    arrow.rotate(R, center=(0, 0, 0))
    arrow.translate([0,0,0])


    visuals= robot_L.get_visual_map(boxes=False, frames=False)
    # visuals= robot_R.get_visual_map(boxes=False, frames=False)
    visuals.extend(robot_R.get_visual_map(boxes=False, frames=False))
    
    visuals.append(base_frame)
    #Uncomment to visualize the object
    #visuals.append(mesh)
    visuals.append(arrow)
    
    # visuals.append(valid_contact_pts)

    # Visualize multiple geometries
    o3d.visualization.draw_geometries(visuals, point_show_normal=True)
    # o3d.visualization.draw_geometries([visuals, valid_contact_pts], point_show_normal=True)



if __name__ == "__main__":

    obj_folder = global_path+"DA2/geometric_object_grasper/assets/objects/unseen"

    while(1):

        obj_name = input("\nEnter the object identification: ")

        grasp_final_ident = input("Enter grasp final identification: ")

        grasp = read_h5_file(obj_name, grasp_final_ident)
        print(os.path.join(obj_folder,'{}.obj'.format(obj_name)))

        obj_T = compute_object_frame(os.path.join(obj_folder,'{}.obj'.format(obj_name)))

        # Print grasp metrics
        print("\nShape Complementarity Score: R:{}, L:{}\nForce Closure: {}\nDexterity: {}\nTorque Optimization: {}\n".format(grasp['metrics']['shape_score'][0],grasp['metrics']['shape_score'][1],
                                                                                                                       grasp['metrics']['force_closure'][0],
                                                                                                                       grasp['metrics']['dexterity'][0],
                                                                                                                       grasp['metrics']['torque_optimization'][0]))
        
       

        grasps_visualization(grasp['grasp_R'], grasp['grasp_L'], grasp['valid_contacts'], grasp['torque_vector'], os.path.join(obj_folder,'{}.obj'.format(obj_name)), obj_T)

        

