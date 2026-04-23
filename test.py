import sys
print(sys.executable)

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer

import os
file_path = os.path.abspath(".")
# get the path of the dualarm robot
mesh_path = file_path + '/model/bifrank_robot.urdf'
urdf = os.path.join(file_path, "model", "bifrank_robot.urdf")

model = pin.buildModelFromUrdf(urdf)
data = model.createData()

print("DoF:", model.nq, model.nv)

# geometry model
geom_model = pin.buildGeomFromUrdf(model, urdf, pin.GeometryType.COLLISION, package_dirs=[file_path]) # or pin.GeometryType.COLLISION ??
geom_model.addAllCollisionPairs()
geom_data = geom_model.createData()

print(f"Number of geometric objects: {geom_model.ngeoms}")
print(f"Number of collision pairs: {len(geom_model.collisionPairs)}")

# remove adjacent collision pairs
srdf_path = os.path.join(file_path, "model", "bifrank_robot.srdf")
pin.removeCollisionPairs(model, geom_model, srdf_path)
geom_data = geom_model.createData()
print(f"Number of collision pairs after SRDF filtering: {len(geom_model.collisionPairs)}")

file_out = open("collision_pairs.txt", "w")
file_out.write("Collision Pairs (after SRDF filtering):\n")
for i, pair in enumerate(geom_model.collisionPairs):
    obj1 = geom_model.geometryObjects[pair.first].name
    obj2 = geom_model.geometryObjects[pair.second].name
    file_out.write(f"Pair {i}: {obj1} <-> {obj2}\n")
file_out.close()

import mujoco
import mujoco.viewer

import numpy as np

mjcf_path = os.path.join(file_path, "model", "scene.xml")
model_mj = mujoco.MjModel.from_xml_path(mjcf_path)
data_mj = mujoco.MjData(model_mj)

mujoco.mj_forward(model_mj, data_mj)
print("MuJoCo nq =", model_mj.nq, " nv =", model_mj.nv, " nu =", model_mj.nu)
print("MuJoCo qpos =", data_mj.qpos)


viewer = mujoco.viewer.launch_passive(model_mj, data_mj)

# pinocchio model states
joint_indices = np.arange(0, model.nq)

joint_initial_pos = [0.0, 0.0, -0.7854, 0.0, -2.35621, 0.0, 1.5708, 0.785398, 
                          0.0, -0.7854, 0.0, -2.35621, 0.0, 1.5708, 0.785398]
data_mj.qpos[joint_indices] = joint_initial_pos
mujoco.mj_forward(model_mj, data_mj)
print("MuJoCo qpos after setting initial pos =", data_mj.qpos)

# The number of pin_joints should match the number of mujoco joints
assert model.nq == model_mj.nq, "Mismatch in DoF between Pinocchio and MuJoCo models"
def get_pin_state_from_mujoco():
    q  = data_mj.qpos[joint_indices].copy()
    dq = data_mj.qvel[joint_indices].copy()
    return q, dq

# print("--- Joint List ---")
# for i in range(model.nq +1):
#     print(f"Joint ID {i}: {model.names[i]}")

# print("--- Frame List ---")
# for i, frame in enumerate(model.frames):
#     print(f"Frame {i}: {frame}")

# while True:
#     mujoco.mj_step(model_mj, data_mj)
#     viewer.sync()

## JOINT SPACE IMPENDANCE CONTROL 
# Computing Dynamics in Pinocchio
def compute_pin_dynamics(q, dq):
    # Mass Matrix
    M = pin.crba(model, data, q)
    # print("Pinocchio Mass Matrix:\n", M)
    # numerical symmetrization
    M = 0.5 * (M + M.T)
    # print("Symmetrized Mass Matrix:\n", M)

    # Non-linear effects
    nle = pin.rnea(model, data, q, dq, np.zeros_like(dq))
    # print("Pinocchio Non-linear effects:\n", nle)

    return M, nle

# impedance control law

# Gains Kp and Kd
Kp = np.diag([100.0] * model.nq)
Kd = np.diag([20.0] * model.nq)

def impedance_control(q, dq, q_des, dq_des, ddq_des, M, nle):
    if ddq_des is None:
        ddq_des = np.zeros_like(q)

    # error terms
    e = q_des - q
    de = dq_des - dq

    # Pinocchio impedance control law --> redundant: already arguments of function
    # M, nle = compute_pin_dynamics(q, dq) 

    # Joint torques
    tau = M @ (ddq_des + Kd @ de + Kp @ e) + nle

    return tau

# Reference trajectories
mujoco.mj_forward(model_mj, data_mj)
q0, dq0 = get_pin_state_from_mujoco()
print("Initial Pinocchio q0:", q0)
print("Initial Pinocchio dq0:", dq0)

q_goal = q0.copy()
q_goal[1] += 0.5  # Move first joint by 0.5 rad
q_goal[8] -= 0.5  # Move seventh joint by -0.5 rad

T_move = 4.0  # seconds

# TODO: change to have collision
def desired_trajectory(t):
    if t >= T_move:
        return q_goal, np.zeros_like(q_goal), np.zeros_like(q_goal)
    else:
        s = t / T_move
        q_des = (1 - s) * q0 + s * q_goal
        dq_des = (q_goal - q0) / T_move
        ddq_des = np.zeros_like(q0)
    return q_des, dq_des, ddq_des

def repulsive_force_quadratic(d, d_start, F_max):
    if d < d_start:
        # formula (15)
        return ((F_max / (d_start **2)) * (d - d_start)**2)
    else:
        return 0

def repulsive_force_linear(d, d_start, F_max):
    if d < d_start:
        # easier to tune but less smooth
        return (F_max * (1.0 - d / d_start))
    else:
        return 0


DT = model_mj.opt.timestep
print("Simulation timestep (DT):", DT)
sim_time = 2.0
steps = int(sim_time / DT)

log_t = []
log_q = []
log_dq = []
log_left_ee_pos = []
log_left_ee_vel = []

log_right_ee_pos = []
log_right_ee_vel = []

t = 0.0

log_coll_dist = []

# viz = MeshcatVisualizer(model, geom_model, geom_model) # (model, collision_model, visual_model)
# viz.initViewer(open=True)

# viz.loadViewerModel("pinocchio")

# num_pairs = len(geom_model.collisionPairs)
# p1_log = [np.zeros(3) for _ in range(num_pairs)] # closest point on first object (lewis)
# p2_log = [np.zeros(3) for _ in range(num_pairs)] # closest point on second object (richard)
# J1_log = [np.zeros((6, model.nv)) for _ in range(num_pairs)]# list of Jacobians at point1
# J2_log = [np.zeros((6, model.nv)) for _ in range(num_pairs)] # list of Jacobians at point2
collision_log = []  # list of dicts, one per triggered pair per timestep


d_start = 0.15 # 15 cm

for k in range(steps):

    # Get state from Mujoco
    q, dq = get_pin_state_from_mujoco()

    # Forward kinematics
    pin.forwardKinematics(model, data, q, dq)
    pin.updateFramePlacements(model, data)
    pin.updateGeometryPlacements(model, data, geom_model, geom_data)

    # Jacobians
    pin.computeJointJacobians(model, data, q)
    
    pin.computeDistances(geom_model, geom_data)
    # min_dist = min([res.min_distance for res in geom_data.distanceResults])
    # log_min_distances.append(min_dist)

    tau_collision = np.zeros(model.nv) # restart at each step

    for i, res in enumerate(geom_data.distanceResults):
        # collision check with minimum distance --> compute jacobians only of that ones.
        dist = res.min_distance
        if dist < d_start:
            #log_coll_dist.append(dist)
            pair = geom_model.collisionPairs[i]
    
            # collision points
            point1 = res.getNearestPoint1() # closest point on first object
            point2 = res.getNearestPoint2() # closest point on second object
            #_log[i] = point2
            # parent joints
            J1_id = geom_model.geometryObjects[pair.first].parentJoint
            J2_id = geom_model.geometryObjects[pair.second].parentJoint

            # Jacobian at point 1
            J1_origin = pin.getJointJacobian(model, data, J1_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            # vector from joint origin to point1
            rel_dist1 = point1 - data.oMi[J1_id].translation
            Jp1 = J1_origin.copy()
            # v_p = v_o + w x r_op --> J_p = J_o + skew[w] * r_op --> J_p = J_o + cross(r_op, J_o[3:6, :])
            # angular part (bottom 3 rows) is unchanged, linear part (top 3 rows) gets cross product with rel_dist1
            Jp1[0:3, :] += np.cross(rel_dist1, J1_origin[3:6, :], axis=0)
            #J1_log[i] = Jp1

            # Jacobian at point 2
            J2_origin = pin.getJointJacobian(model, data, J2_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            # vector from joint origin to point2
            rel_dist2 = point2 - data.oMi[J2_id].translation
            Jp2 = J2_origin.copy()
            # angular part (bottom 3 rows) is unchanged, linear part (top 3 rows) gets cross product with rel_dist2
            Jp2[0:3, :] += np.cross(rel_dist2, J2_origin[3:6, :], axis=0)
            #J2_log[i] = Jp2

            collision_log.append({
                't': t,
                'pair_idx': i,
                'obj1': geom_model.geometryObjects[pair.first].name,
                'obj2': geom_model.geometryObjects[pair.second].name,
                'dist': dist,
                'point1': point1.copy(),
                'point2': point2.copy(),
                'J1': Jp1.copy(),
                'J2': Jp2.copy(),
            })

            # apply avoidance torque
            # force direction: from p2 to p1 (normal)
            normal = res.normal
            force_mag = repulsive_force_linear(dist, d_start, F_max = 3) # TODO: tune F_max, try linear and quadratic
            # force_mag = repulsive_force_quadratic(dist, d_start, F_max = 3) # TODO: tune F_max
            force_vec = force_mag * normal
            # relative jacobian (J1 - J2) but only linear part (top 3 rows) since we want a force, not a torque
            J_rel = Jp1[0:3, :] - Jp2[0:3, :]
            tau_collision += J_rel.T @ force_vec

            # force visualization
            with viewer.lock():
                viewer.user_scn.ngeom = 0  # clear previous arrows
                # draw arrow from p1 in direction of force
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    np.zeros(3),   # size filled below
                    np.zeros(3),   # pos filled below
                    np.zeros(9),   # mat filled below
                    np.array([1.0, 0.0, 0.0, 1.0])  # red RGBA
                )
                arrow_end = point1 + force_vec # * 0.05  # scale for visibility
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    0.01,   # width
                    point1,
                    arrow_end
                )
                viewer.user_scn.ngeom += 1


    # # Identify which specific pair is causing the fixed distance
    # for i, res in enumerate(geom_data.distanceResults):
    #     if abs(res.min_distance - 0.123) < 0.0001:
    #         pair = geom_model.collisionPairs[i]
    #         obj1 = geom_model.geometryObjects[pair.first].name
    #         obj2 = geom_model.geometryObjects[pair.second].name
    #         print(f"TRAP DETECTED: {obj1} and {obj2} are stuck at {res.min_distance}m")
    
    # Desired
    q_des, dq_des, ddq_des = desired_trajectory(t)

    # get dynamics
    M, nle = compute_pin_dynamics(q, dq)

    # Compute torque from Pinocchio impedance
    tau_impedance = impedance_control(q, dq, q_des, dq_des, ddq_des, M, nle)

    tau_tot = tau_impedance #+ tau_collision

    # Apply to Mujoco
    data_mj.ctrl[:] = tau_tot

    # Step simulation
    mujoco.mj_step(model_mj, data_mj)
    viewer.sync()

    # viz.display(q)
    
    # Log
    log_t.append(t)
    log_q.append(q.copy())
    log_dq.append(dq.copy())


    t += DT


log_q = np.array(log_q)
log_t = np.array(log_t)

print(f"Final Joint Positions:\n{log_q[-1]}") # last row
print(f"Desired Joint Positions:\n{q_goal}")

# log_left_ee_pos = np.array(log_links_frames['lewis_fr3_link8'])
# log_right_ee_pos = np.array(log_links_frames['richard_fr3_link8'])
# print(f"Final Left EE Position:\n{log_left_ee_pos[-1]}")
# print(f"Final Right EE Position:\n{log_right_ee_pos[-1]}")

print(f" Number of collision pairs handled: {len(collision_log)}")

file_out = open("jacobians.txt", "w")
# file_out.write("Jacobian at point1 for each collision pair:\n")
# for i in range(num_pairs):
#     file_out.write(f"Collision Pair {i}:\n")
#     file_out.write(f"Point 1: {p1_log[i]}\n")
#     file_out.write(f"Jacobian at Point 1 (J1):\n{J1_log[i]}\n\n")
#     file_out.write(f"Point 2: {p2_log[i]}\n")
#     file_out.write(f"Jacobian at Point 2 (J2):\n{J2_log[i]}\n\n")
for entry in collision_log:
    file_out.write(f"t={entry['t']:.4f}s | pair {entry['pair_idx']}: "
                   f"{entry['obj1']} <-> {entry['obj2']} | dist={entry['dist']:.4f}m\n")
    file_out.write(f"  Point1: {entry['point1']}\n")
    file_out.write(f"  Point2: {entry['point2']}\n\n")
    # file_out.write(f"  J1:\n{entry['J1']}\n")
    # file_out.write(f"  J2:\n{entry['J2']}\n\n")
file_out.close()