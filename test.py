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

# ---------------------------------
# JOINT PD CONTROL
# ---------------------------------

# Kp_pd = np.array([30.0,                                             # base joint
#                   200.0, 200.0, 200.0, 200.0, 100.0, 50.0, 30.0,    # lewis
#                   200.0, 200.0, 200.0, 200.0, 100.0, 50.0, 30.0])   # richard                                          
# Kd_pd = np.array([3.0,
#                   20.0, 20.0, 20.0, 20.0, 10.0, 5.0, 3.0,
#                   20.0, 20.0, 20.0, 20.0, 10.0, 5.0, 3.0])
# Gains Kp and Kd
Kp_pd = np.array([100.0] * model.nq)
Kd_pd = np.array([20.0] * model.nq)

def pd_control(q, dq, q_des, dq_des):
    """
    Simple joint-space PD:
        tau = Kp * (q_des - q) + Kd * (dq_des - dq)
    No mass matrix, no NLE. Stable and easy to tune.
    """
    e  = q_des - q
    de = dq_des - dq
    return Kp_pd * e + Kd_pd * de

# -----------------------------------
# JOINT SPACE IMPENDANCE CONTROL 
# -----------------------------------
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

# -----------------------------------
# Reference trajectories
# -----------------------------------
mujoco.mj_forward(model_mj, data_mj)
q0, dq0 = get_pin_state_from_mujoco()
print("Initial Pinocchio q0:", q0)
print("Initial Pinocchio dq0:", dq0)

q_goal = q0.copy()
q_goal[1] += 0.5  # Move first joint by 0.5 rad (joint1 of lewis)
q_goal[8] -= 0.5  # Move seventh joint by -0.5 rad (joint1 of richard)

T_move = 4.0  # seconds

def desired_trajectory_1(t):
    if t >= T_move:
        return q_goal, np.zeros_like(q_goal), np.zeros_like(q_goal)
    else:
        s = t / T_move
        q_des = (1 - s) * q0 + s * q_goal
        dq_des = (q_goal - q0) / T_move
        ddq_des = np.zeros_like(q0)
    return q_des, dq_des, ddq_des

# -----------------------------------
# TRAJECTORY WITH COLLISION AVOIDANCE
# -----------------------------------
# Open-close, 2 cycles.
q0, _ = get_pin_state_from_mujoco()

# Approach configuration: joint2 of each arm swings inward
APPROACH_DELTA = 0.9   # radians - TODO: tune this to how close you want them (0.9 collision)

q_approach = q0.copy()
q_approach[2] += APPROACH_DELTA     # lewis joint2
q_approach[3] -= APPROACH_DELTA     # lewis joint3 
q_approach[9] += APPROACH_DELTA     # richard joint2 
q_approach[10] += APPROACH_DELTA    # richard joint3 

T_phase = 3.0   # seconds per phase (approach or retract)

def desired_trajectory(t):
    """
    Periodic open-close:
      0 -> T_phase       : approach (home -> close)
      T_phase -> 2*T_phase : retract (close -> home)
      then repeats
    """
    period = 2 * T_phase
    t_mod = t % period
    s = t_mod / T_phase if t_mod < T_phase else 2.0 - t_mod / T_phase
    # smooth with cosine interpolation
    alpha = 0.5 * (1 - np.cos(np.pi * s))  # 0->1->0 smoothly
    q_des  = (1 - alpha) * q0 + alpha * q_approach
    # velocity: d/dt of alpha * (q_approach - q0)
    dalpha_dt = 0.5 * np.pi / T_phase * np.sin(np.pi * s)
    if t_mod >= T_phase:
        dalpha_dt = -dalpha_dt
    dq_des = dalpha_dt * (q_approach - q0)
    ddq_des = np.zeros_like(q0)
    return q_des, dq_des, ddq_des

# -----------------------------------
# REPULSIVE FORCE CALCULATIONS
# -----------------------------------
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

def get_link_name(geom_name):
    """
    Pinocchio names geometry objects as  '{link_name}_{sphere_idx}'.
    Strip the trailing '_N' to get the parent link name.
    E.g.  'lewis_fr3_link3_2'  →  'lewis_fr3_link3'
    """
    parts = geom_name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return geom_name  # fallback

# ─────────────────────────────────────────────────────────
# Post-process collision pairs - one per (link_A, link_B)
#
# After gathering all active pairs we keep only the one with
# minimum distance for each unique unordered link pair.
# ─────────────────────────────────────────────────────────
def filter_collision_pairs_by_link(raw_pairs):
    # raw_pairs : list of dicts with keys:
    #     dist, point1, point2, J1_id, J2_id, normal, obj1, obj2, pair_idx
    # Returns: filtered list — one entry per unique (link1, link2) pair,
    #          the one with minimum dist.
    best = {}  # key: frozenset({link1, link2}) -> dict entry
    for entry in raw_pairs:
        l1 = get_link_name(entry['obj1'])
        l2 = get_link_name(entry['obj2'])
        key = frozenset({l1, l2})
        if key not in best or entry['dist'] < best[key]['dist']:
            best[key] = entry
    return list(best.values())

DT = model_mj.opt.timestep
print("Simulation timestep (DT):", DT)
sim_time = 12.0 # period = 6
steps = int(sim_time / DT)

log_t = []
log_q = []
log_dq = []
log_left_ee_pos = []
log_left_ee_vel = []

log_right_ee_pos = []
log_right_ee_vel = []

t = 0.0

# viz = MeshcatVisualizer(model, geom_model, geom_model) # (model, collision_model, visual_model)
# viz.initViewer(open=True)

# viz.loadViewerModel("pinocchio")

collision_log = []  # list of dicts, one per triggered pair per timestep
log_tau = []

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

    tau_collision = np.zeros(model.nv) # restart at each step
    log_tau_coll = []

    viewer.user_scn.ngeom = 0  # clear previous arrows

    raw_active_pairs = []  # collect all active pairs before filtering
    # loop through ALL active collision pairs
    for i, res in enumerate(geom_data.distanceResults):
        # collision check with minimum distance --> compute jacobians only of that ones.
        dist = res.min_distance
        if dist < d_start:
            pair = geom_model.collisionPairs[i]
            # collision points
            point1 = res.getNearestPoint1() # closest point on first object
            point2 = res.getNearestPoint2() # closest point on second object
            # parent joints
            J1_id = geom_model.geometryObjects[pair.first].parentJoint
            J2_id = geom_model.geometryObjects[pair.second].parentJoint

            raw_active_pairs.append({
                't': t,
                'pair_idx': i,
                'obj1': geom_model.geometryObjects[pair.first].name,
                'obj2': geom_model.geometryObjects[pair.second].name,
                'dist': dist,
                'point1': point1.copy(),
                'point2': point2.copy(),
                'J1_id': J1_id,
                'J2_id': J2_id,
                'normal': res.normal.copy(),
            })
    active_pairs = filter_collision_pairs_by_link(raw_active_pairs)
    
    for entry in active_pairs:
        # Jacobian at point 1
        J1_origin = pin.getJointJacobian(model, data, entry['J1_id'], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        # vector from joint origin to point1
        rel_dist1 = entry['point1'] - data.oMi[entry['J1_id']].translation
        Jp1 = J1_origin.copy()
        # v_p = v_o + w x r_op --> J_p = J_o + skew[w] * r_op --> J_p = J_o + cross(r_op, J_o[3:6, :])
        # angular part (bottom 3 rows) is unchanged, linear part (top 3 rows) gets cross product with rel_dist1
        Jp1[0:3, :] += np.cross(rel_dist1, J1_origin[3:6, :], axis=0)

        # Jacobian at point 2
        J2_origin = pin.getJointJacobian(model, data, entry['J2_id'], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        # vector from joint origin to point2
        rel_dist2 = entry['point2'] - data.oMi[entry['J2_id']].translation
        Jp2 = J2_origin.copy()
        # angular part (bottom 3 rows) is unchanged, linear part (top 3 rows) gets cross product with rel_dist2
        Jp2[0:3, :] += np.cross(rel_dist2, J2_origin[3:6, :], axis=0)
        
        entry['J1'] = Jp1
        entry['J2'] = Jp2
        collision_log.append(entry)

        # apply avoidance torque
        # force direction: from p1 to p2 (- normal)
        normal = - entry['normal']
        # force_mag = repulsive_force_linear(entry['dist'], d_start, F_max = 3) # TODO: tune F_max, try linear and quadratic
        force_mag = repulsive_force_quadratic(entry['dist'], d_start, F_max = 3) # TODO: tune F_max
        force_vec = force_mag * normal
        # relative jacobian (J1 - J2) but only linear part (top 3 rows) since we want a force, not a torque
        J_rel = Jp1[0:3, :] - Jp2[0:3, :]
        tau_collision += J_rel.T @ force_vec
        log_tau_coll.append(tau_collision.copy())

        # force visualization
        with viewer.lock():
            # draw arrow from p1 in direction of force
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.zeros(3),   # size filled below
                np.zeros(3),   # pos filled below
                np.zeros(9),   # mat filled below
                np.array([1.0, 0.0, 0.0, 1.0])  # red RGBA
            )
            arrow_end = entry['point1'] + force_vec 
            mujoco.mjv_connector(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_ARROW,
                0.01,   # width
                entry['point1'],
                arrow_end
            )
            viewer.user_scn.ngeom += 1

    # Desired
    # q_des, dq_des, ddq_des = desired_trajectory_1(t)
    q_des, dq_des, ddq_des = desired_trajectory(t)   # with collisions

    # Joint PD
    # tau_pd = pd_control(q, dq, q_des, dq_des)

    # get dynamics
    M, nle = compute_pin_dynamics(q, dq)

    # Compute torque from Pinocchio impedance
    tau_impedance = impedance_control(q, dq, q_des, dq_des, ddq_des, M, nle)

    tau_tot = tau_impedance + tau_collision # TODO: chage pd with impedance
    log_tau.append({
        't': t,
        'collision': log_tau_coll,
        'impedance': tau_impedance.copy(),
        'total': tau_tot.copy(),})

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

# print(f"Final Joint Positions:\n{log_q[-1]}") # last row
# print(f"Desired Joint Positions:\n{q_goal}")

# log_left_ee_pos = np.array(log_links_frames['lewis_fr3_link8'])
# log_right_ee_pos = np.array(log_links_frames['richard_fr3_link8'])
# print(f"Final Left EE Position:\n{log_left_ee_pos[-1]}")
# print(f"Final Right EE Position:\n{log_right_ee_pos[-1]}")

print(f" Number of collision pairs handled: {len(collision_log)}")

file_out = open("collisions.txt", "w")
for entry in collision_log:
    file_out.write(f"t={entry['t']:.4f}s | pair {entry['pair_idx']}: "
                   f"{entry['obj1']} <-> {entry['obj2']} | dist={entry['dist']:.4f}m\n")
    file_out.write(f"  Point1: {entry['point1']}\n")
    file_out.write(f"  Point2: {entry['point2']}\n\n")
    # file_out.write(f"  J1:\n{entry['J1']}\n")
    # file_out.write(f"  J2:\n{entry['J2']}\n\n")
file_out.close()

file_out = open("torques.txt", "w")
for entry in log_tau:
    file_out.write(f"t={entry['t']:.4f}s | collision tau: {entry['collision']} \n impedance tau: {entry['impedance']} \n total tau: {entry['total']}\n\n")
file_out.close()