import os
file_path = os.path.abspath(".")
print(file_path)

import sys
print(sys.executable)

import pinocchio as pin
import mujoco
import mujoco.viewer

import numpy as np

# get current file
urdf = file_path + "/model/bifrank_robot.urdf"

urdf = os.path.join(file_path, "model", "bifrank_robot.urdf")

model_full = pin.buildModelFromUrdf(urdf)
data_full = model_full.createData()
q0 = pin.neutral(model_full)

model = pin.buildReducedModel(model_full, [], q0)
data = model.createData()

## Get end-effector numbers
# The frame id of the end-effector
end_effector_link_name = ["lewis_fr3_link7", "richard_fr3_link7"]

#TODO: using for Loop to check and get the frame id of end-effectors and store them in a list
#[ ]: check the end-effector names in the urdf file
#[ ]: get the frame ids and store them in a list 
left_ee_frame_id = model.getFrameId(end_effector_link_name[0])
right_ee_frame_id = model.getFrameId(end_effector_link_name[1])
print("left_ee_frame_id: ", left_ee_frame_id, "right_ee_frame_id: ", right_ee_frame_id)
print("DoF:", model.nq, model.nv)



mjcf_path = os.path.join(file_path, "model", "scene.xml")

model_mj = mujoco.MjModel.from_xml_path(mjcf_path)
data_mj = mujoco.MjData(model_mj)

mujoco.mj_forward(model_mj, data_mj)
print("MuJoCo nq =", model_mj.nq, " nv =", model_mj.nv, " nu =", model_mj.nu)

viewer = mujoco.viewer.launch_passive(model_mj, data_mj)

t = 0.0
mujoco.mj_step(model_mj, data_mj)

# for k in range(100000000):
#     mujoco.mj_step(model_mj, data_mj)
#     viewer.sync()

# pinocchio model states
joint_indices = np.arange(0, model.nq)

joint_initial_pos = [0.0, 0.0, -0.7854, 0.0, -2.35621, 0.0, 1.5708, 0.785398, 
                          0.0, -0.7854, 0.0, -2.35621, 0.0, 1.5708, 0.785398]
data_mj.qpos[joint_indices] = joint_initial_pos
data_mj.qvel[joint_indices] = np.zeros_like(model_mj.nv)
mujoco.mj_forward(model_mj, data_mj)
mujoco.mj_step(model_mj, data_mj)
viewer.sync()

left_des_R = data.oMf[left_ee_frame_id].rotation.copy()
right_des_R = data.oMf[right_ee_frame_id].rotation.copy()

# print("Mujoco data_mj.qvel: ", data_mj.qvel)
# print("MuJoCo qpos after setting initial pos =", data_mj.qpos)

# The number of pin_joints should match the number of mujoco joints
assert model.nq == model_mj.nq, "Mismatch in DoF between Pinocchio and MuJoCo models"
def get_pin_state_from_mujoco():
    q  = data_mj.qpos[joint_indices].copy()
    dq = data_mj.qvel[joint_indices].copy()
    return q, dq

def rotation_error(R_des, R):
    """
    Compute orientation error e_o (3D rotation vector).
    R_des : desired rotation matrix (3x3)
    R     : current rotation matrix (3x3)
    """
    R_err = R_des.T @ R
    # Skew-symmetric part → vee operator
    e_o = 0.5 * pin.log3(R_err)   # log3 returns a 3D rotation vector
    return e_o

# pinocchio compute task space state 3D position and velocity
def compute_task_state(q, dq):
    pin.forwardKinematics(model, data, q, dq)
    pin.updateFramePlacements(model, data)
    
    left_ee_SE3 = data.oMf[left_ee_frame_id].copy()
    right_ee_SE3 = data.oMf[right_ee_frame_id].copy()

    # position 
    left_ee_pos = left_ee_SE3.translation.copy()
    right_ee_pos = right_ee_SE3.translation.copy()

    # Rotation matrix via quaternion (Quaternion.toRotationMatrix = quaternionToMatrix)
    left_ee_quater = pin.Quaternion(left_ee_SE3.rotation)
    right_ee_quater = pin.Quaternion(right_ee_SE3.rotation)
    # left_rot_e = rotation_error(left_des_R, left_ee_R)

    # left_rpy = pin.rpy.matrixToRpy(left_ee_SE3.rotation)
    # right_rpy = pin.rpy.matrixToRpy(right_ee_SE3.rotation)

    left_ee_vel = pin.getFrameVelocity(model, data, left_ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).copy()
    right_ee_vel = pin.getFrameVelocity(model, data, right_ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).copy()

    ee_pos = np.hstack([left_ee_pos, left_ee_quater, right_ee_pos, right_ee_quater ])
    ee_vel = np.hstack([left_ee_vel.linear, left_ee_vel.angular, right_ee_vel.linear, right_ee_vel.angular])

    print("left_ee_R: ", left_ee_SE3.rotation)
    print("right_ee_R: ", right_ee_SE3.rotation)

    print("left_ee_quater: ", left_ee_quater)
    print("right_ee_quater: ", right_ee_quater)

    return ee_pos, ee_vel

# pinocchio compute task space 3D Jacobian and its derivative
def compute_task_Jacobian(q, dq):
    pin.forwardKinematics(model, data, q, dq)
    pin.updateFramePlacements(model, data)

    # detect the end-effector number, and to compute the Jacobian
    # the left end-effector
    pin.computeFrameJacobian(model, data, q, left_ee_frame_id, 
                                                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    left_ee_Jacobian_6d = pin.getFrameJacobian(model, data, left_ee_frame_id, 
                                                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    left_ee_dotJacobian_6d = pin.frameJacobianTimeVariation(model, data, q, dq, left_ee_frame_id,    
                                   pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    
    # print("Jacobian_left_ee:\n", Jacobian_left_ee)

    # the right end-effector
    pin.computeFrameJacobian(model, data, q, right_ee_frame_id, 
                                                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    right_ee_Jacobian_6d = pin.getFrameJacobian(model, data, right_ee_frame_id, 
                                                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    
    right_ee_dotJacobian_6d = pin.frameJacobianTimeVariation(model, data, q, dq, right_ee_frame_id, 
                                                             pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    Ja_12d = np.vstack([left_ee_Jacobian_6d, right_ee_Jacobian_6d])  # linear velocity part
    dJa_12d = np.vstack([left_ee_dotJacobian_6d, right_ee_dotJacobian_6d])  # linear velocity part

    return Ja_12d, dJa_12d

# Test the Function of compute_task_Jacobian()
# -----------------------------------------#
# Pinocchio forwardKinematics requires numpy arrays (not Python lists)

# Computing Cartesian Space Dynamics in Pinocchio
def compute_Cartesian_space_dynamics(q, dq):
    pin.forwardKinematics(model, data, q, dq)
    pin.updateFramePlacements(model, data)

    # Joint-space mass matrix
    pin.crba(model, data, q)

    M = data.M
    M = 0.5 * (M + M.T)           # symmetrize numerically

    # Nonlinear effects h(q,dq) = C(q,dq) dq + g(q)
    h = pin.nonLinearEffects(model, data, q, dq)

    # Gravity separated if you want it explicitly
    g = pin.computeGeneralizedGravity(model, data, q)

    dns_C = pin.computeCoriolisMatrix(model, data, q, dq)

    # get end-effector Jacobian and its derivative
    Ja, Jadot = compute_task_Jacobian(q, dq)

    # Operational-space inertia Λ = (J M^{-1} Jᵀ)^{-1}
    MinvJt = np.linalg.solve(M, Ja.T)         # (nv×nv)\(nv×12) ⇒ 15×12

    eps = 1e-6
    A = Ja @ MinvJt
    A = 0.5 * (A + A.T)
    Lambda = np.linalg.inv(A + eps * np.eye(A.shape[0]))
    # Lambda = np.linalg.inv(Ja @ MinvJt)       # 12×12

    # Calculate \miu = Lambda( Ja @ M^{-1} @ C - dotJa)\dotq
    JM_inv_C = Ja @ np.linalg.solve(M, dns_C)   # m×nv
    mu = Lambda @ (JM_inv_C - Jadot) # @ dq # shape (m,)

    # Calculate the J_sharp
    J_sharp = np.linalg.solve(M, Ja.T) @ Lambda

    # print("M = ", M)
    # print("MinvJt: ", MinvJt)
    # print("Lambda: ", Lambda)
    # print("JM_inv_C: ", JM_inv_C)
    # print("mu: ", mu)
    # print("J_sharp: ", J_sharp)

    # # "Coriolis/centrifugal/bias" term in task space (Khatib style):
    # # \Gamma = Ja{-T} C Ja{-1} - Λ Jadot Ja{-1}
    # #        = (Ja^{-T} C - Λ Jadot) Ja^{-1}
    # invJa = np.linalg.pinv(Ja)          # (6×nv)\(nv×nv) ⇒ 6×nv
    # Gamma = (invJa.T @ dns_C - Lambda @ Jadot) @ invJa
    
    # # Gravity in task space:
    # p = Ja^{-T} g

    # print("Lambda's shape:\n", Lambda.shape)
    # print("Gamma's shape:\n", Gamma.shape)
    # print("F_g's shape:\n", F_g.shape)
    
    return Lambda, mu, J_sharp, g  #Lambda, mu, p

# Test the Function of compute_task_Jacobian()
# -----------------------------------------#
# q = np.random.rand(model.nq)
# dq = np.random.rand(model.nv)
# Lambda, Gamma, F_g = compute_Cartesian_space_dynamics(q, dq)
# -----------------------------------------# 

# Impedance control law

# Stiffness and Damping Matrices
# end-edffector of both arms 
# K_p = np.diag([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]) 
K_p_diag = 30.0
task_size = 12 # 6 DoF for left arm + 6 DoF for right arm
diag_vector = np.full(task_size, K_p_diag)
K_p = np.diag(diag_vector)

# end-edffector of both arms left{x,y,z}, right{x,y,z}
K_d_diag = 60.0
task_size = 12 # 6 DoF for left arm + 6 DoF for right arm
diag_vector = np.full(task_size, K_d_diag)
K_d = np.diag(diag_vector)

# Cartesian Impedance Control Law 
def cartesian_impedance_control(x, dot_x, x_des, dot_x_des, ddot_x_des, Lambda, mu, Ja_sharp):
    # error terms
    # TODO 

    e_pos = x_des - x # only position
    # if x_quat is here should change to SO3 and use rotation error
    e_rot = rotation_error(left_des_R, x[:3].reshape(3,3)) # only orientation
    de = dot_x_des - dot_x 

    # F = Lambda @ ddot_x_des + mu @ Ja_sharp @ dot_x_des + K_d @ de + K_p @ e    
    F = Lambda @ ddot_x_des + mu @ Ja_sharp @ dot_x_des + K_d @ de + K_p @ e    

    return F # not torque yet

# Reference trajectories
from numpy.ma import zeros

data_mj.qpos[joint_indices] = joint_initial_pos
data_mj.qvel[joint_indices] = np.zeros_like(model_mj.nv)

mujoco.mj_forward(model_mj, data_mj)
mujoco.mj_step(model_mj, data_mj)
viewer.sync()
q0, dq0 = get_pin_state_from_mujoco()

pin.forwardKinematics(model, data, q0, dq0)
pin.updateFramePlacements(model, data)
left_des_R = data.oMf[left_ee_frame_id].rotation.copy()
right_des_R = data.oMf[right_ee_frame_id].rotation.copy()
# print("left_des_R: ", left_des_R)
# print("right_des_R: ", right_des_R)
left_des_R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
right_des_R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


# print("Initial Pinocchio q0:", q0)
# print("Initial Pinocchio dq0:", dq0)

# get the initial end-effectors' position
ini_ee_pos, ini_ee_vel = compute_task_state(q0, dq0)
# print("Initial end-effector position:", ini_ee_pos)
# print("Initial end-effector velocity:", ini_ee_vel)

# desired end-effector position
left_offset_ee_pos = [0.0, 0.2, 0.0]
# 1. rpy row pitch yaw to calculate the desired rotation matrix 321 or  123
# 2. change the desired rotation matrix to quaternion
# 3. SLERP interpolation in quaternion space to get the desired rotation matrix at each time step
right_offset_ee_pos = [0.0, -0.2, 0.0]

offset = np.hstack([left_offset_ee_pos, left_offset_ee_rot, right_offset_ee_pos, right_offset_ee_rot]) # left and right end-effector
des_ee_pos = ini_ee_pos + offset
des_ee_vel = np.zeros_like(des_ee_pos)
des_ee_acc = np.zeros_like(des_ee_pos)
print("des_ee_pos: ", des_ee_pos)   


T_move = 4  # seconds

# In ros the desired_trajectory should get from a topic
# TODO: the postion and orienttation trajectory design
def desired_trajectory(t):
    if t >= T_move:
        return des_ee_pos, np.zeros_like(des_ee_vel), np.zeros_like(des_ee_acc)

    tau = t / T_move
    tau2 = tau * tau
    tau3 = tau2 * tau
    tau4 = tau3 * tau
    tau5 = tau4 * tau

    # position (left and right end-effector)
    x_des = ini_ee_pos + (des_ee_pos - ini_ee_pos) * (10*tau3 - 15*tau4 + 6*tau5)

    # velocity
    dot_x_des = (des_ee_pos - ini_ee_pos) * (30*tau2 - 60*tau3 + 30*tau4) / T_move

    # acceleration
    ddot_x_des = (des_ee_pos - ini_ee_pos) * (60*tau - 180*tau2 + 120*tau3) / (T_move*T_move)

    # TODO: SLERP quaternion interpolation for orientation trajectory
    # quter

    return x_des, dot_x_des, ddot_x_des

# control step
def control_step(t):
    # --- 1. read state from Mujoco and map to Pinocchio (Sensor) --- #
    q, dq = get_pin_state_from_mujoco()
        
    x, dot_x = compute_task_state(q, dq)
    
    # --- 2. Kinematics & Jacobians in Pinocchio --- #
    Ja, dotJa = compute_task_Jacobian(q, dq)

    # --- 3. Compute Lambda, Gamma, F_g --- #
    Lambda, mu, J_sharp, g= compute_Cartesian_space_dynamics(q, dq)

    # --- 4. Impedance torque --- #
    x_des, dot_x_des, ddot_x_des = desired_trajectory(t)
    F = cartesian_impedance_control(x, dot_x, x_des, dot_x_des, ddot_x_des, Lambda, mu, J_sharp)

    tau = g + Ja.T @ F 
    # if np.abs(Ja.T @ F).max() > 1e-1:
        # print(Ja.T @ F)
    # print("q: ", q)
    # print("dq: ", dq)
    # print("Ja.T @ F: ", Ja.T @ F)
    # print("g: ", g)
    # print("F: ", F)
    # print("tau: ", tau)
    # print("Ja: ", Ja)

    # invJa = np.linalg.pinv(Ja) 
    # error = Ja @ invJa - np.eye(6)
    # error = Ja.T @ F_g - g
    # tau = g
    
    # if np.abs(error).max() > 1e-1:
    #     print(error)

    # Apply to Mujoco
    # data_mj.qfrc_applied[:] = 0.0
    data_mj.ctrl[:] = tau

    return x, dot_x

# Test the Function of control_step()
# -----------------------------------------#
# t = 0
# for i, I in enumerate(model.inertias):
#     print(model.names[i], I.mass, I.lever, I.inertia)

# x, dot_x = control_step(t)
# print(x, dot_x)
# -----------------------------------------# 

DT = model_mj.opt.timestep
sim_time = 10.0
steps = int(sim_time / DT)

log_t = []
log_x = []
log_dx = []
log_ddx = []

t = 0.0
mujoco.mj_step(model_mj, data_mj)

for k in range(steps):
    x, dot_x = control_step(t)
    mujoco.mj_step(model_mj, data_mj)
    viewer.sync()
    
    t += DT

    # Log
    log_t.append(k)
    log_x.append(x.copy())
    log_dx.append(dot_x.copy())