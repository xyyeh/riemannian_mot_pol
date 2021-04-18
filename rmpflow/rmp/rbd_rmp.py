from rmpflow.rmp.rmp import RMPRoot, RMPNode
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import numpy as np
import os

import eigen as e
import rbdyn as rbd
import sva as s
from rbdyn.parsers import *

import math

class RBDRMPNode(RMPNode):
    """
    Builds a new rmb node, psi is forward kinematics
    """
    def __init__(
        self,
        name,
        parent,
        robot,
        base_link,
        end_link,
        urdf_path="",
        offset=np.zeros((3, 1)),
    ):
        # robot structure
        self.robot = robot

        # selection matrix to swap coordinates
        self.SelMatrix = e.MatrixXd.Zero(6,6)
        self.SelMatrix.coeff(0,3,1)
        self.SelMatrix.coeff(1,4,1)
        self.SelMatrix.coeff(2,5,1)
        self.SelMatrix.coeff(3,0,1)
        self.SelMatrix.coeff(4,1,1)
        self.SelMatrix.coeff(5,2,1)

        # forward kinematics
        def update_joint_pos(q):
            self.robot.mbc.q = [
                [],
                [q[0]],
                [q[1]],
                [q[2]],
                [q[3]],
                [q[4]],
                [q[5]],
                [q[6]],
            ]

        def update_joint_vel(dq):
            self.robot.mbc.alpha = [
                [],
                [dq[0]],
                [dq[1]],
                [dq[2]],
                [dq[3]],
                [dq[4]],
                [dq[5]],
                [dq[6]],
            ]

        def psi(q):
            update_joint_pos(q)
            rbd.forwardKinematics(self.robot.mb, self.robot.mbc)
            sv = self.robot.mbc.bodyPosW[-1]
            p = sv.translation()
            r = R.from_matrix(np.array(sv.rotation().transpose())) 
            eul = r.as_euler('zyx', degrees=False)
            return np.array([[p.x(), p.y(), p.z(), eul[0], eul[1], eul[2]]]).transpose()

        def J(q):
            """
            Jacobian taking in a numpy q array
            """
            update_joint_pos(q)
            update_joint_vel(dq)
            jac = rbd.Jacobian(self.robot.mb, self.robot.mb.bodies()[-1].name().decode("utf-8"))
            swapped_jac = e.MatrixXd(6, self.robot.mb.nrDof())
            jac.fullJacobian(self.robot.mb, jac.jacobian(self.robot.mb, self.robot.mbc), swapped_jac)
            J = self.SelMatrix*swapped_jac
            return np.array(J)

        def J_dot(q, dq):
            """
            Jacobian dot taking in a numpy q and dq array
            """
            update_joint_states(q)
            jac = rbd.Jacobian(self.robot.mb, self.robot.mb.bodies()[-1].name().decode("utf-8"))
            swapped_jac_dot = e.MatrixXd(6, self.robot.mb.nrDof())
            jac.fullJacobian(self.robot.mb, jac.jacobianDot(self.robot.mb, self.robot.mbc), swapped_jac_dot)
            J_dot = self.SelMatrix*swapped_jac_dot
            return np.array(J_dot)

        super().__init__(name, parent, psi, J, J_dot, verbose=False)

    def update_kinematics(self, q, dq):
        """
        Update kinematics using values from physics engine
        @param q A list of joint angles
        @param dq A list of joint velocities
        """
        self.robot.mbc.q = [
            [],
            [q[0]],
            [q[1]],
            [q[2]],
            [q[3]],
            [q[4]],
            [q[5]],
            [q[6]],
        ]
        self.robot.mbc.alpha = [
            [],
            [dq[0]],
            [dq[1]],
            [dq[2]],
            [dq[3]],
            [dq[4]],
            [dq[5]],
            [dq[6]],
        ]

        # forward kinematics
        rbd.forwardKinematics(self.robot.mb, self.robot.mbc)
        rbd.forwardVelocity(self.robot.mb, self.robot.mbc)

    def update_dynamics(self):
        """
        Update dynamics using values from physics engine to compute M, Minv and h
        @return M, Minv and h
        """
        # mass matrix
        fd = rbd.ForwardDynamics(self.robot.mb)
        fd.computeH(self.robot.mb, self.robot.mbc)
        self.M = fd.H()
        self.Minv = self.M.inverse()
        # nonlinear effects vector
        fd.computeC(self.robot.mb, self.robot.mbc)
        self.h = fd.C()

        return M, Minv, h


class ProjectionNode(RMPNode):
    """
    Constructs a new node with map that passes the parameters (in the same order) as specified by param_map
    Param_map is the same length as the state vector of the parent node, with 1's in the indices for parameters to be passed and 0's in the indices for parameters to be withheld
    """

    def __init__(self, name, parent, param_map):
        one_map = param_map.astype("int32")
        mat = np.zeros((np.sum(one_map), one_map.size), dtype="float64")

        i_mat = 0
        for i in range(0, one_map.size):
            if one_map[i] == 1:
                mat[i_mat][i] = 1
                i_mat += 1

        psi = lambda y: np.dot(mat, y)
        J = lambda x: mat
        J_dot = lambda x, xd: np.zeros_like(mat)
        super().__init__(name, parent, psi, J, J_dot)


class PositionProjection(ProjectionNode):
    """
    Convenience method to pass position from RBDRMPNode state
    """

    def __init__(self, name, parent):
        super().__init__(name, parent, np.array([1, 1, 1, 0, 0, 0]))


class RotationProjection(ProjectionNode):
    """
    Convenience method to pass rotation (Euler ZYX) from RBDRMPNode state
    """

    def __init__(self, name, parent):
        super().__init__(name, parent, np.array([0, 0, 0, 1, 1, 1]))


def rotation_mat_to_euler(rotation):
    """
    Converts rotation matrix to euler zyx
    @param rotation     Rotation matrix
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return x, y, z


def rmp_tree_from_urdf(urdf_path="", base_name="root"):
    """
    Constructs rmp-tree from robot urdf, exposing all actuated joints

    @param urdf_path    URDF path
    @param base_name    name of the root node

    @return root of rmpflow tree, leaf dictionary containing all RMPNodes of actuated joints, indexed by joint names
    """
    # load from urdf
    robot = from_urdf_file(urdf_path)
    robot.mbc.gravity = e.Vector3d(0, 0, 9.81)
    robot.mbc.zero(robot.mb)

    # find all joints
    jnts = list(filter(lambda j: j.type() == 0, robot.mb.joints()))
    jnt_names = list(map(lambda j: j.name().decode("utf-8"), jnts))

    # find all child links, assuming that we have a serial chain
    link_names = [
        robot.mb.bodies()[i].name().decode("utf-8")
        for i in range(1, len(robot.mb.bodies()))
    ]

    # find all parent links, assuming that we have a serial chain
    parent_dict = {}
    for i in range(1, len(link_names)):
        parent_dict[link_names[i]] = link_names[i - 1]

    # construct RMP root
    root = RMPRoot("root")

    # construct tree with link names
    qlen = len(link_names)
    leaf_dict = {}
    proj_dict = {}

    # construct branch for each segment
    for i in range(1, qlen + 1):
        seg_name = link_names[i - 1]

        # isolate the joint angles
        proj_vect = np.copy(
            proj_dict.get(parent_dict.get(seg_name), np.array([0] * qlen))
        )

        # index represents actuator joint angle
        proj_vect[i - 1] = 1

        # store in dictionary for child link's reference
        proj_dict[seg_name] = proj_vect

        # setup node
        proj_node = ProjectionNode("proj_" + seg_name, root, proj_vect)
        seg_node = RBDRMPNode(seg_name, proj_node, robot, base_name, seg_name)
        leaf_dict[seg_node] = seg_node

    return root, leaf_dict


if __name__ == "__main__":

    robot = from_urdf_file(urdf_path)

    root = RMPRoot("root")
    # leaf = RBDRMPNode("leaf", root, robot, base_name, seg_name)