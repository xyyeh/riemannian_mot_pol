from rmpflow.rmp.rmp import RMPRoot, RMPNode
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import numpy as np
import os

import eigen as e
import rbdyn as rbd
import sva as s
from rbdyn.parsers import *


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
        p = from_urdf_file(urdf_path)
        p.mbc.gravity = e.Vector3d(0, 0, 9.81)
        p.mbc.zero(p.mb)

        def J(q):
            """
            Jacobian taking in a numpy q array
            """
            pass


def rmp_tree_from_urdf(urdf_path="", base_name="root"):
    """
    Constructs rmp-tree from robot urdf, exposing all actuated joints

    @param urdf_path    URDF path
    @param base_name    name of the root node

    @return root of rmpflow tree, leaf dictionary containing all RMPNodes of actuated joints, indexed by joint names
    """
    # find all joints
    jnts = list(filter(lambda j: j.type() == 0, p.mb.joints()))
    jnt_names = list(map(lambda j: j.name().decode("utf-8"), jnts))

    # find all child links
    link_names = list(filter(lambda b: b.index))
    
