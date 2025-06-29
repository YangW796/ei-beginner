# coding= utf8
"""
.. module:: chain
This module implements the Chain class.
"""
import numpy as np
import json
import os
from typing import List
import warnings

# IKPY imports
from .urdf import URDF
from . import inverse_kinematics as ik
from . import link as link_lib


class Chain:
    """The base Chain class

    Parameters
    ----------
    links: list[ikpy.link.Link]
        List of the links of the chain
    active_links_mask: list
        A list of boolean indicating that whether or not the corresponding link is active
    """
    def __init__(self, links):
        self.links = links
        self._length = sum([link.length for link in links])
        self.active_links_mask=np.array([True] * len(links))

    def __len__(self):
        return len(self.links)

    def forward_kinematics(self, joints: List):
        """Returns the transformation matrix of the forward kinematics
        Parameters
        ----------
        joints: list
            The list of the positions of each joint. Note : Inactive joints must be in the list.
        Returns
        -------
        frame_matrix:
            The transformation matrix
        """
        frame_matrix = np.eye(4)
        for index, (link, joint_parameters) in enumerate(zip(self.links, joints)):
            frame_matrix = np.dot(frame_matrix, np.asarray(link.get_link_frame_matrix(joint_parameters)))
        return frame_matrix

    def inverse_kinematics(self, target_position, target_orientation, orientation_mode):
        """
        Parameters
        ----------
        target_position: np.ndarray
            Vector of shape (3,): the target point
        target_orientation: np.ndarray
            Vector of shape (3,): the target orientation
        orientation_mode: str
            Orientation to target. Choices:
            * None: No orientation
            * "X": Target the X axis
            * "Y": Target the Y axis
            * "Z": Target the Z axis
            * "all": Target the entire frame (e.g. the three axes) (not currently supported)
        Returns
        -------
        list:
            The list of the positions of each joint according to the target. Note : Inactive joints are in the list.
        """
        frame_target = np.eye(4)

        # Compute orientation
        if orientation_mode is not None:
            if orientation_mode == "X":
                frame_target[:3, 0] = target_orientation
            elif orientation_mode == "Y":
                frame_target[:3, 1] = target_orientation
            elif orientation_mode == "Z":
                frame_target[:3, 2] = target_orientation
            elif orientation_mode == "all":
                frame_target[:3, :3] = target_orientation
            else:
                raise ValueError("Unknown orientation mode: {}".format(orientation_mode))

        # Compute target
        frame_target[:3, 3] = target_position
        
        initial_position = [0] * len(self.links)
        return  ik.inverse_kinematic_optimization(self, frame_target, initial_position, orientation_mode)

    @classmethod
    def from_urdf_file(cls, urdf_file, base_elements=None, last_link_vector=None, base_element_type="link",symbolic=True):
        if base_elements is None:
            base_elements = ["base_link"]
        links = URDF.get_urdf_parameters(urdf_file, base_elements=base_elements, last_link_vector=last_link_vector, base_element_type=base_element_type, symbolic=symbolic)
        # Add an origin link at the beginning
        chain = cls([link_lib.OriginLink()] + links)
        return chain

    def active_to_full(self, active_joints, initial_position):
        full_joints = np.array(initial_position, copy=True, dtype=np.float64)
        np.place(full_joints, self.active_links_mask, active_joints)
        return full_joints

    def active_from_full(self, joints):
        return np.compress(self.active_links_mask, joints, axis=0)

