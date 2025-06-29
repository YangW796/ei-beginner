# coding= utf8
import scipy.optimize
import numpy as np



ORIENTATION_COEFF = 1.


def inverse_kinematic_optimization(chain, target_frame, starting_nodes_angles,orientation_mode):
    """
    Computes the inverse kinematic on the specified target

    Parameters
    ----------
    chain: ikpy.chain.Chain
        The chain used for the Inverse kinematics.
    target_frame: numpy.array
        The desired target.
    starting_nodes_angles: numpy.array
        The initial pose of your chain.
    orientation_mode: str
        Orientation to target. Choices:
        * None: No orientation
        * "X": Target the X axis
        * "Y": Target the Y axis
        * "Z": Target the Z axis
        * "all": Target the three axes
    """

    # Begin with the position
    target = target_frame[:3, -1]

    # Initial function call when optimizing
    def optimize_basis(x):
        y = chain.active_to_full(x, starting_nodes_angles)
        fk = chain.forward_kinematics(y)

        return fk

    # Compute error to target
    def optimize_target_function(fk):
        target_error = (fk[:3, -1] - target)
        return target_error

    if orientation_mode == "X":
        target_orientation = target_frame[:3, 0]

        def get_orientation(fk):
            return fk[:3, 0]

    elif orientation_mode == "Y":
        target_orientation = target_frame[:3, 1]

        def get_orientation(fk):
            return fk[:3, 1]

    elif orientation_mode == "Z":
        target_orientation = target_frame[:3, 2]

        def get_orientation(fk):
            return fk[:3, 2]

    elif orientation_mode == "all":
        target_orientation = target_frame[:3, :3]

        def get_orientation(fk):
            return fk[:3, :3]
    else:
        raise ValueError("Unknown orientation mode: {}".format(orientation_mode))

    def optimize_function(x):
        fk = optimize_basis(x)
        target_error = optimize_target_function(fk)
        orientation_error = (get_orientation(fk) - target_orientation).ravel()
        total_error = np.concatenate([target_error, ORIENTATION_COEFF * orientation_error])

        return total_error

    # Compute bounds
    real_bounds = [link.bounds for link in chain.links]
    # real_bounds = real_bounds[chain.first_active_joint:]
    real_bounds = chain.active_from_full(real_bounds)

    real_bounds = np.moveaxis(real_bounds, -1, 0)
    res = scipy.optimize.least_squares( optimize_function, chain.active_from_full(starting_nodes_angles), bounds=real_bounds)

    return chain.active_to_full(res.x, starting_nodes_angles)
