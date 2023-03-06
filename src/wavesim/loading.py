'''
Code for generating wave load using Morison Loading

ADD REF HERE

'''
import numpy as np


def morison_load(u, du, diameter=1.0, rho=1024.0, c_m=1.0, c_d=1.0):
    """compute unit Morison load for a vertical cylinder

    Args:
        u (np.ndarray): horizontal velocity [m/s]
        du (np.ndarray): horizontal acceleration [m/s^2]
        diameter (float, optional): _description_. Defaults to 1.0. [m]
        rho (float, optional): _description_. Defaults to 1024.0. [kg/m^3]
        c_m (float, optional): _description_. Defaults to 1.0. [unitless]
        c_d (float, optional): _description_. Defaults to 1.0. [unitless]

    Returns:
        np.ndarray: horizontal unit morrison load [N/m]
    """

    return rho * c_m * (np.pi / 4) * (diameter ** 2) * du + 0.5 * rho * c_d * diameter * u * np.abs(u)
