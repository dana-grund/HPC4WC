"""
implementation of toroidal planetary physics to be used in shallow water model
cylindrical coordinates with r, theta and z are used, 
as well as toroidal coordinates with phi and theta, along mayor and minor radii respectively, where phi=0 is the outer equator
theta is not used for fields that are symmetric about the z axis
"""

import numpy as np


def toroidal_gravity(r, z, rho, major_r,grav_const=1, integration_points=100):
    """
    returns the gravitational acceleration vector in cylindrical coordinates
    works by approximating the torus as a series of point masses along a circle and integrating over them

    input parameters:
    r : cylindrical radius
    z : height above the equatorial plane
    rho : mass density of the planet
    major_r : major radius of the torus
    grav_const : gravitational constant 
    integration_points : number of integration points to use in the numerical integration of the gravitational acceleration
    

    returns:
    a_r : acceleration in the r direction
    a_z : acceleration in the z direction

    """

    d_theta = np.pi / integration_points
    point_mass = rho * major_r * d_theta

    a_r = 0
    a_z = 0

    # we only need to integrate over half the torus, since the other half is symmetric and the force is in the opposite direction and cancels out
    for theta in np.linspace(0, np.pi, integration_points):
        # distance between point of interest and point mass on circle on the z=0 plane
        distance_z0 = np.sqrt(
            np.square(major_r)
            + np.square(major_r + r)
            - 2 * major_r * (major_r + r) * np.cos(theta)
        )
        # 3d distance between point of interest and point mass
        distance = np.sqrt(np.square(distance_z0) + np.square(z))

        # total acceleration
        a = grav_const*point_mass/np.square(distance)
        
        # todo: from here on, somethings wrong
        # force in the r direction (times 2 because of symmetry)
        distance_r = np.sqrt(np.square(r) + np.square(np.sin(theta)*major_r))
        a_r += 2* a * (distance_r / distance)

        # force in the z direction (times 2 because of symmetry)
        distance_z = np.sqrt(np.square(z) + np.square(np.sin(theta)*major_r))
        a_z += 2* a * (distance_z / distance)

    return a_r, a_z


def toroidal2cylindrical(phi, major_r, minor_r, theta=None):
    """
    input parameters:
    converts from toroidal to cylindrical coordinates
    phi: toroidal angle, phi=0 is the outer equator
    major_r: major radius of the torus
    minor_r: minor radius of the torus
    theta: cylindrical angle

    returns:
    r: cylindrical radius
    z: height above the equatorial plane
    theta: cylindrical angle (if theta is given)
    """

    r = major_r + minor_r * np.cos(phi)
    z = minor_r * np.sin(phi)

    if theta is None:
        return r, z
    else:
        return r, z, theta
