"""
implementation of toroidal planetary physics to be used in shallow water model
cylindrical coordinates with r, theta and z are used, 
as well as toroidal coordinates with phi and theta, along mayor and minor radii respectively, where phi=0 is the outer equator
theta is not used for fields that are symmetric about the z axis
"""

import numpy as np


def toroidal_gravity(r, z, r_major, rho=1, grav_const=1, integration_points=100):
    """
    returns the gravitational acceleration vector in cylindrical coordinates
    works by approximating the torus as a series of point masses along a circle and integrating over them

    input parameters:
    r : cylindrical radius
    z : height above the equatorial plane
    rho : mass density of the planet
    r_major : major radius of the torus
    grav_const : gravitational constant
    integration_points : number of integration points to use in the numerical integration of the gravitational acceleration


    returns:
    a_r : acceleration in the r direction
    a_z : acceleration in the z direction

    """

    rho = 1
    d_theta = np.pi / integration_points
    point_mass = rho * r_major * d_theta

    a_r = 0
    a_z = 0
    a_theta = 0

    for theta in np.linspace(0, 2 * np.pi, 2 * integration_points):
        # distance between point of interest and point mass on circle on the z=0 plane
        distance_z0 = np.sqrt(
            np.square(r_major) + np.square(r) - 2 * r_major * r * np.cos(theta)
        )

        # 3d distance between point of interest and point mass
        distance = np.sqrt(np.square(distance_z0) + np.square(z))

        # total acceleration
        a = grav_const * point_mass / np.square(distance)

        # force in the r direction
        distance_r = r - r_major * np.cos(theta)

        a_r += -a * (distance_r / distance)

        # force in the theta direction
        distance_theta = r_major * np.sin(theta)

        a_theta += -a * (distance_theta / distance)

        # force in the z direction
        a_z += -a * (z / distance)

    return a_r, a_z


def scale_gravity(a_r, a_z, phi, g_0):
    """
    scale the gravitational acceleration by the set gravitational acceleration at the equator

    input parameters:
    a_r : acceleration in the r direction
    a_z : acceleration in the z direction
    phi : toroidal angle, phi=0 is the outer equator
    g_0 : gravitational acceleration at the equator
    """
    # index where phi = 0
    phi_0 = np.where(phi == 0)
    a_equ = np.sqrt(np.square(a_r[phi_0]) + np.square(a_z[phi_0]))
    a_r = a_r * g_0 / a_equ
    a_z = a_z * g_0 / a_equ

    return a_r, a_z


def centrifugal_acceleration(r, omega):
    """
    returns the centrifugal acceleration vector in cylindrical coordinates

    input parameters:
    r : cylindrical radius
    omega : angular velocity

    returns:
    a_r : acceleration in the r direction
    """

    v = omega * r
    a_r = np.square(v) / r

    return a_r


def toroidal_coriolis_acceleration(phi, v_r, v_phi, v_theta, r_major, r_minor, omega):
    """
    returns the coriolis acceleration vector in toroidal coordinates

    input parameters:
    phi: toroidal angle, phi=0 is the outer equator
    v_r: velocity in the toroidal r direction
    v_phi: velocity in the toroidal phi direction
    v_theta: velocity in the toroidal theta direction
    r_major: major radius of the torus
    r_minor: minor radius of the torus
    omega: angular velocity

    returns:
    a_r: acceleration in toroidal the r direction
    a_phi: acceleration in the toroidal phi direction
    a_theta: acceleration in the toroidal theta direction
    """

    a_theta = 2 * omega * (v_phi * np.sin(phi) - v_r * np.cos(phi))
    a_phi = 2 * omega * (-v_theta * np.sin(phi))
    a_r = 2 * omega * (v_theta * np.cos(phi))

    return a_r, a_phi, a_theta


def toroidal2cylindrical(phi, r_major, r_minor, theta=None):
    """
    input parameters:
    converts from toroidal to cylindrical coordinates
    phi: toroidal angle, phi=0 is the outer equator
    r_major: major radius of the torus
    r_minor: minor radius of the torus
    theta: cylindrical angle

    returns:
    r: cylindrical radius
    z: height above the equatorial plane
    theta: cylindrical angle (if theta is given)
    """

    r = r_major + r_minor * np.cos(phi)
    z = r_minor * np.sin(phi)

    if theta is None:
        return r, z
    else:
        return r, z, theta


def vector_cylindrical2toroidal(phi, a_r, a_z, a_theta=None):
    """
    cordinate transformation of a vector a from cylindrical (r,z,theta) to toroidal coordinates (r,phi,theta)
    input parameters:
    r: cylindrical radius
    z: height above the equatorial plane
    theta: cylindrical angle
    a_r: acceleration in the r direction
    a_z: acceleration in the z direction
    a_theta: acceleration in the theta direction

    returns:
    a_t_r: acceleration in the r_minor direction (normal to the torus surface)
    a_t_phi: acceleration in the phi direction (horizontal to the torus surface, along the minor radius)
    a_t_theta: acceleration in the theta direction (along the major radius)
    """

    a_t_r = a_r * np.cos(phi) + a_z * np.sin(phi)
    a_t_phi = -a_r * np.sin(phi) + a_z * np.cos(phi)

    if a_theta is None:
        return a_t_r, a_t_phi
    else:
        return a_t_r, a_t_phi, a_theta
