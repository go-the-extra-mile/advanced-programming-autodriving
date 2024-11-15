import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time
import sys


def normalize(v):
    norm = np.linalg.norm(v, axis=0) + 0.00001
    return v / norm.reshape(1, v.shape[1])


def curvature(waypoints):
    """
    Curvature as the sum of the normalized dot product between the way elements
    Implement second term of the smoothing objective.

    args:
        waypoints [2, num_waypoints] !!!!! (2D array of shape (2, num_waypoints))
    """
    x = waypoints[0]
    y = waypoints[1]
    dx = np.diff(x)
    dy = np.diff(y)
    waypoints_matrix = np.concatenate((dx, dy), axis=0).reshape(2, -1)
    waypoints_matrix = normalize(waypoints_matrix)
    dot_products = np.sum(waypoints_matrix[:, :-1] * waypoints_matrix[:, 1:], axis=0)
    term = np.sum(dot_products)

    return term


def smoothing_objective(waypoints, waypoints_center, weight_curvature=40):
    """
    Objective for path smoothing

    args:
        waypoints [2 * num_waypoints] !!!!! (1D flattened array)
        waypoints_center [2 * num_waypoints] !!!!! (1D flattened array)
        weight_curvature (default=40)
    """
    # mean least square error between waypoint and way point center
    ls_tocenter = np.mean((waypoints_center - waypoints) ** 2)

    # derive curvature
    curv = curvature(waypoints.reshape(2, -1))

    return -1 * weight_curvature * curv + ls_tocenter


def waypoint_prediction(
    roadside1_spline, roadside2_spline, num_waypoints=6, way_type="smooth"
):
    """
    Predict waypoint via two different methods:
    - center
    - smooth

    args:
        roadside1_spline
        roadside2_spline
        num_waypoints (default=6)
        waytype (default="smoothed")
    """
    if way_type == "center":
        # create spline arguments
        t = np.linspace(start=0, stop=1, num=num_waypoints)

        # derive roadside points from spline
        lane_boundary1_points_points = np.array(splev(t, roadside1_spline))
        lane_boundary2_points_points = np.array(splev(t, roadside2_spline))

        # derive center between corresponding roadside points
        waypoints = np.zeros((2, num_waypoints))
        for i in range(num_waypoints):
            waypoints[0, i] = (
                lane_boundary1_points_points[0, i] + lane_boundary2_points_points[0, i]
            ) / 2
            waypoints[1, i] = (
                lane_boundary1_points_points[1, i] + lane_boundary2_points_points[1, i]
            ) / 2

        # output way_points with shape(2 x Num_waypoints)
        return waypoints

    elif way_type == "smooth":
        # create spline points
        t = np.linspace(start=0, stop=1, num=num_waypoints)

        # roadside points from spline
        lane_boundary1_points_points = np.array(splev(t, roadside1_spline))
        lane_boundary2_points_points = np.array(splev(t, roadside2_spline))

        # center between corresponding roadside points
        waypoints_center = np.zeros((2, num_waypoints))
        for i in range(num_waypoints):
            waypoints_center[0, i] = (
                lane_boundary1_points_points[0, i] + lane_boundary2_points_points[0, i]
            ) / 2
            waypoints_center[1, i] = (
                lane_boundary1_points_points[1, i] + lane_boundary2_points_points[1, i]
            ) / 2

        # init optimized waypoints
        init_waypoints = np.copy(waypoints_center)

        # optimization
        res = minimize(
            smoothing_objective,
            init_waypoints.reshape(-1),
            args=(waypoints_center.reshape(-1),),
            method="L-BFGS-B",
            bounds=[(0, 95)] * num_waypoints * 2,
        )

        return res.x.reshape(2, -1)


def target_speed_prediction(
    waypoints, num_waypoints_used=5, max_speed=60, exp_constant=4.5, offset_speed=30
):
    """
    Predict target speed given waypoints
    Implement the function using curvature()

    args:
        waypoints [2, num_waypoints]
        num_waypoints_used (default=5)
        max_speed (default=60)
        exp_constant (default=4.5)
        offset_speed (default=30)

    output:
        target_speed (float)
    """

    waypoints_used = waypoints[:, :num_waypoints_used]
    min_speed = max_speed - offset_speed

    return (
        offset_speed
        * np.exp(
            -exp_constant * np.abs(num_waypoints_used - 2 - curvature(waypoints_used))
        )
        + min_speed
    )
