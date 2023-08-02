#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:55:28 2023

@author: ricardo
"""

import numpy as np
import os
import math
import functools


#145.4410 145.4410 135.6993 107.8946
fx = 145.4410
fy = 145.4410
cx = 135.6993
cy = 107.8946


def project_depth_map(depth_map, camera_transform):
    # Get the dimensions of the depth map
    height, width = depth_map.shape

    # Generate grid coordinates for each pixel in the depth map
    y_coords, x_coords = np.meshgrid(range(height), range(width), indexing='ij')
    camera_transform_new = camera_transform[:3, :3]
    # Convert grid coordinates to homogeneous coordinates
    #homogeneous_coords = np.stack([x_coords.flatten(), y_coords.flatten(), np.ones_like(x_coords.flatten())])
    homogeneous_coords = np.stack([(x_coords.flatten() - cx) / fx, (y_coords.flatten() - cy) / fy, np.ones_like(x_coords.flatten())])
    #print(homogeneous_coords.shape)
    #print(camera_transform.shape)
    # Ensure that the camera_transform is a 3x3 or 4x4 matrix
    if camera_transform.shape == (3, 3):
        # Convert 3x3 to 4x4 transformation matrix by appending [0, 0, 0, 1] as the last row and column
        camera_transform = np.vstack([camera_transform, [0, 0, 1]])
        camera_transform = np.hstack([camera_transform, [[0], [0], [0], [1]]])

    # Apply camera transformation to the homogeneous coordinates
    transformed_coords = np.dot(camera_transform_new, homogeneous_coords)

    # Convert homogeneous coordinates back to 2D coordinates
    proj_x_coords = transformed_coords[0] / transformed_coords[2]
    proj_y_coords = transformed_coords[1] / transformed_coords[2]

    # Clip the projected coordinates to fit within the depth map dimensions
    proj_x_coords = np.clip(proj_x_coords, 0, width - 1)
    proj_y_coords = np.clip(proj_y_coords, 0, height - 1)

    # Interpolate the projected depth values from the depth map
    projected_depth_map = np.zeros_like(depth_map)
    for y in range(height):
        for x in range(width):
            proj_x, proj_y = proj_x_coords[y * width + x], proj_y_coords[y * width + x]
            proj_x0, proj_y0 = int(proj_x), int(proj_y)
            proj_x1, proj_y1 = min(proj_x0 + 1, width - 1), min(proj_y0 + 1, height - 1)

            # Bilinear interpolation
            dx, dy = proj_x - proj_x0, proj_y - proj_y0
            projected_depth_map[y, x] = (1 - dy) * ((1 - dx) * depth_map[proj_y0, proj_x0] + dx * depth_map[proj_y0, proj_x1]) + \
                                        dy * ((1 - dx) * depth_map[proj_y1, proj_x0] + dx * depth_map[proj_y1, proj_x1])

    return projected_depth_map





def windowed_depth_averaging(depth_maps, camera_transforms, current_depth_map, window_size):
    num_keyframes = len(depth_maps)
    window_start = max(0, num_keyframes - window_size)
    window_depth_maps = depth_maps[window_start:num_keyframes-1]

    # Project the depth maps in the window to the current keyframe
    projected_depth_maps = []
    for depth_map, camera_transform in zip(window_depth_maps, camera_transforms[window_start:num_keyframes-1]):
        projected_depth_map = project_depth_map(depth_map, camera_transform)
        projected_depth_maps.append(projected_depth_map)

    # Calculate the weighted average of the projected depth maps
    weighted_avg_depth_map = np.mean(projected_depth_maps, axis=0)

    # Calculate the final depth map by averaging with the current depth map
    final_depth_map = (weighted_avg_depth_map + current_depth_map) / 2.0

    return final_depth_map

def euler2mat(z=0, y=0, x=0, isRadian=True):
    ''' Return matrix for rotations around z, y and x axes
    Uses the z, then y, then x convention above
    Parameters
    ----------
    z : scalar
         Rotation angle in radians around z-axis (performed first)
    y : scalar
         Rotation angle in radians around y-axis
    x : scalar
         Rotation angle in radians around x-axis (performed last)
    Returns
    -------
    M : array shape (3,3)
         Rotation matrix giving same rotation as for given angles
    Examples
    --------
    >>> zrot = 1.3 # radians
    >>> yrot = -0.1
    >>> xrot = 0.2
    >>> M = euler2mat(zrot, yrot, xrot)
    >>> M.shape == (3, 3)
    True
    The output rotation matrix is equal to the composition of the
    individual rotations
    >>> M1 = euler2mat(zrot)
    >>> M2 = euler2mat(0, yrot)
    >>> M3 = euler2mat(0, 0, xrot)
    >>> composed_M = np.dot(M3, np.dot(M2, M1))
    >>> np.allclose(M, composed_M)
    True
    You can specify rotations by named arguments
    >>> np.all(M3 == euler2mat(x=xrot))
    True
    When applying M to a vector, the vector should column vector to the
    right of M.  If the right hand side is a 2D array rather than a
    vector, then each column of the 2D array represents a vector.
    >>> vec = np.array([1, 0, 0]).reshape((3,1))
    >>> v2 = np.dot(M, vec)
    >>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
    >>> vecs2 = np.dot(M, vecs)
    Rotations are counter-clockwise.
    >>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
    >>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
    True
    >>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
    >>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
    True
    >>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
    >>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
    True
    Notes
    -----
    The direction of rotation is given by the right-hand rule (orient
    the thumb of the right hand along the axis around which the rotation
    occurs, with the end of the thumb at the positive end of the axis;
    curl your fingers; the direction your fingers curl is the direction
    of rotation).  Therefore, the rotations are counterclockwise if
    looking along the axis of rotation from positive to negative.
    '''

    if not isRadian:
        z = ((np.pi)/180.) * z
        y = ((np.pi)/180.) * y
        x = ((np.pi)/180.) * x
    assert z>=(-np.pi) and z < np.pi, 'Inapprorpriate z: %f' % z
    assert y>=(-np.pi) and y < np.pi, 'Inapprorpriate y: %f' % y
    assert x>=(-np.pi) and x < np.pi, 'Inapprorpriate x: %f' % x    

    Ms = []
    if z:
            cosz = math.cos(z)
            sinz = math.sin(z)
            Ms.append(np.array(
                            [[cosz, -sinz, 0],
                             [sinz, cosz, 0],
                             [0, 0, 1]]))
    if y:
            cosy = math.cos(y)
            siny = math.sin(y)
            Ms.append(np.array(
                            [[cosy, 0, siny],
                             [0, 1, 0],
                             [-siny, 0, cosy]]))
    if x:
            cosx = math.cos(x)
            sinx = math.sin(x)
            Ms.append(np.array(
                            [[1, 0, 0],
                             [0, cosx, -sinx],
                             [0, sinx, cosx]]))
    if Ms:
            return functools.reduce(np.dot, Ms[::-1])
    return np.eye(3)


def pose_vec_to_mat(vec):
    tx = vec[0]
    ty = vec[1]
    tz = vec[2]
    trans = np.array([tx, ty, tz]).reshape((3,1))
    rot = euler2mat(vec[5], vec[4], vec[3])
    Tmat = np.concatenate((rot, trans), axis=1)
    hfiller = np.array([0, 0, 0, 1]).reshape((1,4))
    Tmat = np.concatenate((Tmat, hfiller), axis=0)
    return Tmat



# Get the list of all files and directories
path = "/home/vicroni/Desktop/Sequences/rnnslam_real_data_2_enhanced/out/depth/"
path_refined = "/home/vicroni/Desktop/Sequences/rnnslam_real_data_2_enhanced/out/depth_refined/"
dir_list = os.listdir(path)
 
depths_bin = []
poses_bin = []
depths = []
poses = []

final_depths = []

for i in  dir_list:
    if "depth" in i:
        depths_bin.append(i)
    else:
        poses_bin.append(i)
    

depths_bin.sort()
poses_bin.sort()

for idx, i in enumerate(depths_bin):
    print(i)
    if (len(np.fromfile(path+depths_bin[idx], dtype=np.float32)) > 0 and len(np.fromfile(path+poses_bin[idx], dtype=np.float32)) > 0):
        depths.append(np.fromfile(path+depths_bin[idx], dtype=np.float32).reshape(270,216))
        poses.append(pose_vec_to_mat(np.fromfile(path+poses_bin[idx], dtype=np.float32)))



window_size = 7

for i in range(len(depths)-1, -1, -1):
    #print(i)
    current_depth_map = depths[i]
    
    final_depth_map = windowed_depth_averaging(depths, poses, current_depth_map, window_size)
    
    depth_output_path = os.path.join(path_refined, depths_bin[i])
    print(i,len(depths))
    file2 = open(depth_output_path,'a+') 
    with open(depth_output_path, 'wb') as file:
        final_depth_map.astype(np.float32).tofile(file)

