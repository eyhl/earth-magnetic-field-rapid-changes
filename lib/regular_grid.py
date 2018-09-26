import numpy as np
from math import atan2, sin, cos, sqrt, pi

def define_icosahedral_grid():
    '''
    Creates a 3d icosahedral grid. It has 20 faces and 12 vertices.
    :return:
        icosahedral_points: list of tuples with 3d-coordinates for the vertices
        icosahedral_triangles: list of tuples with indices referring to which points that makes up each triangle
    '''
    # generate base icosahedral grid
    s, c = 2 / sqrt(5), 1 / sqrt(5)

    # get all top points for triangles in icosahedral grid
    top_points = [(0, 0, 1)] + [(s * cos(i * 2 * pi / 5.), s * sin(i * 2 * pi / 5.), c) for i in range(5)]

    # get all bottom points for triangles in icosahedral grid
    bottom_points = [(-x, y, -z) for (x, y, z) in top_points]

    icosahedral_points = top_points + bottom_points

    # get indices pointing out each triangle made up by the icosahedral points
    icosahedral_triangles = [(0, i + 1, (i + 1) % 5 + 1) for i in range(5)] + \
                            [(6, i + 7, (i + 1) % 5 + 7) for i in range(5)] + \
                            [(i + 1, (i + 1) % 5 + 1, (7 - i) % 5 + 7) for i in range(5)] + \
                            [(i + 1, (7 - i) % 5 + 7, (8 - i) % 5 + 7) for i in range(5)]

    return icosahedral_points, icosahedral_triangles


def refine_triangle(refinement_degree):
    '''
    This funcion refines a equilateral unit triangle based on input refinement degree. Based on the idea
    of finding midpoints between neighboring points within each sub triangle
           2
          /\
         /  \
    mp02/____\mp12       Construct new triangles
       /\    /\
      /  \  /  \
     /____\/____\
    0    mp01    1

    :param refinement_degree: the degree of refinement
    :return:
        fixed_triangle: the (x, y)-vertices for refined fixed 2d triangle.
        n_points: total number of points in grid
        fixed_hexagon: the (x, y)-vertices for refined fixed 2d hexagon.
    '''

    # define fixed unit triangle
    fixed_triangle = np.array([[[-0.5, 0], [0.5, 0], [0, sqrt(3) / 2]]])

    # duplicate fixed triangle for refinement
    refined_triangle = fixed_triangle

    # Loop refines triangle iteratively into 4 sub triangles per loop
    for i in range(refinement_degree):
        if refinement_degree == 0:  # no refinement
            break

        mp01 = (refined_triangle[:, 0, :] + refined_triangle[:, 1, :]) / 2
        mp02 = (refined_triangle[:, 0, :] + refined_triangle[:, 2, :]) / 2
        mp12 = (refined_triangle[:, 1, :] + refined_triangle[:, 2, :]) / 2

        # collect the points of each of the 4 subtriangles
        # t1=[0,mp01,mp02], t2=[1,mp12,mp01]
        # t3=[2,mp02,mp12], t4 =[mp01,mp12,mp02]
        t1 = np.array(list(zip(refined_triangle[:, 0, :], mp01, mp02)))
        t2 = np.array(list(zip(refined_triangle[:, 1, :], mp12, mp01)))
        t3 = np.array(list(zip(refined_triangle[:, 2, :], mp02, mp12)))
        t4 = np.array(list(zip(mp01, mp12, mp02)))

        refined_triangle = np.concatenate((t1, t2, t3, t4), axis=0)

    # midpoints of triangles are hexagon vertices, found by:
    fixed_hexagon = np.sum(refined_triangle, axis=1) / 3

    # TODO: ved lejlighed så spørg lige stefan om det her.
    n_points = np.prod(np.shape(refined_triangle)[0:2])  # number of points (no. of triangles in refinement * 3)

    fixed_triangle = np.resize(refined_triangle, (n_points, 2))  # resize into a single list

    fixed_triangle = np.unique(fixed_triangle, axis=0)  # removes duplicate coordinates (local xy)

    return fixed_triangle, fixed_hexagon, n_points


def get_indices(fixed_triangle, fixed_hexagon):
    dist_xy = fixed_triangle[:, :, None] - np.transpose(fixed_hexagon)
    dist = (np.sum(dist_xy ** 2, axis=1)).round(10)  # no need to take sqrt
    indices = np.where(dist.transpose() == dist.min())  # indices of minimum values
    indices = np.reshape(indices[1], (-1, 3))

    return indices


def project_grid_to_sphere(refined_grid, icosahedral_points, icosahedral_triangles, indices):
    '''
    Projects a icosahedral grid (3d) onto a sphere. It works through each triangle (face) in the icosahedral grid and
    creates a grid inside that triangle based on the refined grid.
    If spherical_coords are set to True, the icosahedral vertices coordinates are returned in spherical coordinates
    (radians)

    :param fixed_grid:
    :param indices:
    :return:
        icosahedral_sphere: 3d spherical grid (x, y, z) as (n, 3)-array where n is number of points
        icosahedral_faces: indices pointing to coordinates, one per face.
    '''

    # project triangular grid in 2d onto sphere
    n_vertices = len(refined_grid)  # number of vertices
    n_faces = len(indices)  # number of faces
    icosahedral_sphere = np.zeros((20 * n_vertices, 3))
    icosahedral_faces = np.zeros((20 * n_faces, 3))

    for j in range(20):
        s1 = icosahedral_points[icosahedral_triangles[j][0]]
        s2 = icosahedral_points[icosahedral_triangles[j][1]]
        s3 = icosahedral_points[icosahedral_triangles[j][2]]
        for i in range(n_vertices):
            p = [refined_grid[i, 0], refined_grid[i, 1]]  # 2D point to be mapped to sphere
            icosahedral_sphere[n_vertices * j + i, :] = map_gridpoint_to_sphere(p, s1, s2, s3)

        icosahedral_faces[(n_faces * j):(n_faces * (j + 1)), :] = indices + j * n_vertices

    # print(icosahedral_sphere)
    icosahedral_faces = icosahedral_faces.astype(int)

    return icosahedral_sphere, icosahedral_faces


def barycentric_coords(p):
    # barycentric coords for triangle (-0.5,0),(0.5,0),(0,sqrt(3)/2)
    x, y = p
    lambda3 = y * 2. / sqrt(3.)  # lambda3*sqrt(3)/2 = y
    lambda2 = x + 0.5 * (1 - lambda3)  # 0.5*(lambda2 - lambda1) = x
    lambda1 = 1 - lambda2 - lambda3  # lambda1 + lambda2 + lambda3 = 1
    return lambda1, lambda2, lambda3


def scalar_product(p1, p2):
    return sum([p1[i] * p2[i] for i in range(len(p1))])


def slerp(p0, p1, t):
    # uniform interpolation of arc defined by p0, p1 (around origin)
    # t=0 -> p0, t=1 -> p1
    assert abs(scalar_product(p0, p0) - scalar_product(p1, p1)) < 1e-7
    ang0_cos = scalar_product(p0, p1) / scalar_product(p0, p0)
    ang0_sin = sqrt(1 - ang0_cos * ang0_cos)
    ang0 = atan2(ang0_sin, ang0_cos)
    l0 = sin((1 - t) * ang0)
    l1 = sin(t * ang0)
    return tuple([(l0 * p0[i] + l1 * p1[i]) / ang0_sin for i in range(len(p0))])


def map_gridpoint_to_sphere(p, s1, s2, s3):
    # map 2D point p to spherical triangle s1, s2, s3 (3D vectors of equal length)
    lambda1, lambda2, lambda3 = barycentric_coords(p)
    if abs(lambda3 - 1) < 1e-10:
        return s3
    lambda2s = lambda2 / (lambda1 + lambda2)
    p12 = slerp(s1, s2, lambda2s)
    return slerp(p12, s3, lambda3)