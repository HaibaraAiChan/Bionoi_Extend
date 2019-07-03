from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import pandas as pd
import matplotlib
import os
from biopandas.mol2 import PandasMol2
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from sklearn.cluster import KMeans
from math import sqrt, asin, atan2, log, pi, tan

from project_mult_Dirction import xoy_positive_proj
from project_mult_Dirction import xoy_negative_proj
from project_mult_Dirction import yoz_positive_proj
from project_mult_Dirction import yoz_negative_proj
from project_mult_Dirction import zox_positive_proj
from project_mult_Dirction import zox_negative_proj


def k_different_colors(k: int):
    colors = dict(**mcolors.CSS4_COLORS)

    rgb = lambda color: mcolors.to_rgba(color)[:3]
    hsv = lambda color: mcolors.rgb_to_hsv(color)

    col_dict = [(k, rgb(k)) for c, k in colors.items()]
    X = np.array([j for i, j in col_dict])

    # Perform kmeans on rqb vectors
    kmeans = KMeans(n_clusters=k)
    kmeans = kmeans.fit(X)
    # Getting the cluster labels
    labels = kmeans.predict(X)
    # Centroid values
    C = kmeans.cluster_centers_

    # Find one color near each of the k cluster centers
    closest_colors = np.array([np.sum((X - C[i]) ** 2, axis=1) for i in range(C.shape[0])])
    keys = sorted(closest_colors.argmin(axis=1))

    return [col_dict[i][0] for i in keys]


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    Source
    -------
    Copied from https://gist.github.com/pv/8036995
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def fig_to_numpy(fig, alpha=1) -> np.ndarray:
    '''
    Converts matplotlib figure to a numpy array.

    Source
    ------
    Adapted from https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    '''

    # Setup figure
    fig.patch.set_alpha(alpha)
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data


def miller(x, y, z):
    radius = sqrt(x ** 2 + y ** 2 + z ** 2)
    latitude = asin(z / radius)
    longitude = atan2(y, x)
    lat = 5 / 4 * log(tan(pi / 4 + 2 / 5 * latitude))
    return lat, longitude

"""
return transformation coordinates(matrix: X*3) 
Principal Axes Alignment
"""
def alignment(pocket, proDirct):

    pocket_coords = np.array([pocket.x, pocket.y, pocket.z]).T
    pocket_center = np.mean(pocket_coords, axis=0)  # calculate mean of each column
    pocket_coords = pocket_coords - pocket_center   # Centralization
    inertia = np.cov(pocket_coords.T)               # get covariance matrix (of centralized data)
    e_values, e_vectors = np.linalg.eig(inertia)    # linear algebra eigenvalue eigenvector
    sorted_index = np.argsort(e_values)[::-1]       # sort eigenvalues (increase)and reverse (decrease)
    sorted_vectors = e_vectors[:, sorted_index]

    if proDirct == 1:
        transformation_matrix = xoy_positive_proj(sorted_vectors)
    elif proDirct == 2:
        transformation_matrix = xoy_negative_proj(sorted_vectors)
    elif proDirct == 3:
        transformation_matrix = yoz_positive_proj(sorted_vectors)
    elif proDirct == 4:
        transformation_matrix = yoz_negative_proj(sorted_vectors)
    elif proDirct == 5:
        transformation_matrix = zox_positive_proj(sorted_vectors)
    elif proDirct == 6:
        transformation_matrix = zox_negative_proj(sorted_vectors)

    transformed_coords = (np.matmul(transformation_matrix, pocket_coords.T)).T
    # transformed_coords = (transformation_matrix.dot(pocket_coords.T)).T

    return transformed_coords


def voronoi_atoms_coords(bs,  bs_out=None,  projection=miller, proDirct=None):
    # Suppresses warning
    pd.options.mode.chained_assignment = None
    print(os.path.basename(bs))
    # Read molecules in mol2 format
    mol2 = PandasMol2().read_mol2(bs)
    atoms = mol2.df[['subst_id', 'subst_name', 'atom_type', 'atom_name', 'x', 'y', 'z']]
    atoms.columns = ['res_id', 'residue_type', 'atom_type', 'atom_name', 'x', 'y', 'z']
    atoms['residue_type'] = atoms['residue_type'].apply(lambda x: x[0:3])

    # Align to principal Axis
    trans_coords = alignment(atoms, proDirct)
    # get the transformation coordinate
    mol2.df['x'] = trans_coords[:, 0]
    mol2.df['y'] = trans_coords[:, 1]
    mol2.df['z'] = trans_coords[:, 2]
    filename = os.path.basename(bs)
    filename_without_tail = filename.split('.')[0]

    mol2.df.to_csv(bs_out+filename_without_tail, float_format="%10.4f", sep='\t', index=False)

    return


def Bionoi_coord(mol, bs_out, proDirct):
    # Run
    atoms_coords = voronoi_atoms_coords(mol,
                                    bs_out=bs_out,
                                    proDirct=proDirct)

    return atoms_coords
