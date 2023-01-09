import os
from typing import Union
import numpy as np
import pandas as pd
from osgeo import gdal
from osgeo.osr import CoordinateTransformation, SpatialReference
from rnet import gdal_merge
from rnet.utils import random_string


def idw_query(points: np.ndarray, crs: int, *paths: str, r: float = 1e-3,
              p: int = 2, return_xy: bool = False) -> np.ndarray:
    '''
    Query elevations of two-dimensional points via IDW interpolation.

    Parameters
    ----------
    points : :class:`~numpy.ndarray`, shape (N, 2)
        Points whose elevations are queried.
    crs : int
        EPSG code of CRS in which point coordinates are represented.
    *paths : str
        Path(s) to TIF file(s) storing elevation data.
    r : float, optional
        Search radius for nearby points in degrees. The default is 0.001.
    p : int, optional
        Power setting for IDW interpolation. The default is 2.
    return_xy : bool, optional
        If True, :math:`(x, y, z)` coordinates are returned. Otherwise,
        only :math:`z`-coordinates are returned. The default is False.
    
    Returns
    -------
    elevations : :class:`~numpy.ndarray`, shape (N,)
        Array of elevations where ``elevations[i]`` is the elevation at
        ``points[i]``.
    '''
    # Read TIF sources
    if len(paths) == 0:
        raise ValueError('missing TIF sources')
    elif len(paths) == 1:
        path = paths[0]
    else:
        path = os.path.join(os.path.commonpath(paths), random_string('merged_', '.tif'))
        gdal_merge.main(['', '-o', path, *paths])
    source = gdal.Open(path)
    x0, dx, _, y0, _, dy = source.GetGeoTransform()
    nx = source.RasterXSize
    ny = source.RasterYSize
    x = np.arange(x0, x0 + nx * dx, dx)
    y = np.arange(y0, y0 + ny * dy, dy)
    z = source.GetRasterBand(1).ReadAsArray()

    # Find elevations
    if crs == 4326:
        points_ = points
    else:
        points_ = transform_coords(points, src=crs, dst=4326)
    indices_x = np.searchsorted(x, points_[:,0])
    indices_y = ny - np.searchsorted(y[::-1], points_[:,1])
    dx = int(np.abs(r / dx))
    dy = int(np.abs(r / dy))
    left = np.clip(indices_x - dx, 0, None)
    right = np.clip(indices_x + dx, None, nx)
    top = np.clip(indices_y - dy, 0, None)
    bottom = np.clip(indices_y + dy, None, ny)

    elevations = []
    for p, l, r, t, b in zip(points_, left, right, top, bottom):
        z_ = z[t:b,l:r]  # Elevations of nearby points
        xs, ys = np.meshgrid(x[l:r]-p[0], y[t:b]-p[1])
        d = np.sqrt(xs**2 + ys**2)  # Distances to nearby points
        elev = float(np.sum(z_/d) / np.sum(1/d))
        elevations.append(elev)
    
    if return_xy:
        return np.column_stack((points, elevations))
    else:
        return np.array(elevations)


def transform_coords(*args, src: int, dst: int, copy: bool = False
                     ) -> Union[np.ndarray, pd.DataFrame]:
    '''
    Transform two- or three-dimensional coordinates from one CRS to
    another.

    Parameters
    ----------
    *args : tuple
        Either of the following:

            * :class:`~numpy.ndarray` of coordinates, or
            * 2-tuple containing :class:`~pandas.DataFrame` and list of
              column names containing coordinates to be transformed.

    src, dst : int
        EPSG codes of source and destination CRSs.
    
    Other parameters
    ----------------
    copy : bool, optional
        If True, then a copy of the :class:`~pandas.DataFrame` is
        returned. Otherwise, the given :class:`~pandas.DataFrame` is
        updated in place. The default is False.

    Returns
    -------
    transformed : :class:`~numpy.ndarray` or :class:`~pandas.DataFrame`
        Transformed coordinates.
    '''
    src_ = SpatialReference()
    src_.ImportFromEPSG(src)
    dst_ = SpatialReference()
    dst_.ImportFromEPSG(dst)
    ct = CoordinateTransformation(src_, dst_)

    if len(args) == 1:
        coords = args[0]
    elif len(args) == 2:
        df, cols = args
        if copy:
            df_ = df.copy()
        else:
            df_ = df
        coords = df_[cols].to_numpy()

    M = coords.shape[1]
    if M == 2:
        transformed = np.array(ct.TransformPoints(coords[:,[1,0]]))[:,[1,0]]
    elif M == 3:
        transformed = np.array(ct.TransformPoints(coords[:,[1,0,2]]))[:,[1,0,2]]
    
    if len(args) == 1:
        return transformed
    elif len(args) == 2:
        df_[cols] = transformed
        return df_
