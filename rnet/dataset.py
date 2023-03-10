from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
import sys
from typing import Any, Dict, Iterable, Generator, List, NamedTuple, Set, Tuple, Union
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from rnet.coordinates import transform_coords, idw_query, densify
from rnet.env import _QGIS_AVAILABLE, require_qgis
from rnet.geometry import polyline_length

if _QGIS_AVAILABLE:
    from qgis.core import (
        QgsFeature,
        QgsGeometry,
        QgsPointXY,
        QgsTask
    )


Action = Tuple[int, int]  # (connection_id, destination_id)


class Field(NamedTuple):
    name: str
    type: str
    required: bool
    include: bool = True
    default: Any = np.nan


class Error(Exception):
    pass


class DimensionError(Error):
    '''
    Raised if requested dimensions exceed available dimensions.

    Parameters
    ----------
    av, req : int
        Available and requested dimensions.
    '''

    def __init__(self, av: int, req: int):
        super().__init__(f'{av} dimensions available, {req} requested')


class MissingColumnError(Error):
    '''
    Raised if a required column is missing from a dataset.

    Parameters
    ----------
    column_name : str
        Name of missing column.
    '''

    def __init__(self, column_name: str):
        super().__init__(f'missing required column {column_name!r}')


class Dataset(ABC):
    '''
    Base class for all datasets.

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        Data frame.
    crs : int or None, optional
        EPSG code of CRS in which coordinates are represented. The
        default is None.
    '''

    def __init__(self, df: pd.DataFrame, crs: int = None) -> None:
        self._df = validate(df, self.FIELDS)
        self._crs = crs

    def __contains__(self, id_: int) -> bool:
        return id_ in self._df.index

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: {len(self):,} rows (EPSG:{self._crs})>'

    def __iter__(self):
        return self._df.itertuples(name=self._ELEMENT_NAME)

    @property
    def crs(self) -> int:
        '''
        EPSG code of the CRS in which coordinates are represented.

        Returns
        -------
        int
        '''
        return self._crs

    @property
    def df(self) -> pd.DataFrame:
        '''
        Frame representing the dataset.

        Returns
        -------
        :class:`~pandas.DataFrame`
        '''
        return self._df[self._active_columns()]

    def _active_columns(self) -> List[str]:
        '''
        Return list of active columns.

        A column is active if (a) it is a required column, or (b) it is
        an optional column containing non-default values.

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            Data frame.
        fields : List[Field]
            List of fields.

        Returns
        -------
        cols : List[str]
            List containing names of active columns.
        '''
        cols = []
        for field in self.FIELDS:
            if field.required:
                cols.append(field.name)
            else:
                try:
                    assert not np.all(np.isnan(self._df[field.name]))
                except AssertionError:
                    continue
                except:
                    cols.append(field.name)
                else:
                    cols.append(field.name)
        return cols

    def reset_dtypes(self) -> None:
        '''
        Apply prescribed data types.
        '''
        self._df = self._df.astype({f.name: f.type for f in self.FIELDS})

    def to_csv(self, path_to_csv: str, *, columns: List[str] = 'active'
               ) -> None:
        '''
        Export dataset to a CSV file.

        Parameters
        ----------
        path_to_csv : str
            Export path.
        columns : {'active', 'all'} or List[str], optional
            If 'active', then only active columns are exported. If
            'all', then all columns are exported. Alternatively, specify
            which columns are exported by passing a list of column
            names. The default is 'active'.
        '''
        np.set_printoptions(threshold=sys.maxsize)
        if columns == 'active':
            self.df.to_csv(path_to_csv)
        elif columns == 'all':
            self._df.to_csv(path_to_csv)
        else:
            self._df[columns].to_csv(path_to_csv)
        np.set_printoptions(threshold=1000)

    def to_pickle(self, path_to_pickle: str, *, columns: List[str] = 'active'
                  ) -> None:
        '''
        Pickle dataset.

        Parameters
        ----------
        path_to_pickle : str
            Export path.
        columns : {'active', 'all'} or List[str], optional
            If 'active', then only active columns are exported. If
            'all', then all columns are exported. Alternatively, specify
            which columns are exported by passing a list of column
            names. The default is 'active'.
        '''
        if columns == 'active':
            self.df.to_pickle(path_to_pickle)
        elif columns == 'all':
            self._df.to_pickle(path_to_pickle)
        else:
            self._df[columns].to_pickle(path_to_pickle)

    @classmethod
    def from_csv(cls, path_to_csv: str, crs: int):
        '''
        Construct dataset from a CSV file.

        Parameters
        ----------
        path_to_csv : str
            Import path.
        crs : int
            EPSG code of CRS in which coodinates are represented.
        '''
        return cls(pd.read_csv(path_to_csv), crs)

    @classmethod
    def from_pickle(cls, path_to_pickle: str, crs: int):
        '''
        Construct dataset from a pickle file

        Parameters
        ----------
        path_to_pickle : str
            Import path.
        crs : int
            EPSG code of CRS in which coodinates are represented.
        '''
        return cls(pd.read_pickle(path_to_pickle), crs)


def dataset(layer_name: Union[str, None] = None):
    '''
    Decorator for setting class attributes of a dataset.

    Parameters
    ----------
    name : str or None, optional
        Layer name used when rendering features.
    '''
    def decorate(cls):
        cls._ELEMENT_NAME = cls.__name__.replace('Data', '')
        if not layer_name:
            cls._LAYER_NAME = cls._ELEMENT_NAME.lower() + 's'
        else:
            cls._LAYER_NAME = layer_name
        return cls
    return decorate


def validate(df: pd.DataFrame, fields: List[Field]) -> pd.DataFrame:
    '''
    Ensure that a data frame has all required fields.

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        Data frame to be validated.
    fields : List[Field]
        List of fields.

    Returns
    -------
    df : :class:`pandas.DataFrame`

    Raises
    ------
    MissingColumnError
        If a required column is missing.
    '''
    # TODO: raise error if crs is not given for coordinates
    for field in fields:
        if field.name in df.columns:
            continue
        elif field.required:
            raise MissingColumnError(field.name)
        else:
            df[field.name] = field.default
    return df


@dataset()
class PointData(Dataset):
    '''
    Class representing point data.

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        Data frame.
    crs : int
        EPSG code of CRS in which point coordinates are represented.

    See also
    --------
    :class:`VertexData`
        Class representing vertex data.
    :class:`NodeData`
        Class representing node data.
    '''

    FIELDS = (
        Field('x', 'float64', True, False),
        Field('y', 'float64', True, False),
        Field('z', 'float64', False)
    )

    def coords(self, dims: int = None) -> np.ndarray:
        '''
        Return array of point coordinates.

        Parameters
        ----------
        dims : {2, 3} or None, optional
            Whether to return two- or three-dimensional coordinates. If
            None, then the :attr:`dims` property is used. The default is
            None.

        Returns
        -------
        :class:`numpy.ndarray`, shape (N, 2) or (N, 3)
            Point coordinates.

        Raises
        ------
        DimensionError
            If `dims` exceeds the :attr:`dims` attribute.
        '''
        if dims is None:
            dims = self.dims
        if dims == 2:
            cols = ['x', 'y']
        elif dims == 3:
            if self.dims == 2:
                raise DimensionError(2)
            cols = ['x', 'y', 'z']
        return self._df[cols].to_numpy(dtype=float)

    @property
    def dims(self):
        '''
        Whether points stored in the dataset are two- or
        three-dimensional.

        Returns
        -------
        dims : int
            Number of dimensions.
        '''
        if 'z' in self._active_columns():
            return 3
        else:
            return 2

    def elevate(self, *paths: str, r: int = 1e-3, p: int = 2):
        '''
        Compute elevations and update :math:`z`-coordinates.

        Parameters
        ----------
        *paths : str
            Path(s) to TIF file(s) storing elevation data.
        r : float, optional
            Search radius for nearby points in degrees. The default is
            0.001.
        p : int, optional
            Power setting for IDW interpolation. The default is 2.

        See also
        --------
        :meth:`flatten`
            Remove :math:`z`-coordinate from all points.
        :func:`idw_query`
            Query elevations of two-dimensional points via IDW
            interpolation.
        '''
        self._df['z'] = idw_query(self.coords(2), self.crs, *paths, r=r, p=p)

    @require_qgis
    def features(self, task: QgsTask) -> Generator[QgsFeature, None, None]:
        '''
        Generate features for insertion into a vector layer.

        Parameters
        ----------
        task : :class:`~qgis.core.QgsTask`
            Task for rendering features.

        Yields
        ------
        :class:`~qgis.core.QgsFeature`
        '''
        num_rows = len(self)
        includes = [index for index, field in enumerate(self.FIELDS, 1)
                    if field.include]
        for i, item in enumerate(self):
            task.setProgress(i/num_rows*100)
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPointXY(
                QgsPointXY(item.x, item.y)))
            feat.setAttributes(
                [item.Index] + [item[index] for index in includes])
            yield feat

    def flatten(self):
        '''
        Remove :math:`z`-coordinate from all points.

        See also
        --------
        :meth:`elevate`
            Compute :math:`z`-coordinates and update data frame.
        '''
        self._df['z'] = None

    def info(self, sep: str = '\n', ret: bool = False) -> None:
        '''
        Print dataset information.

        Parameters
        ----------
        sep : str, optional
            Separator between lines. The default is '\n'.
        ret : bool, optional
            If True, the information is returned as a string. Otherwise,
            it is printed. The default is False.

        Returns
        -------
        info : str
            If True, then the information is returned.
        '''
        coords = self.coords(2)
        xmin, ymin = np.min(coords, axis=0)
        xmax, ymax = np.max(coords, axis=0)
        info = sep.join([
            str(self.__class__),
            f'Count: {len(self):,}',
            f'CRS: EPSG:{self.crs}',
            f'dims: {self.dims}',
            f'xmin: {xmin:.07f}',
            f'ymin: {ymin:.07f}',
            f'xmax: {xmax:.07f}',
            f'ymax: {ymax:.07f}'
        ])
        if ret:
            return info
        else:
            print(info)

    def mask(self, xmin: float = None, ymin: float = None, xmax: float = None,
             ymax: float = None) -> np.ndarray:
        '''
        Return mask.

        Parameters
        ----------
        xmin, ymin, xmax, ymax : float or None, optional
            Minimum and maximum :math:`x`- and :math:`y`-coordinates.
            The defaults are None.

        Returns
        -------
        mask : :class:`~numpy.ndarray`
            Masked array.
        '''
        mask = np.full(len(self), True)
        if xmin:
            mask = mask & (self._df['x'].to_numpy() > xmin)
        if ymin:
            mask = mask & (self._df['y'].to_numpy() > ymin)
        if xmax:
            mask = mask & (self._df['x'].to_numpy() < xmax)
        if ymax:
            mask = mask & (self._df['y'].to_numpy() < ymax)
        return mask

    def transform(self, dst: int) -> None:
        '''
        Transform point coordinates to another CRS.

        Parameters
        ----------
        dst : int
            EPSG code of the destination CRS.
        '''
        if self.dims == 2:
            cols = ['x', 'y']
        elif self.dims == 3:
            cols = ['x', 'y', 'z']
        transform_coords(self._df, cols, src=self.crs, dst=dst)
        self._crs = dst


@dataset('vertices')
class VertexData(PointData):
    '''
    Class representing vertex data.

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        Data frame.
    crs : int
        EPSG code of CRS in which vertex coordinates are represented.
    '''
    pass


@dataset()
class NodeData(PointData):
    '''
    Class representing node data.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Data frame.
    crs : int
        EPSG code of CRS in which node coordinates are represented.
    '''
    pass


@dataset()
class BorderNodeData(PointData):
    '''
    Class representing border node data.

    .. versionadded:: 0.0.7

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Data frame.
    crs : int
        EPSG code of CRS in which node coordinates are represented.
    '''

    FIELDS = (
        Field('x', 'float64', True, False),
        Field('y', 'float64', True, False),
        Field('z', 'float64', False),
        Field('group', 'uint16', False)
    )

    def to_dict(self) -> Dict[int, List[int]]:
        '''
        Return dictionary mapping region ID to sorted list of border
        nodes.

        .. versionadded:: 0.0.7

        Returns
        -------
        Dict[int, List[int]]
        '''
        border_nodes = defaultdict(list)
        for node in self:
            border_nodes[node.group].append(node.Index)
        return dict(border_nodes)


@dataset()
class PlaceData(PointData):
    '''
    Class for representing place data.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Data frame.
    crs : int
        EPSG code of CRS in which node coordinates are represented.
    '''

    FIELDS = (
        Field('name', 'object', True),
        Field('x', 'float64', True, False),
        Field('y', 'float64', True, False),
        Field('z', 'float64', False),
        Field('group', 'uint32', False, default=-1)
    )

    def area_nodes(self, nodes: NodeData, radius: float) -> Dict[int, Set[int]]:
        '''
        Return dictionary mapping region ID to set of area nodes.

        .. versionadded:: 0.0.7

        Parameters
        ----------
        nodes : :class:`NodeData`
            Node data.
        radius : float
            Place radius. All points within this radius of a place are
            added to the set of nodes of the corresponding place group.

        Returns
        -------
        Dict[int, Set[int]]
        '''
        tree = cKDTree(nodes.coords(2))
        neighbors = tree.query_ball_point(self.coords(2), radius)
        area_nodes = defaultdict(set)
        for group_id, group_members in self.groups().items():
            for place_id in group_members:
                area_nodes[group_id].update(neighbors[place_id])
        return dict(area_nodes)

    def groups(self) -> Dict[int, Set[int]]:
        '''
        Return dictionary mapping place ID to group ID.

        .. versionadded:: 0.0.7

        Returns
        -------
        Dict[int, Set[int]]
        '''
        groups = defaultdict(set)
        for place in self:
            groups[place.group].add(place.Index)
        return dict(groups)

    @classmethod
    def from_csvs(cls, *paths: str, crs: int) -> 'PlaceData':
        '''
        Read place data from multiple CSV files.

        Parameters
        ----------
        *paths : str
            Paths to CSV files.
        crs : int
            EPSG code of CRS in which place coordinates are represented.

        Returns
        -------
        :class:`PlaceData`
            Place data.
        '''
        if len(paths) == 1:
            return cls.from_csv(paths[0], crs)
        df = pd.concat([pd.read_csv(path) for path in paths])
        _, indices = np.unique(df[['x', 'y']].to_numpy(dtype=float),
                               return_index=True)
        df = df.iloc[indices]
        df = df.reset_index(drop=True)
        return cls(df, crs)


@dataset()
class AreaData(Dataset):
    '''
    Class for representing area data.

    .. versionadded:: 0.0.7

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Data frame.
    crs : int
        EPSG code of CRS in which node coordinates are represented.
    '''

    FIELDS = (
        Field('coords', 'object', True, False),
    )

    @require_qgis
    def features(self, task: QgsTask) -> Generator[QgsFeature, None, None]:
        '''
        Generate features for insertion into a vector layer.

        Parameters
        ----------
        task : :class:`~qgis.core.QgsTask`
            Task for rendering features.

        Yields
        ------
        :class:`~qgis.core.QgsFeature`
        '''
        num_rows = len(self)
        includes = [index for index, field in enumerate(self.FIELDS, 1)
                    if field.include]
        for i, item in enumerate(self):
            task.setProgress(i/num_rows*100)
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPolygonXY([
                [QgsPointXY(x, y) for (x, y) in item.coords]]))
            feat.setAttributes(
                [item.Index] + [item[index] for index in includes])
            yield feat


@dataset()
class ConnectionData(Dataset):
    '''
    Class for representing a dataset of connections between points.

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        Data frame containing connection data. Required columns are 'i',
        'j', and 'tag'. Optional columns are 'length' and 'coords'.
    crs : int or None, optional
        EPSG code of CRS in which connection coordinates are
        represented. None if coordinates are not stored in the dataset.
        The default is None.
    directed : bool
        Whether connections are directed.
    '''

    FIELDS = (
        Field('i', 'uint32', True),
        Field('j', 'uint32', True),
        Field('tag', 'category', True),
        Field('length', 'float64', False),
        Field('coords', 'object', False, False)
    )

    def __init__(self, df: pd.DataFrame, crs: int = None, *, directed: bool
                 ) -> None:
        super().__init__(df, crs)
        self._directed = directed

    def actions(self) -> Dict[int, List[Action]]:
        '''
        Return dictionary mapping point to actions. Actions are
        represented by a 2-tuple containing the connection and
        destination IDs.

        Returns
        -------
        actions : Dict[int, List[Action]]
        '''
        actions = defaultdict(list)
        if self.directed:
            for connection in self:
                actions[connection.i].append((connection.Index, connection.j))
        else:
            for connection in self:
                actions[connection.i].append((connection.Index, connection.j))
                actions[connection.j].append((connection.Index, connection.i))
        return dict(actions)

    @abstractmethod
    def coords(self, dims: int = None) -> np.ndarray:
        '''
        Return array of connection coordinates.

        Parameters
        ----------
        dims : {2, 3} or None, optional
            Whether to return two- or three-dimensional coordinates. If
            None, then the :attr:`dims` attribute is used. The default
            is None.

        Returns
        -------
        :class:`~numpy.ndarray`
            Connection coordinates.

        Raises
        ------
        DimensionError
            If `dims` exceeds the :attr:`dims` attribute.
        '''
        pass

    def densify(self, interval: float) -> None:
        '''
        Densify connections in the dataset.

        Parameters
        ----------
        interval : float
            Interval between densified points.
        '''
        self._df['coords'] = list(map(partial(densify, interval=interval),
                                      self.coords(2)))

    @property
    def dims(self) -> Union[int, None]:
        '''
        Number of dimensions stored in coordinates. None if coordinates
        are not stored in this dataset.

        Returns
        -------
        int or None
        '''
        if 'coords' in self._active_columns():
            return self._df['coords'].iloc[0].shape[1]
        else:
            return

    @property
    def directed(self) -> bool:
        '''
        Whether connections are directed.

        Returns
        -------
        bool
        '''
        return self._directed

    @abstractmethod
    def elevate(self, *paths: str, r: float = 1e-3, p: int = 2) -> None:
        '''
        Compute elevations and update :math:`z`-coordinates.

        Parameters
        ----------
        *paths : str
            Path(s) to TIF file(s) storing elevation data.
        r : float, optional
            Search radius for nearby points in degrees. The default is
            0.001.
        p : int, optional
            Power setting for IDW interpolation. The default is 2.

        See also
        --------
        :meth:`flatten`
            Remove elevations from all coordinates.
        :func:`idw_query`
            Query elevations of two-dimensional points via IDW
            interpolation.
        '''
        pass

    @require_qgis
    def features(self, task: QgsTask) -> Generator[QgsFeature, None, None]:
        '''
        Generate features for insertion into a vector layer.

        Parameters
        ----------
        task : :class:`~qgis.core.QgsTask`
            Task for rendering features.

        Yields
        ------
        :class:`~qgis.core.QgsFeature`
        '''
        num_rows = len(self)
        includes = [index for index, field in enumerate(self.FIELDS, 1)
                    if field.include]
        for i, item in enumerate(self):
            task.setProgress(i/num_rows*100)
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPolylineXY(
                [QgsPointXY(x, y) for (x, y) in item.coords]))
            feat.setAttributes(
                [item.Index] + [item[index] for index in includes])
            yield feat

    def flatten(self) -> None:
        '''
        Remove :math:`z`-coordinates from all connections.

        See also
        --------
        :meth:`elevate`
            Compute elevations and update :math:`z`-coordinates.
        '''
        xy = self.coords(2)
        self._df['coords'] = list(xy)
        self._df['length'] = list(map(polyline_length, xy))

    def get(self, i: int = None, j: int = None) -> Iterable[tuple]:
        '''
        Retrieve from the dataset a specific connection, :math:`(i, j)`,
        connections starting from point :math:`i`, or connections ending
        at point :math:`j`.

        Parameters
        ----------
        i, j : int or None, optional
            Start and end point ids. If `i` and `j` are both None, then
            a randomly chosen connection is returned.

        Returns
        -------
        Iterable[tuple]
        '''
        if i is None and j is None:
            ids = [np.random.choice(len(self))]
        elif i is not None and j is not None:
            ar1 = np.flatnonzero(self._df['i'] == i)
            ar2 = np.flatnonzero(self._df['j'] == j)
            ids = np.intersect1d(ar1, ar2, True)
        elif i is not None:
            ids = np.flatnonzero(self._df['i'] == i)
        elif j is not None:
            ids = np.flatnonzero(self._df['j'] == j)
        return self._df.iloc[ids].itertuples(name=self._ELEMENT_NAME)

    def info(self, sep: str = '\n', ret: bool = False) -> None:
        '''
        Print dataset information.

        Parameters
        ----------
        sep : str, optional
            Separator between lines. The default is '\n'.
        ret : bool, optional
            If True, the information is returned as a string. Otherwise,
            it is printed. The default is False.

        Returns
        -------
        info : str
            If True, then the information is returned.
        '''
        coords = np.vstack(self.coords(2))
        xmin, ymin = np.min(coords, axis=0)
        xmax, ymax = np.max(coords, axis=0)
        info = sep.join([
            str(self.__class__),
            f'Count: {len(self):,}',
            f'CRS: EPSG:{self.crs}',
            f'dims: {self.dims}',
            f'xmin: {xmin:.07f}',
            f'ymin: {ymin:.07f}',
            f'xmax: {xmax:.07f}',
            f'ymax: {ymax:.07f}'
        ])
        if ret:
            return info
        else:
            print(info)

    @abstractmethod
    def mask(self, xmin: float = None, ymin: float = None, xmax: float = None,
             ymax: float = None) -> np.ndarray:
        '''
        Return mask.

        Parameters
        ----------
        xmin, ymin, xmax, ymax : float or None, optional
            Minimum and maximum :math:`x`- and :math:`y`-coordinates.
            The defaults are None.

        Returns
        -------
        mask : :class:`~numpy.ndarray`
            Masked array.
        '''
        pass

    def neighbors(self) -> Dict[int, Set[int]]:
        '''
        Return dictionary mapping point to neighboring points.

        Returns
        -------
        Dict[int, Set[int]]
            Dictionary mapping point to set of neighboring points.
        '''
        neighbors = defaultdict(set)
        if self.directed:
            for (i, j) in self.pairs():
                neighbors[i].add(j)
        else:
            for (i, j) in self.pairs():
                neighbors[i].add(j)
                neighbors[j].add(i)
        return dict(neighbors)

    def pairs(self) -> np.ndarray:
        '''
        Return array of :math:`(i, j)` pairs.

        Returns
        -------
        :class:`~numpy.ndarray`, shape (N, 2)
            Array of :math:`(i, j)` pairs.
        '''
        return self._df[['i', 'j']].to_numpy(dtype=int)

    @abstractmethod
    def transform(self, dst: int) -> None:
        '''
        Transform coordinates to another CRS.

        Parameters
        ----------
        dst : int
            EPSG code of the destination CRS.
        '''
        pass


@dataset()
class LinkData(ConnectionData):

    def coords(self, dims: int = None) -> np.ndarray:
        '''
        Return link coordinates.

        Parameters
        ----------
        dims : {2, 3} or None, optional
            Whether to return two- or three-dimensional coordinates. If
            None, then the :attr:`dims` attribute is used. The default
            is None.

        Returns
        -------
        :class:`~numpy.ndarray`, shape (N, 2, M)
            Link coordinates. :math:`N` is the number of links in the
            dataset, and :math:`M` is the number of coordinate
            dimensions.

        Raises
        ------
        DimensionError
            If `dims` exceeds the :attr:`dims` attribute.
        '''
        if self.dims:
            coords = np.vstack(self._df['coords']).reshape(-1, 2, self.dims)
            if (dims is None) or (dims == self.dims):
                return coords
            elif dims == 2:
                return coords[:, :, :2]
            raise DimensionError(2, 3)
        raise DimensionError(0, dims)

    def elevate(self, *paths: str, r: float = 1e-3, p: int = 2) -> None:
        '''
        Compute :math:`z`-coordinates and update data frame.

        Parameters
        ----------
        *paths : str
            Path(s) to TIF file(s) storing elevation data.
        r : float, optional
            Search radius for nearby points in degrees. The default is
            0.001.
        p : int, optional
            Power setting for IDW interpolation. The default is 2.

        See also
        --------
        :meth:`flatten`
            Remove :math:`z`-coordinates from all links.
        :func:`idw_query`
            Query elevations of two-dimensional points via IDW
            interpolation.
        '''
        xyz = idw_query(np.vstack(self.coords(2)), self.crs, *paths, r=r, p=p,
                        return_xy=True).reshape(-1, 2, 3)
        self._df['coords'] = list(xyz)
        self._df['length'] = list(map(polyline_length, xyz))

    def mask(self, xmin: float = None, ymin: float = None, xmax: float = None,
             ymax: float = None) -> np.ndarray:
        '''
        Return mask.

        Parameters
        ----------
        xmin, ymin, xmax, ymax : float or None, optional
            Minimum and maximum :math:`x`- and :math:`y`-coordinates.
            The defaults are None.

        Returns
        -------
        mask : :class:`~numpy.ndarray`
            Masked array.
        '''
        coords = self.coords(2).reshape(-1, 2)
        mask = np.full(2 * len(self), True)
        if xmin:
            mask = mask & (coords[:, 0] > xmin)
        if ymin:
            mask = mask & (coords[:, 1] > ymin)
        if xmax:
            mask = mask & (coords[:, 0] < xmax)
        if ymax:
            mask = mask & (coords[:, 1] < ymax)
        mask = mask.reshape(-1, 2)
        return np.all(mask, axis=1)

    def transform(self, dst: int) -> None:
        '''
        Transform link coordinates to another CRS.

        Parameters
        ----------
        dst : int
            EPSG code of the destination CRS.
        '''
        transformed = transform_coords(
            np.vstack(self._df['coords']), src=self.crs, dst=dst
        ).reshape(-1, 2, self.dims)
        self._df['coords'] = list(transformed)
        self._df['length'] = list(map(polyline_length, transformed))
        self._crs = dst


@dataset()
class EdgeData(ConnectionData):

    def coords(self, dims: int = None) -> np.ndarray:
        '''
        Return edge coordinates.

        Parameters
        ----------
        dims : {2, 3} or None, optional
            Whether to return two- or three-dimensional coordinates. If
            None, then the :attr:`dims` attribute is used. The default
            is None.

        Returns
        -------
        :class:`numpy.ndarray`, shape (N,)
            Edge coordinates. :math:`N` is the number of edges in the
            dataset.

        Raises
        ------
        DimensionError
            If `dims` exceeds the :attr:`dims` attribute.
        '''
        if self.dims:
            coords = self._df['coords'].to_numpy(dtype='object')
            if (dims is None) or (dims == self.dims):
                return coords
            elif dims == 2:
                return np.array([coords_[:, :2] for coords_ in coords],
                                dtype='object')
            raise DimensionError(2, 3)
        raise DimensionError(0, dims)

    def elevate(self, *paths: str, r: float = 1e-3, p: int = 2) -> None:
        '''
        Compute :math:`z`-coordinates and update data frame.

        Parameters
        ----------
        *paths : str
            Path(s) to TIF file(s) storing elevation data.
        r : float, optional
            Search radius for nearby points in degrees. The default is
            0.001.
        p : int, optional
            Power setting for IDW interpolation. The default is 2.

        See also
        --------
        :meth:`flatten`
            Remove :math:`z`-coordinates from all edges.
        :func:`idw_query`
            Query elevations of two-dimensional points via IDW
            interpolation.
        '''
        xyz = idw_query(np.vstack(self.coords(2)), self.crs, *paths, r=r, p=p,
                        return_xy=True)
        coords = []
        for length in map(len, self._df['coords']):
            coords.append(xyz[:length])
            xyz = xyz[length:]
        self._df['coords'] = coords
        self._df['length'] = list(map(polyline_length, coords))

    def mask(self, xmin: float = None, ymin: float = None, xmax: float = None,
             ymax: float = None) -> np.ndarray:
        '''
        Return mask.

        Parameters
        ----------
        xmin, ymin, xmax, ymax : float or None, optional
            Minimum and maximum :math:`x`- and :math:`y`-coordinates.
            The defaults are None.

        Returns
        -------
        mask : :class:`~numpy.ndarray`
            Masked array.
        '''
        coords = np.vstack(self.coords(2))
        bools = np.full(len(coords), True)
        if xmin:
            bools = bools & (coords[:, 0] > xmin)
        if ymin:
            bools = bools & (coords[:, 1] > ymin)
        if xmax:
            bools = bools & (coords[:, 0] < xmax)
        if ymax:
            bools = bools & (coords[:, 1] < ymax)
        mask = []
        for length in map(len, self._df['coords']):
            mask.append(np.all(bools[:length]))
            bools = bools[length:]
        return np.array(mask)

    def transform(self, dst: int) -> None:
        '''
        Transform edge coordinates to another CRS.

        Parameters
        ----------
        dst : int
            EPSG code of the destination CRS.
        '''
        transformed = transform_coords(
            np.vstack(self._df['coords']), src=self.crs, dst=dst)
        coords = []
        for length in map(len, self._df['coords']):
            coords.append(transformed[:length])
            transformed = transformed[length:]
        self._df['coords'] = list(coords)
        self._df['length'] = list(map(polyline_length, coords))
        self._crs = dst

    def weights(self) -> Dict[int, Dict[int, float]]:
        '''
        Return dictionary whose keys are source nodes and values are
        a mapping from destination nodes to corresponding edge weights.

        .. versionadded:: 0.0.7

        Returns
        -------
        Dict[int, Dict[int, float]]
        '''
        weights = defaultdict(lambda: defaultdict(float))
        if self.directed:
            for edge in self:
                weights[edge.i][edge.j] = edge.length
        else:
            for edge in self:
                weights[edge.i][edge.j] = edge.length
                weights[edge.j][edge.i] = edge.length
        for i in weights.keys():
            weights[i] = dict(weights[i])
        return dict(weights)
