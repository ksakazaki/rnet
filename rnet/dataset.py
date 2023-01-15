from abc import ABC, abstractmethod
from collections import defaultdict
import dataclasses
from dataclasses import dataclass
from functools import partial
import sys
from typing import Dict, List, Set, Tuple, Union
import numpy as np
import pandas as pd
from rnet.coordinates import transform_coords, idw_query, densify
from rnet.geometry import polyline_length


Action = Tuple[int, int]  # (connection_id, destination_id)


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
        self._df = validate(df, self._BASE_ELEMENT)
        self._crs = crs

    def __contains__(self, id_: int) -> bool:
        return id_ in self._df.index

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: {len(self):,} rows (EPSG:{self._crs})>'

    def __iter__(self):
        return self._df.itertuples(name=self._BASE_ELEMENT.__name__)

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
        return self._df[active_columns(self._df, self._BASE_ELEMENT)]

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


def dataset(base: dataclass, layer_name: str = None):
    '''
    Decorator for setting class attributes of a dataset.

    Parameters
    ----------
    base : :class:`dataclasses.dataclass`
        Data class representing base element. This data class is used to
        validate the data frame with which the :class:`Dataset` instance
        is initialized. Namely, data class fields without default values
        are required, and those with default values are optional.
    name : str or None, optional
        Layer name used when rendering features.
    '''
    def decorate(cls):
        cls._BASE_ELEMENT = base
        if layer_name:
            cls._LAYER_NAME = base.__name__.lower() + 's'
        else:
            cls._LAYER_NAME = layer_name
        return cls
    return decorate


def validate(df: pd.DataFrame, base: dataclass) -> pd.DataFrame:
    '''
    Ensure that a data frame has all required columns.

    Required columns are the fields in `base` that do not have a default
    value. If a required column is missing, then an error is raised.

    Optional columns are the fields in `base` that have a default value.
    If an optional column is missing, then it is added to `df`.

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        Data frame to be validated.
    base : :class:`~dataclasses.dataclass`
        Base dataclass.

    Returns
    -------
    df : :class:`pandas.DataFrame`
    
    Raises
    ------
    MissingColumnError
        If a required column is missing.
    '''
    # TODO: raise error if crs is not given for coordinates
    for field in dataclasses.fields(base):
        if field.name in df.columns:
            continue
        elif isinstance(field.default, dataclasses._MISSING_TYPE):
            raise MissingColumnError(field.name)
        else:
            df[field.name] = field.default
    return df


def active_columns(df: pd.DataFrame, base: dataclasses.dataclass) -> List[str]:
    '''
    Return names of active columns in a data frame.

    A column is active if (a) it is a required column, or (b) it is an
    optional column containing non-default values.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Data frame.
    base : :class:`dataclasses.dataclass`
        Base data class.

    Returns
    -------
    List[str]
    '''
    cols = []
    for field in dataclasses.fields(base):
        if isinstance(field.default, dataclasses._MISSING_TYPE):
            cols.append(field.name)
        elif np.all(df[field.name].to_numpy() == field.default):
            continue
        else:
            cols.append(field.name)
    return cols


@dataclass
class Point:
    '''
    Class representing a two- or three-dimensional point.

    Parameters
    ----------
    x, y : float
        :math:`x`- and :math:`y`-coordinates.
    z : float, optional
        :math:`z`-coordinate. The default is None.
    '''
    x: float
    y: float
    z: float = None


@dataset(Point)
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
        if 'z' in active_columns(self._df, self._BASE_ELEMENT):
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

    def flatten(self):
        '''
        Remove :math:`z`-coordinate from all points.

        See also
        --------
        :meth:`elevate`
            Compute :math:`z`-coordinates and update data frame.
        '''
        self._df['z'] = None

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


@dataclass
class Vertex(Point):
    '''
    Class representing a two- or three-dimensional vertex.

    Vertices are the points along each road that define their
    geometries.

    Parameters
    ----------
    x, y : float
        :math:`x`- and :math:`y`-coordinates.
    z : float, optional
        :math:`z`-coordinate. The default is None.
    '''
    pass


@dataset(Vertex, 'vertices')
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


@dataclass
class Node(Point):
    '''
    Class representing a two- or three-dimensional node.

    Nodes represent intersections and dead-ends in the road network
    model.

    Parameters
    ----------
    x, y : float
        :math:`x`- and :math:`y`-coordinates.
    z : float, optional
        :math:`z`-coordinate. The default is None.
    '''
    pass


@dataset(Node)
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


@dataclass
class Connection:
    i: int
    j: int
    tag: str
    length: float = None
    coords: np.ndarray = None


@dataset(Connection)
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
        if 'coords' in active_columns(self._df, self._BASE_ELEMENT):
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

    def get(self, i: int = None, j: int = None
            ) -> Union[Connection, List[Connection]]:
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
        :class:`Connection` or List[:class:`Connection`]
            A :class:`Connection` or list of :class:`Connections`. An
            empty list is returned if no corresponding connections are
            found.
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
        out = []
        for id_ in ids:
            element = self._BASE_ELEMENT(*self._df.iloc[id_])
            element.Index = id_
            out.append(element)
        return out[0] if len(out) == 1 else out

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


@dataclass
class Link(Connection):
    pass


@dataset(Link)
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
            coords = np.vstack(self._df['coords']).reshape(-1,2,self.dims)
            if (dims is None) or (dims == self.dims):
                return coords
            elif dims == 2:
                return coords[:,:,:2]
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
                        return_xy=True).reshape(-1,2,3)
        self._df['coords'] = list(xyz)
        self._df['length'] = list(map(polyline_length, xyz))

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
            ).reshape(-1,2,self.dims)
        self._df['coords'] = list(transformed)
        self._df['length'] = list(map(polyline_length, transformed))
        self._crs = dst


@dataclass
class Edge(Connection):
    pass


@dataset(Edge)
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
                return np.array([coords_[:,:2] for coords_ in coords],
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
