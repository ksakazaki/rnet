from collections import Counter, defaultdict
from itertools import chain
import pickle
from typing import Dict, List, Set
import os
import numpy as np
import pandas as pd
from osgeo import ogr
from rnet.dataset import VertexData, LinkData, NodeData, EdgeData
from rnet.geometry import polyline_length


__all__ = ['Model', 'model', 'read_osm', 'read_osms', 'simplify']


_OSM_HIERARCHY = {
    'living_street' : 0,
    'residential'   : 1,
    'unclassified'  : 2,
    'tertiary_link' : 3,
    'tertiary'      : 4,
    'secondary_link': 5,
    'secondary'     : 6,
    'primary_link'  : 7,
    'primary'       : 8,
    'trunk_link'    : 9,
    'trunk'         : 10,
    'motorway_link' : 11,
    'motorway'      : 12
    }


_OSM_TAGS = set(_OSM_HIERARCHY)


def read_osms(*paths: str, crs: int = 4326, return_vertices: bool = False,
              return_links: bool = False, layer_name: str = 'lines',
              exclude: List[str] = []):
    '''
    Read multiple OSM files.

    Parameters
    ----------
    *paths : str
        OSM file paths.
    crs : int, optional
        EPSG code of desired CRS for point coordinates. OSM features are
        represented in EPSG:4326. The default is 4326. If another EPSG code is
        given, then point coordinates are transformed.
    return_vertices : bool, optional
        If True, also return vertex data. Vertices are two-dimensional points
        used to define the geometry of each road. The default is False.
    return_links : bool, optional
        If True, also return link data. Links are line segments defined by
        unordered pairs of vertices. The default is False.
    
    Other parameters
    ----------------
    layer_name : str, optional
        Name of the layer from which OSM features are read. The default is
        'lines'.
    exclude : List[str], optional
        Tags to exclude. By default, the following tags are included:
        'living_street', 'residential', 'unclassified',  'tertiary_link',
        'tertiary', 'secondary_link', 'secondary', 'primary_link', 'primary',
        'trunk_link', 'trunk', 'motorway_link', 'motorway'.

    Returns
    -------
    nodes : :class:`NodeData`
        Nodes extracted from the OSM files.
    edges : :class:`EdgeData`
        Directed edges extracted from the OSM files.
    vertices : :class:`VertexData`, optional
        Vertices extracted from the OSM files. Only provided if
        `return_vertices` is True.
    links : :class:`LinkData`, optional
        Undirected links extracted from the OSM files. Only provided if
        `return_links` is True.

    See also
    --------
    :func:`read_osm`
        Read a single OSM file.

    References
    ----------
    https://wiki.openstreetmap.org/wiki/Key:highway
    '''
    driver = ogr.GetDriverByName('OSM')
    keep_tags = _OSM_TAGS.difference(exclude)

    # Read files
    all_points = []
    all_tags = []
    seen = set()
    for fp in paths:
        source = driver.Open(fp)
        for feat in source.GetLayer(layer_name):
            fid = feat.GetFID()
            if fid in seen:
                continue
            else:
                seen.add(fid)
            tag = feat.GetField('highway')
            if tag in keep_tags:
                all_points.append(feat.GetGeometryRef().GetPoints())
                all_tags.append(tag)

    # Extract vertices
    vertices, inv = np.unique(np.concatenate(all_points), axis=0, return_inverse=True)
    vertices = VertexData(pd.DataFrame(vertices, columns=['x', 'y']), 4326)
    if crs != 4326:
        vertices.transform(crs)

    # Extract links
    links = []
    for length, tag in zip(map(len, all_points), all_tags):
        links.append(zip(inv[:length-1], inv[1:length], [tag]*length))
        inv = inv[length:]
    links = LinkData(
        pd.DataFrame(chain.from_iterable(links), columns=['i', 'j', 'tag']),
        directed=False)

    # Extract nodes
    nodes_ = np.sort(
        [k for k, v in Counter(list(links.pairs().flatten())).items() if v != 2]
        )
    nodes = NodeData(vertices._df.iloc[nodes_], crs)

    # Extract edges
    nodes_ = set(nodes_)
    actions = links.actions()
    vseqs = []  # vertex sequences
    lseqs = []  # link sequences
    for i in nodes_:
        for action, j in actions[i]:
            vseqs.append([i, j])
            lseqs.append([action])
            while True:
                actions_ = actions[vseqs[-1][-1]]
                if len(actions_) == 2:
                    try:
                        action_ = actions_[0]
                        assert action_[0] != lseqs[-1][-1]
                    except AssertionError:
                        action_ = actions_[1]
                    finally:
                        vseqs[-1].append(action_[1])
                        lseqs[-1].append(action_[0])
                else:
                    break
    i = [vseq[0] for vseq in vseqs]
    j = [vseq[-1] for vseq in vseqs]
    tags = links._df['tag'].iloc[[lseq[0] for lseq in lseqs]]
    vcoords = vertices.coords(2)[list(chain.from_iterable(vseqs))]
    coords = []
    for length in map(len, vseqs):
        coords.append(vcoords[:length])
        vcoords = vcoords[length:]
    lengths = list(map(polyline_length, coords))
    edges = EdgeData(
        pd.DataFrame(zip(i, j, tags, lengths, coords),
                     columns=['i', 'j', 'tag', 'length', 'coords']),
        crs, directed=True)

    # Re-index nodes
    nodes._df = nodes._df.reset_index(drop=True)
    _, inverse = np.unique(edges.pairs().flatten(), return_inverse=True)
    edges._df[['i', 'j']] = inverse.reshape(-1,2)

    # Return
    out = [nodes, edges]
    if return_vertices:
        out.append(vertices)
    if return_links:
        coords = vertices.coords(2)[links.pairs().flatten()].reshape(-1,2,2)
        links._df['coords'] = list(coords)
        links._df['length'] = list(map(polyline_length, coords))
        links._crs = crs
        out.append(links)
    for dataset in out:
        dataset.reset_dtypes()
    return tuple(out)


def read_osm(path: str, *, crs: int = 4326, return_vertices: bool = False,
             return_links: bool = False, layer_name: str = 'lines',
             exclude: List[str] = []):
    '''
    Read a single OSM file.

    Parameters
    ----------
    path : str
        OSM file path.
    crs : int, optional
        EPSG code of desired CRS for point coordinates. OSM features are
        represented in EPSG:4326. The default is 4326. If another EPSG code is
        given, then point coordinates are transformed.
    return_vertices : bool, optional
        If True, also return vertex data. Vertices are two-dimensional points
        used to define the geometry of each road. The default is False.
    return_links : bool, optional
        If True, also return link data. Links are line segments defined by
        unordered pairs of vertices. The default is False.
    
    Other parameters
    ----------------
    layer_name : str, optional
        Name of the layer from which OSM features are read. The default is
        'lines'.
    exclude : List[str], optional
        Tags to exclude. By default, the following tags are included:
        'living_street', 'residential', 'unclassified',  'tertiary_link',
        'tertiary', 'secondary_link', 'secondary', 'primary_link', 'primary',
        'trunk_link', 'trunk', 'motorway_link', 'motorway'.

    Returns
    -------
    nodes : :class:`NodeData`
        Nodes extracted from the OSM files.
    edges : :class:`EdgeData`
        Directed edges extracted from the OSM files.
    vertices : :class:`VertexData`, optional
        Vertices extracted from the OSM files. Only provided if
        `return_vertices` is True.
    links : :class:`LinkData`, optional
        Undirected links extracted from the OSM files. Only provided if
        `return_links` is True.

    See also
    --------
    :func:`read_osms`
        Read multiple OSM files.

    References
    ----------
    https://wiki.openstreetmap.org/wiki/Key:highway
    '''
    return read_osms(path, crs=crs, return_vertices=return_vertices,
                     return_links=return_links, layer_name=layer_name,
                     exclude=exclude)


class Model:
    '''
    Class representing an RNet model.

    Parameters
    ----------
    nodes : :class:`NodeData`
        Node data.
    edges : :class:`EdgeData`
        Edge data.
    vertices : :class:`VertexData` or None, optional
        Vertex data.
    links : :class:`LinkData` or None, optional
        Link data.
    
    See also
    --------
    :func:`model`
        Construct a model from multiple data sources.
    :func:`simplify`
        Return simplified model.
    '''

    def __init__(self, nodes, edges, vertices=None, links=None):
        self.nodes = nodes
        self.edges = edges
        self.vertices = vertices
        self.links = links

    @property
    def crs(self) -> int:
        '''
        EPSG code of model CRS.

        Returns
        -------
        int
        '''
        raise NotImplementedError

    @classmethod
    def from_pickle(cls, path_to_pickle: str) -> 'Model':
        '''
        Read model from pickled representation.
        
        Parameters
        ----------
        path_to_pickle : str
            File path.
        
        Return
        ------
        :class:`Model`

        See also
        --------
        :meth:`to_pickle`
            Write model to disk in pickled representation.
        '''
        with open(path_to_pickle, "rb") as f:
            model = pickle.load(f)
        return model

    def densify(self, interval: float):
        '''
        Densify connections.

        Parameters
        ----------
        interval : float
            Interval between densified points.
        '''
        self.edges.densify(interval)
        if self.links is not None:
            self.links.densify(interval)

    @property
    def dims(self) -> int:
        '''
        Number of coordinate dimensions.

        Returns
        -------
        int
        '''
        raise NotImplementedError

    def elevate(self, *paths, r: float = 1e-3, p: int = 2):
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
        '''
        self.nodes.elevate(*paths, r=r, p=p)
        self.edges.elevate(*paths, r=r, p=p)

    def to_pickle(self, path_to_pickle: str) -> None:
        '''
        Write model to disk in pickled representation.

        Parameters
        ----------
        path_to_pickle : str
            File path.

        See also
        --------
        :meth:`from_pickle`
            Read model from pickled representation.
        '''
        with open(path_to_pickle, "wb") as f:
            pickle.dump(self, f)

    def transform(self, dst: int) -> None:
        '''
        Transform coordinates to another CRS.

        Parameters
        ----------
        dst : int
            EPSG code of destination CRS.
        '''
        self.nodes.transform(dst)
        self.edges.transform(dst)
        if self.vertices is not None:
            self.vertices.transform(dst)
        if self.links is not None:
            self.links.transform(dst)


def model(*paths, crs: int = 4326, keep_vertices: bool = False,
          keep_links: bool = False, layer_name: str = 'lines',
          exclude: List[str] = [], r: float = 1e-3, p: int = 2) -> Model:
    '''
    Construct a model from multiple data sources.

    Parameters
    ----------
    *paths : str
        A single directory path or multiple file paths. The following
        file types are supported:

            * OSM files containing street map data
            * TIF files containing elevation data

    crs : int, optional
        EPSG code of model CRS. The default is 4326.

    Other parameters
    ----------------
    keep_vertices, keep_links : bool, optional
        If True, then the sets of vertices and links are retained in the
        model. The defaults are False.
    layer_name : str, optional
        Name of the layer from which OSM features are read. The default
        is 'lines'.
    exclude : List[str], optional
        Tags to exclude when reading OSM files. By default, the
        following tags are included: 'living_street', 'residential',
        'unclassified', 'tertiary_link', 'tertiary', 'secondary_link',
        'secondary', 'primary_link', 'primary', 'trunk_link', 'trunk',
        'motorway_link', 'motorway'.
    r : float, optional
        Search radius for nearby points in degrees, used for querying
        elevations via IDW interpolation. The default is 0.001.
    p : int, optional
        Power setting for IDW interpolation. The default is 2.
    
    Returns
    -------
    :class:`Model`
    '''
    # Gather file paths
    unpacked = []
    for path in paths:
        if os.path.isfile(path):
            unpacked.append(path)
        elif os.path.isdir(path):
            unpacked.extend([os.path.join(path, n) for n in os.listdir(path)])
        else:
            continue
    sorted = defaultdict(list)
    for path in unpacked:
        sorted[os.path.splitext(path)[1]].append(path)

    # Gather map data
    if sorted['.osm']:
        nodes, edges, vertices, links = read_osms(
            *sorted['.osm'], crs=crs, layer_name=layer_name, exclude=exclude,
            return_vertices=True, return_links=True)
    else:
        nodes = None
        edges = None
        vertices = None
        links = None

    # Calculate elevations
    if sorted['.osm'] and sorted['.tif']:
        nodes.elevate(*sorted['.tif'], r=r, p=p)
        edges.elevate(*sorted['.tif'], r=r, p=p)
        if keep_vertices:
            vertices.elevate(*sorted['.tif'], r=r, p=p)
        if keep_links:
            links.elevate(*sorted['.tif'], r=r, p=p)

    if not keep_vertices:
        vertices = None
    if not keep_links:
        links = None
    return Model(nodes, edges, vertices, links)


def _ccl(neighbors: Dict[int, Set[int]]) -> List[Set[int]]:
    '''
    Clustering via connected-component labeling.
    
    Parameters
    ----------
    neighbors : Dict[int, Set[int]]
        Dictionary mapping node to its neighbors.
    
    Returns
    -------
    List[Set[int]]
        List of connected components.
    '''
    clusters = []
    seen = set()
    for s in neighbors:
        if s in seen:
            continue
        clusters.append({s})
        seen.add(s)
        queue = [s]
        while len(queue) > 0:
            m = queue.pop(0)
            for n in neighbors[m]:
                if n in seen:
                    continue
                clusters[-1].add(n)
                seen.add(n)
                queue.append(n)
    return clusters


def simplify(model: Model, *, xmin: float = None, ymin: float = None,
             xmax: float = None, ymax: float = None) -> Model:
    '''
    Return simplified model.

    Parameters
    ----------
    model : :class:`Model`
        Original model.
    xmin, ymin, xmax, ymax : float or None, optional
        Bounds within which features are kept.

    Returns
    -------
    :class:`Model`
        Simplified model.

    Examples
    --------
    >>> m = rn.model("shinjuku.osm")
    >>> n = rn.simplify(m, xmin=139.7)
    '''
    nodes = NodeData(
        model.nodes._df.iloc[model.nodes.mask(xmin, ymin, xmax, ymax)],
        model.nodes._crs
        )
    edges = EdgeData(
        model.edges._df.iloc[model.edges.mask(xmin, ymin, xmax, ymax)],
        model.edges._crs,
        directed=model.edges._directed
        )
    if model.vertices is not None:
        vertices = VertexData(
            model.vertices._df.iloc[model.vertices.mask(xmin, ymin, xmax, ymax)],
            model.vertices._crs
        )
    else:
        vertices = None
    if model.links is not None:
        links = LinkData(
            model.links._df.iloc[model.links.mask(xmin, ymin, xmax, ymax)],
            model.links._crs,
            directed=model.links._directed
        )
    else:
        links = None
    return Model(nodes, edges, vertices, links)
