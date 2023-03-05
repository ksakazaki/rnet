from collections import Counter, defaultdict
from itertools import chain, count
import pickle
from typing import List, Set, Tuple
import os
import numpy as np
import pandas as pd
from osgeo import ogr
from scipy.spatial import cKDTree
from rnet.algorithms import ccl
from rnet.dataset import Dataset, VertexData, LinkData, NodeData, EdgeData, PlaceData, AreaData
from rnet.geometry import Circle, outer_arcs, polyline_length


__all__ = ['Model', 'model', 'read_osm', 'read_osms', 'simplify']


_OSM_HIERARCHY = {
    'living_street': 0,
    'residential': 1,
    'unclassified': 2,
    'tertiary_link': 3,
    'tertiary': 4,
    'secondary_link': 5,
    'secondary': 6,
    'primary_link': 7,
    'primary': 8,
    'trunk_link': 9,
    'trunk': 10,
    'motorway_link': 11,
    'motorway': 12
}


_OSM_TAGS = set(_OSM_HIERARCHY)


def read_osms(*paths: str, crs: int = 4326, layer_name: str = 'lines',
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
    vertices : :class:`VertexData`
        Vertices extracted from the OSM files.
    links : :class:`LinkData`
        Undirected links extracted from the OSM files.

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
    vertices, inv = np.unique(
        np.concatenate(all_points), axis=0, return_inverse=True)
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
        crs, directed=False)
    coords = vertices.coords(2)[links.pairs().flatten()].reshape(-1, 2, 2)
    links._df['coords'] = list(coords)
    links._df['length'] = list(map(polyline_length, coords))
    links._crs = crs

    return vertices, links


def read_osm(path: str, *, crs: int = 4326, layer_name: str = 'lines',
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
    vertices : :class:`VertexData`
        Vertices extracted from the OSM files.
    links : :class:`LinkData`
        Undirected links extracted from the OSM files.

    See also
    --------
    :func:`read_osms`
        Read multiple OSM files.

    References
    ----------
    https://wiki.openstreetmap.org/wiki/Key:highway
    '''
    return read_osms(path, crs=crs, layer_name=layer_name, exclude=exclude)


class Model:
    '''
    Class representing an RNet model.

    Parameters
    ----------
    nodes : :class:`NodeData`
        Node data.
    edges : :class:`EdgeData`
        Edge data.
    **others : Dict[str, :class:`Dataset`]
        Other datasets.

    See also
    --------
    :func:`model`
        Construct a model from multiple data sources.
    :func:`simplify`
        Return simplified model.
    '''

    def __init__(self, nodes: NodeData, edges: EdgeData, **others):
        self.nodes = nodes
        self.edges = edges
        for name, dataset in others.items():
            setattr(self, name, dataset)

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

        .. versionadded:: 0.0.6

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
        if hasattr(self, 'links'):
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

    def info(self) -> None:
        '''
        Print model information.

        .. versionadded:: 0.0.6
        '''
        print(self.__class__, '', 'Attributes:', sep='\n |  ')
        for k, v in self.__dict__.items():
            if isinstance(v, Dataset):
                print(f' |\n |  {k!r}', end='\n |  ')
                print(v.info(sep='\n |  ', ret=True))

    def to_pickle(self, path_to_pickle: str) -> None:
        '''
        Write model to disk in pickled representation.

        .. versionadded:: 0.0.6

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
        if hasattr(self, 'vertices'):
            self.vertices.transform(dst)
        if hasattr(self, 'links'):
            self.links.transform(dst)


def model(*paths, crs: int = 4326, keep_vertices: bool = False,
          keep_links: bool = False, reindex_nodes: bool = True,
          layer_name: str = 'lines', exclude: List[str] = [],
          r: float = 1e-3, p: int = 2, place_radius: float = 0) -> Model:
    '''
    Construct a model from multiple data sources.

    Parameters
    ----------
    *paths : str
        A single directory path or multiple file paths. The following
        file types are supported:

            * OSM files containing street map data
            * TIF files containing elevation data
            * CSV files containing place data

    crs : int, optional
        EPSG code of model CRS. The default is 4326.

    Other parameters
    ----------------
    keep_vertices, keep_links : bool, optional
        If True, then the sets of vertices and links are retained in the
        model. The defaults are False.
    reindex_nodes : bool, optional
        If True, then nodes are indexed using a range starting at 0.
        Otherwise, nodes inherit IDs from the set of vertices.
        The default is True.
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
    place_radius : float, optional
        If given, this radius is used to form place groups and create
        areas. Places form a group if they are located within this
        radius. The area of a place group is formed by the union of
        circles centered at each group member with this radius. Units
        should match those of the given `crs`. The default is 0.

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

    # Dictionary for storing datasets
    others = {}

    # Gather map data and extract map nodes
    if sorted['.osm']:
        vertices, links = read_osms(
            *sorted['.osm'], crs=crs, layer_name=layer_name, exclude=exclude)
        neighbor_counts = Counter(list(links.pairs().flatten()))
        node_ids = np.sort([k for k, v in neighbor_counts.items() if v != 2])
    else:
        vertices = None
        links = None
        node_ids = []

    # Gather place data
    if sorted['.csv']:
        places, areas, border_nodes, links = \
            model_places(*sorted['.csv'], crs=crs, radius=place_radius,
                         links=links, start=len(vertices))
        others['places'] = places
        others['areas'] = areas
        others['border_nodes'] = border_nodes
        vertices = VertexData(
            pd.concat([vertices.df, border_nodes.df]), crs)
        node_ids = np.union1d(node_ids, list(border_nodes._df.index))

    # Extract nodes and edges
    nodes = NodeData(vertices._df.iloc[node_ids], crs)
    edges = extract_edges(links, vertices, set(node_ids))
    if reindex_nodes:
        nodes._df = nodes._df.reset_index(drop=True)
        _, inverse = np.unique(edges.pairs().flatten(), return_inverse=True)
        edges._df[['i', 'j']] = inverse.reshape(-1, 2)

    # Calculate elevations
    if sorted['.osm'] and sorted['.tif']:
        nodes.elevate(*sorted['.tif'], r=r, p=p)
        edges.elevate(*sorted['.tif'], r=r, p=p)
        if keep_vertices:
            vertices.elevate(*sorted['.tif'], r=r, p=p)
        if keep_links:
            links.elevate(*sorted['.tif'], r=r, p=p)

    if keep_vertices:
        others['vertices'] = vertices
    if keep_links:
        others['links'] = links
    return Model(nodes, edges, **others)


def extract_edges(links: LinkData, vertices: VertexData, node_ids: Set[int]
                  ) -> EdgeData:
    '''
    Extract directed edges from link data.

    Parameters
    ----------
    links : :class:`LinkData`
        Link data.
    vertices : :class:`VertexData`
        Vertex data that provides the coordinates of link endpoints.
    node_ids : Set[int]
        Set of node IDs. Edges begin and end at a node whose ID is in
        this set. Node coordinates are taken from the set of vertices.

    Returns
    -------
    :class:`EdgeData`
        Directed edge data.
    '''
    actions = links.actions()
    vseqs = []  # vertex sequences
    lseqs = []  # link sequences
    for i in node_ids:
        for action, j in actions[i]:
            vseqs.append([i, j])
            lseqs.append([action])
            while vseqs[-1][-1] not in node_ids:
                actions_ = actions[vseqs[-1][-1]]
                try:
                    action_ = actions_[0]
                    assert action_[0] != lseqs[-1][-1]
                except AssertionError:
                    action_ = actions_[1]
                finally:
                    vseqs[-1].append(action_[1])
                    lseqs[-1].append(action_[0])
    i = [vseq[0] for vseq in vseqs]
    j = [vseq[-1] for vseq in vseqs]
    tags = links._df['tag'].iloc[[lseq[0] for lseq in lseqs]]
    vcoords = vertices.coords(2)[list(chain.from_iterable(vseqs))]
    coords = []
    for length in map(len, vseqs):
        coords.append(vcoords[:length])
        vcoords = vcoords[length:]
    lengths = list(map(polyline_length, coords))
    edges_df = pd.DataFrame(zip(i, j, tags, lengths, coords),
                            columns=['i', 'j', 'tag', 'length', 'coords'])
    return EdgeData(edges_df, links.crs, directed=True)


def model_places(*paths: str, crs: int, radius: float, links: LinkData,
                 start: int = 0) -> Tuple[PlaceData, AreaData, NodeData, LinkData]:
    '''
    Model places by forming place groups and extracting border nodes.

    Parameters
    ----------
    *paths : str
        Paths to CSV files containing place data.
    crs : int
        EPSG code of CRS in which place coordinates are represented.
    radius : float or None
        Place radius. Units should match those of the given `crs`.
    links : :class:`LinkData`
        Links used to find border nodes. Border nodes are points of
        intersection between a link in this set and an outer arc of
        a place group.
    start : int, optional
        Starting index for border nodes. The default is 0.

    Returns
    -------
    places : :class:`PlaceData`
        Place data.
    areas : :class:`AreaData
        Area data.
    border_nodes : :class:`NodeData`
        Border nodes.
    links : :class:`LinkData`
        Updated link data.
    '''
    # Read place data from files
    places = PlaceData.from_csvs(*paths, crs=4326)
    if crs != 4326:
        places.transform(crs)

    # Search for neighboring places
    place_coords = places.coords(2)
    tree = cKDTree(place_coords)
    neighbors = defaultdict(set)
    for (i, j) in tree.query_pairs(radius):
        neighbors[i].add(j)
        neighbors[j].add(i)
    neighbors = dict(neighbors)
    num_places = len(places)
    for index in range(num_places):
        if index not in neighbors:
            neighbors[index] = set()

    # Form place groups
    groups = ccl(neighbors)
    group_ids = [None] * num_places
    for group_id, group_members in enumerate(groups, 1):
        for member in group_members:
            group_ids[member] = group_id
    places._df['group'] = group_ids

    # Find area coordinates
    all_arcs = []
    area_coords = []
    for group in groups:
        circles = [Circle(*place_coords[place_id], radius)
                   for place_id in group]
        arc_points = []
        for arc in outer_arcs(*circles):
            all_arcs.append(arc)
            arc_points.append(arc.points()[:-1])
        area_coords.append(np.vstack(arc_points))
    areas = AreaData(pd.DataFrame(zip(area_coords), columns=['coords']), crs)

    # Find border nodes
    node_id = count(start)
    link_coords = links.coords(2)
    all_border_points = []
    all_split_points = defaultdict(list)
    for arc in all_arcs:
        indices, _, border_points = arc.intersections(link_coords)
        all_border_points.append(border_points)
        for (link_id, point) in zip(indices, border_points):
            all_split_points[link_id].append((next(node_id), point))
    all_split_points = dict(all_split_points)
    border_nodes = NodeData(
        pd.DataFrame(np.vstack(all_border_points), columns=['x', 'y']), crs)
    border_nodes._df.index += start

    # Split links at border nodes
    i = []
    j = []
    tags = []
    coords = []
    for link_id, split_points in all_split_points.items():
        link = links._df.iloc[link_id]
        if len(split_points) == 1:
            node_id, node_coords = split_points[0]
            i.extend([link.i, node_id])
            j.extend([node_id, link.j])
            tags.extend([link.tag] * 2)
            coords.extend([
                np.vstack((link.coords[0], node_coords)),
                np.vstack((node_coords, link.coords[1]))
            ])
        else:
            x1, x2 = link.coords[:, 0]
            dx = x2 - x1
            scale_factors = [(p[0] - x1) / dx for (_, p) in split_points]
            sorted_split_points = [split_points[index]
                                   for index in np.argsort(scale_factors)]
            vseq = [link.i] \
                + [split_point[0] for split_point in sorted_split_points] \
                + [link.j]
            coord_seq = np.vstack((
                link.coords[0],
                [split_point[1] for split_point in sorted_split_points],
                link.coords[1]))
            num_segments = len(vseq) - 1
            i.extend(vseq[:-1])
            j.extend(vseq[1:])
            tags.extend([link.tag] * num_segments)
            coords.extend([coord_seq[k:k+2] for k in range(num_segments)])
    lengths = list(map(polyline_length, coords))
    links = links._df.drop(list(all_split_points))
    links = pd.concat([
        links,
        pd.DataFrame(zip(i, j, tags, lengths, coords),
                     columns=['i', 'j', 'tag', 'length', 'coords'])])
    links = links.reset_index(drop=True)
    links = LinkData(links, crs, directed=False)
    return places, areas, border_nodes, links


def simplify(model: Model, *, xmin: float = None, ymin: float = None,
             xmax: float = None, ymax: float = None) -> Model:
    '''
    Return simplified model.

    .. versionadded:: 0.0.5

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
    >>> m = rn.model("./resources/shinjuku.osm")
    >>> m.info()
    <class 'rnet.model.Model'>
     |
     |  Attributes:
     |
     |  'nodes'
     |  <class 'rnet.dataset.NodeData'>
     |  Count: 944
     |  CRS: EPSG:4326
     |  dims: 3
     |  xmin: 139.6867708
     |  ymin: 35.6782595
     |  xmax: 139.7213249
     |  ymax: 35.7018425
     |
     |  'edges'
     |  <class 'rnet.dataset.EdgeData'>
     |  Count: 2,816
     |  CRS: EPSG:4326
     |  dims: 2
     |  xmin: 139.6867708
     |  ymin: 35.6782595
     |  xmax: 139.7213249
     |  ymax: 35.7018425

    Construct a simplified model by applying a lower bound on the
    :math:`x`-coordinates::

        >>> n = rn.simplify(m, xmin=139.7)
        >>> n.info()
        <class 'rnet.model.Model'>
         |
         |  Attributes:
         |
         |  'nodes'
         |  <class 'rnet.dataset.NodeData'>
         |  Count: 475
         |  CRS: EPSG:4326
         |  dims: 3
         |  xmin: 139.7000571
         |  ymin: 35.6782595
         |  xmax: 139.7213249
         |  ymax: 35.6982890
         |
         |  'edges'
         |  <class 'rnet.dataset.EdgeData'>
         |  Count: 1,418
         |  CRS: EPSG:4326
         |  dims: 2
         |  xmin: 139.7000571
         |  ymin: 35.6782595
         |  xmax: 139.7213249
         |  ymax: 35.6982890
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
    if hasattr(model, 'vertices'):
        vertices = VertexData(
            model.vertices._df.iloc[
                model.vertices.mask(xmin, ymin, xmax, ymax)],
            model.vertices._crs
        )
    else:
        vertices = None
    if hasattr(model, 'links'):
        links = LinkData(
            model.links._df.iloc[model.links.mask(xmin, ymin, xmax, ymax)],
            model.links._crs,
            directed=model.links._directed
        )
    else:
        links = None
    return Model(nodes, edges, vertices, links)
