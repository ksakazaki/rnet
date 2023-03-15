from __future__ import annotations
from typing import List
import numpy as np
from rnet.env import _QGIS_AVAILABLE, require_qgis
from rnet.dataset import Field, Dataset, PointData, ConnectionData, AreaData
from rnet.model import Model
from rnet.taskmanager import create_and_queue

__all__ = ['render']

if _QGIS_AVAILABLE:
    from PyQt5.QtCore import QVariant
    from qgis.core import (
        QgsFeature,
        QgsFeatureSink,
        QgsField,
        QgsGeometry,
        QgsPointXY,
        QgsProject,
        QgsTask,
        QgsVectorLayer
    )
else:
    QgsTask = object


class RenderTask(QgsTask):
    '''
    Task for rendering features.

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        Frame containing the data to be rendered.
    vl : :class:`~qgis.core.QgsVectorLayer`
        Layer on which the features are rendered.
    description : str
        Task description.
    '''

    @require_qgis
    def __init__(self, dataset: Dataset, vl: QgsVectorLayer, description: str):
        self.data = dataset
        self.vl = vl
        super().__init__(description)

    def run(self) -> bool:
        self.vl.dataProvider().addFeatures(self.data.features(self),
                                           QgsFeatureSink.FastInsert)
        return True

    def finished(self, success: bool) -> None:
        if success:
            QgsProject.instance().addMapLayer(self.vl)


@require_qgis
def _create_temp_layer(geometry: str, layername: str, crs: int,
                       fields: List[Field]):
    '''
    Return a temporary vector layer.

    Parameters
    ----------
    geometry : {'point', 'linestring', 'polygon'}
        Geometry type.
    layername : str
        Layer name.
    crs : int
        EPSG code of the layer CRS.
    fields : list of :class:`Field`
        List of fields.

    Returns
    -------
    :class:`~qgis.core.QgsVectorLayer`
        A temporary layer.
    '''
    vl = QgsVectorLayer(f'{geometry}?crs=epsg:{crs}', layername, 'memory')
    all_fields = []
    all_fields.append(QgsField('Index', QVariant.Int))
    for field in filter(lambda field: field.include, fields):
        if 'float' in field.type:
            type_ = QVariant.Double
        elif 'int' in field.type:
            type_ = QVariant.Int
        else:
            type_ = QVariant.String
        all_fields.append(QgsField(field.name, type_))
    vl.dataProvider().addAttributes(all_fields)
    vl.updateFields()
    return vl


@require_qgis
def render(dataset: Dataset):
    '''
    Render a dataset in QGIS.

    .. versionadded:: 0.0.7

    Parameters
    ----------
    dataset : :class:`Dataset`
        The dataset to be rendered.

    Returns
    -------
    :class:`~qgis.core.QgsVectorLayer`
        Temporary vector layer.
    '''
    if isinstance(dataset, PointData):
        geometry = 'point'
    elif isinstance(dataset, ConnectionData):
        geometry = 'linestring'
    elif isinstance(dataset, AreaData):
        geometry = 'polygon'
    temp_vl = _create_temp_layer(
        geometry, dataset._LAYER_NAME, dataset.crs, dataset.FIELDS)
    create_and_queue(RenderTask, dataset, temp_vl, 'Rendering features')
    return temp_vl


@require_qgis
def render_path(model: Model, path: List[int]) -> None:
    '''
    Render a path in a temporary layer.

    .. versionadded:: 0.0.7

    Parameters
    ----------
    model : :class:`Model`
        RNet model.
    path : List[int]
        Sequence of node IDs.
    '''
    path_coords = []
    for (i, j) in zip(path[:-1], path[1:]):
        edge = next(model.edges.get(i, j))
        path_coords.append(edge.coords)
    path_coords = np.vstack(path_coords)

    feat = QgsFeature()
    feat.setGeometry(QgsGeometry.fromPolylineXY(
        [QgsPointXY(x, y) for (x, y) in path_coords]))
    temp_vl = _create_temp_layer('linestring', 'path', model.edges.crs, [])
    temp_vl.dataProvider().addFeatures([feat])
    QgsProject.instance().addMapLayer(temp_vl)
