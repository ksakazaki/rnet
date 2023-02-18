from functools import wraps
import os

from osgeo import ogr
ogr.Open(os.path.join(os.path.dirname(
    os.path.dirname(__file__)), 'resources', 'shinjuku.osm'))

try:
    from qgis.core import QgsProject
except:
    _QGIS_AVAILABLE = False
else:
    _QGIS_AVAILABLE = True


def require_qgis(func):
    '''
    Wrapper for functions that require QGIS library.

    Raises
    ------
    NotImplementedError
        If a function with this decorator is run outside of QGIS.

    Example
    -------

        from qgis.core import QgsProject
        from rnet import require_qgis

        @require_qgis
        def get_crs():
            return QgsProject.instance().crs()
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    @wraps(func)
    def null_wrapper(*args, **kwargs):
        raise NotImplementedError(f'function {func.__name__!r} requires QGIS')

    if _QGIS_AVAILABLE:
        return wrapper
    else:
        return null_wrapper
