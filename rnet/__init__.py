import os
from osgeo import ogr
ogr.Open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'test.osm'))

from rnet.coordinates import *
from rnet.model import *
from rnet.utils import *

__version__ = '0.0.3'
__author__ = 'Kota Sakazaki'