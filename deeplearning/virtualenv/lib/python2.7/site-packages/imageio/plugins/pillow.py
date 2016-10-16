# -*- coding: utf-8 -*-
# Copyright (c) 2016, imageio contributors
# imageio is distributed under the terms of the (new) BSD License.

""" Plugin that wraps the the Pillow library.
"""

from __future__ import absolute_import, print_function, division

from .. import formats
from ..core import Format, image_as_uint
from ._freeimage import fi, IO_FLAGS, FNAME_PER_PLATFORM  # noqa


FULL_FORMATS = ('.bmp', '.eps', '.gif', '.icns', '.im',
                '.jpeg', '.jpg', '.jpeg2000',
                '.msp', '.pcx', '.png', '.ppm', '.spider', '.tiff', '.webp',
                '.xbm')

READABLE_FORMATS = ('.cur', '.dcx', '.dds', '.fli', '.flc', '.fpx', '.gbr',
                    '.gd', '.ico', '.icns', '.imt', '.iptc', '.naa', '.mcidas',
                    '.mpo', '.pcd', '.psd', '.cgi', '.tga', '.wal', '.xpm')

WRITABLE_FORMATS = ('.palm', '.pdf', '.pixar')

# Pillow "knows" these, without being able to read or write them
IDENTIFIABLE_FORMATS = ('.bufr', '.fits', '.grib', '.hdf5', '.mpeg', '.wmf')


# todo: Pillow ImageGrab module supports grabbing the screen on Win and OSX. how awesome it that!

# todo: also ImageSequence module

# todo: format has short name by default, unless it already exists,
# in which case it gets a "PIL_" prefix. Make possible to use that prefix
# even for the formats that got the prefix-less name.
