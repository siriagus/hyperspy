# -*- coding: utf-8 -*-

import logging

_logger = logging.getLogger(__name__)

try:
    import sip
    _logger.debug('Setting Qt API to v2')
    sip.setapi('QVariant', 2)
    sip.setapi('QDate', 2)
    sip.setapi('QDateTime', 2)
    sip.setapi('QTextStream', 2)
    sip.setapi('QTime', 2)
    sip.setapi('QUrl', 2)
    del sip
except (ImportError, ValueError):
    _logger.debug('sip not present, Qt API not set')
    pass

from hyperspy import docstrings

__doc__ = """
HyperSpy: a multi-dimensional data analysis package for Python
==============================================================

Documentation is available in the docstrings and online at
http://hyperspy.org/hyperspy-doc/current/index.html.

All public packages, functions and classes are in :mod:`~hyperspy.api`. All
other packages and modules are for internal consumption and should not be
needed for data analysis.

%s

More details in the :mod:`~hyperspy.api` docstring.

""" % docstrings.START_HSPY

import os

os.environ['QT_API'] = "pyqt"


from . import Release

__all__ = ["api"]
__version__ = Release.version
