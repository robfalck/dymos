"""
dymos - A framework for optimization of dynamic systems, using OpenMDAO.
"""
__version__ = '1.10.1-dev'


__all__ = ['Phase', 'AnalyticPhase',
           'GaussLobatto', 'Radau', 'ExplicitShooting', 'Analytic', 'Birkhoff',
           'GaussLobattoGrid', 'RadauGrid', 'UniformGrid', 'BirkhoffGrid',
           'Trajectory',
           'run_problem',
           'load_case',
           'options']


from .phase import Phase, AnalyticPhase
from .transcriptions import GaussLobatto, Radau, ExplicitShooting, Analytic, Birkhoff
from .transcriptions.grid_data import GaussLobattoGrid, RadauGrid, UniformGrid, BirkhoffGrid
from .trajectory.trajectory import Trajectory
from .run_problem import run_problem
from .load_case import load_case
from ._options import options
