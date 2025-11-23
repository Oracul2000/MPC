"""
MPC (Model Predictive Control) controllers for vehicle control.

Available controllers:
- MPCController: Basic kinematic MPC (legacy)
- MPCControllerImproved: Advanced dynamic MPC with CTE constraints
"""

from .mpc import MPCController
from .mpc_improved import MPCControllerImproved

__all__ = ['MPCController', 'MPCControllerImproved']
