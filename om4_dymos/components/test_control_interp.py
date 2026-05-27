import sys
import os
import numpy as np

sys.path.insert(0, os.path.expanduser('~/Codes/om4.git'))
sys.path.insert(0, '.')

from om4_dymos.components.control_interp_comp import ControlInterpComp
from om4_dymos.options import ControlOptions

# Simple scenario: 3 input nodes, 3 output nodes, identity matrix
L = np.eye(3)
D = np.eye(3) * 2.0  # just mock differentiation
D2 = np.eye(3) * 4.0

comp = ControlInterpComp(
    num_input_nodes=3,
    num_output_nodes=3,
    L=L,
    D=D,
    D2=D2,
    control_options={'theta': ControlOptions(name='theta', shape=(1,), units='rad')},
)

dt_dstau = np.array([1.0, 1.0, 1.0])
theta_in = np.array([[1.0], [2.0], [3.0]])

# Simulating __call__ dynamically parsing the kwargs unpacking
results = comp(dt_dstau, **{'controls:theta': theta_in})

# The component returns a tuple of the outputs based on iteration order
# The order is: val, rate, rate2 for each control
val, rate, rate2 = results

print('Value:')
print(val)
print('Rate:')
print(rate)
print('Rate2:')
print(rate2)
