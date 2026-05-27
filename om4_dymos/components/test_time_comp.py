import sys
import os
import numpy as np

sys.path.insert(0, os.path.expanduser('~/Codes/om4.git'))
sys.path.insert(0, '.')

from om4_dymos.components.time_comp import TimeComp

# Mock grid data
num_nodes = 3
node_ptau = np.array([-1.0, 0.0, 1.0])
node_dptau_dstau = np.array([1.0, 1.0, 1.0])

comp = TimeComp(
    num_nodes=num_nodes,
    node_ptau=node_ptau,
    node_dptau_dstau=node_dptau_dstau,
    units='s',
)

# Simulate __call__
t, t_phase, dt_dstau = comp(t_initial=np.array([10.0]), t_duration=np.array([5.0]))

print(f't: {t}')
print(f't_phase: {t_phase}')
print(f'dt_dstau: {dt_dstau}')
