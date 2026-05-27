from typing import Dict, Any
import numpy as np

from om4_dymos.transcription import TranscriptionBase
from om4_dymos.transcriptions.components.state_interp_comp import StateInterpComp
from om4_dymos.transcriptions.components.collocation_comp import CollocationComp
from om4_dymos.components.time_comp import TimeComp
from om4_dymos.components.control_interp_comp import ControlInterpComp
from om4.components.indepvarcomp import IndepVarComp
from om4.specs.varspec import ContinuousVar
from pydantic import Field


def lgr(n, include_endpoint=False, tol=1.0e-15):
    n1 = n
    n = n - 1
    x = -np.cos(2.0 * np.pi * np.arange(0, n1) / (2 * n + 1))
    P = np.zeros((n1, n1 + 1))
    xold = np.ones_like(x)
    free = np.arange(1, n1)
    for i in range(100):
        if np.all(np.abs(x - xold) <= tol):
            break
        xold[:] = x
        P[0, :] = np.power(-1.0, np.arange(0, n1 + 1))
        P[free, 0] = 1.0
        P[free, 1] = x[free]
        for k in range(1, n1):
            P[free, k + 1] = (
                (2 * (k + 1) - 1) * x[free] * P[free, k]
                - ((k + 1) - 1) * P[free, k - 1]
            ) / (k + 1)
        f = ((1.0 - xold[free]) / n1) * (P[free, n] + P[free, n1])
        fprime = P[free, n] - P[free, n1]
        x[free] = xold[free] - f / fprime
    else:
        raise RuntimeError('Failed to converge LGR nodes')
    w = np.zeros(n1)
    w[0] = 2.0 / n1**2
    w[free] = (1.0 - x[free]) / (n1 * P[free, n]) ** 2
    if include_endpoint:
        x = np.concatenate([x, [1.0]])
        w = np.concatenate([w, [0.0]])
    return x, w


def lagrange_matrices(x_disc, x_interp):
    nd = len(x_disc)
    ni = len(x_interp)
    temp = np.zeros((ni, nd))
    diff = np.reshape(x_disc, (nd, 1)) - np.reshape(x_disc, (1, nd))
    np.fill_diagonal(diff, 1.0)
    wb = np.prod(1.0 / diff, axis=1)

    diff_i = np.reshape(x_interp, (ni, 1)) - np.reshape(x_disc, (1, nd))

    Li = np.zeros((ni, nd))
    for j in range(nd):
        temp[:] = diff_i[:]
        temp[:, j] = 1.0
        Li[:, j] = wb[j] * np.prod(temp, axis=1)

    Di = np.zeros((ni, nd))
    for j in range(nd):
        for k in range(nd):
            if k != j:
                temp[:] = diff_i[:]
                temp[:, j] = 1.0
                temp[:, k] = 1.0
                Di[:, j] += wb[j] * np.prod(temp, axis=1)
    return Li, Di


class RadauNew(TranscriptionBase):
    num_segments: int = Field(default=10, description='Number of segments')
    order: int = Field(default=3, description='Polynomial order')
    compressed: bool = Field(
        default=True, description='Whether to use a compressed grid'
    )

    def build_spec(self, phase_options: Any) -> tuple[Dict[str, Any], list[dict]]:
        subs = {}
        conns = []

        nodes_per_seg = self.order + 1
        x_col_seg, _ = lgr(nodes_per_seg - 1, include_endpoint=False)
        x_disc_seg, _ = lgr(nodes_per_seg - 1, include_endpoint=True)

        x_control_in_seg = x_col_seg

        Li_seg, Ad_seg = lagrange_matrices(x_disc_seg, x_col_seg)
        Li_control_seg, D_control_seg = lagrange_matrices(x_control_in_seg, x_col_seg)
        D2_control_seg = np.dot(D_control_seg, D_control_seg)

        from scipy.linalg import block_diag

        Ad = block_diag(*[Ad_seg for _ in range(self.num_segments)])
        Li = block_diag(*[Li_seg for _ in range(self.num_segments)])

        Li_control = block_diag(*[Li_control_seg for _ in range(self.num_segments)])
        D_control = block_diag(*[D_control_seg for _ in range(self.num_segments)])
        D2_control = block_diag(*[D2_control_seg for _ in range(self.num_segments)])

        num_col_nodes = self.num_segments * (nodes_per_seg - 1)
        num_disc_nodes = self.num_segments * nodes_per_seg
        num_control_in_nodes = num_col_nodes

        segment_ends = np.linspace(-1, 1, self.num_segments + 1)
        node_ptau_col = []
        node_dptau_dstau = []
        for i in range(self.num_segments):
            v0 = segment_ends[i]
            v1 = segment_ends[i + 1]
            seg_len = v1 - v0
            mapped_col = v0 + 0.5 * seg_len * (x_col_seg + 1.0)
            node_ptau_col.extend(mapped_col)
            node_dptau_dstau.extend([0.5 * seg_len] * len(x_col_seg))

        node_ptau_col = np.array(node_ptau_col)
        node_dptau_dstau = np.array(node_dptau_dstau)

        ivc_outputs = {}
        t_opts = phase_options.time_options
        ivc_outputs['t_initial'] = ContinuousVar(
            val=t_opts.initial_val, units=t_opts.units
        )
        ivc_outputs['t_duration'] = ContinuousVar(
            val=t_opts.duration_val, units=t_opts.units
        )

        for name, opts in phase_options.state_options.items():
            shape = (
                tuple(opts.shape)
                if isinstance(opts.shape, (list, tuple))
                else (opts.shape,)
            )
            if not shape:
                shape = ()
            ivc_outputs[f'states:{name}'] = ContinuousVar(
                shape=(num_disc_nodes,) + shape, val=opts.initial_val, units=opts.units
            )

        for name, opts in phase_options.control_options.items():
            shape = (
                tuple(opts.shape)
                if isinstance(opts.shape, (list, tuple))
                else (opts.shape,)
            )
            if not shape:
                shape = ()
            ivc_outputs[f'controls:{name}'] = ContinuousVar(
                shape=(num_control_in_nodes,) + shape, val=opts.val, units=opts.units
            )

        subs['indep_vars'] = IndepVarComp(outputs=ivc_outputs)

        subs['time'] = TimeComp(
            num_nodes=num_col_nodes,
            node_ptau=node_ptau_col,
            node_dptau_dstau=node_dptau_dstau,
            units=t_opts.units,
        )
        conns.append({'src': 'indep_vars.t_initial', 'tgt': 'time.t_initial'})
        conns.append({'src': 'indep_vars.t_duration', 'tgt': 'time.t_duration'})

        for tgt in t_opts.targets:
            conns.append({'src': 'time.t', 'tgt': f'ode_all.{tgt}'})

        if phase_options.control_options:
            subs['control_interp'] = ControlInterpComp(
                num_input_nodes=num_control_in_nodes,
                num_output_nodes=num_col_nodes,
                control_options=phase_options.control_options,
                L=Li_control,
                D=D_control,
                D2=D2_control,
            )
            conns.append({'src': 'time.dt_dstau', 'tgt': 'control_interp.dt_dstau'})
            for name, opts in phase_options.control_options.items():
                conns.append(
                    {
                        'src': f'indep_vars.controls:{name}',
                        'tgt': f'control_interp.controls:{name}',
                    }
                )
                for tgt in opts.targets:
                    conns.append(
                        {
                            'src': f'control_interp.control_values:{name}',
                            'tgt': f'ode_all.{tgt}',
                        }
                    )

        if phase_options.state_options:
            subs['state_interp'] = StateInterpComp(
                num_disc_nodes=num_disc_nodes,
                num_col_nodes=num_col_nodes,
                state_options=phase_options.state_options,
                Ad=Ad,
                Li=Li,
            )
            conns.append({'src': 'time.dt_dstau', 'tgt': 'state_interp.dt_dstau'})
            for name in phase_options.state_options:
                conns.append(
                    {
                        'src': f'indep_vars.states:{name}',
                        'tgt': f'state_interp.state_disc:{name}',
                    }
                )

        subs['ode_all'] = phase_options.ode_class(num_nodes=num_col_nodes)

        for name, opts in phase_options.state_options.items():
            for target in opts.targets:
                conns.append(
                    {
                        'src': f'state_interp.state_col:{name}',
                        'tgt': f'ode_all.{target}',
                    }
                )

        if phase_options.state_options:
            subs['defects'] = CollocationComp(
                num_col_nodes=num_col_nodes, state_options=phase_options.state_options
            )
            conns.append({'src': 'time.dt_dstau', 'tgt': 'defects.dt_dstau'})
            for name, opts in phase_options.state_options.items():
                conns.append(
                    {
                        'src': f'state_interp.staterate_col:{name}',
                        'tgt': f'defects.f_approx:{name}',
                    }
                )
                conns.append(
                    {
                        'src': f'ode_all.{opts.rate_source}',
                        'tgt': f'defects.f_computed:{name}',
                    }
                )

        return subs, conns
