"""Projected LSTM using an ordinary fused LSTM after an exact first step.

The effective recurrent matrix is W_hh @ W_hr. The first step consumes the
public projected hidden state directly, preserving gradients and detached
chunk carry even if W_hr changed between chunks. No effective weights are
cached or detached. The external state remains (projected h, full-width c).
"""

import torch
from torch import nn
from torch.nn import functional as F

from .projected_lstm import ProjectedLSTM, _replace_projected_lstm


class FusedProjectedLSTM(ProjectedLSTM):
    """Optional faster path using private ``torch._VF.lstm`` for the tail.

    See the package README for version risk and exact first-step rationale.
    """

    def forward(self, input, hx=None):
        if not isinstance(input, torch.Tensor) or input.ndim != 3:
            raise ValueError("Expected dense (batch, time, input) tensor")
        batch, steps, width = input.shape
        if width != self.input_size or steps == 0:
            raise ValueError("Expected configured input width and nonzero time length")
        if hx is None:
            hidden = input.new_zeros(batch, self.proj_size)
            cell = input.new_zeros(batch, self.hidden_size)
        else:
            hidden, cell = hx
            if tuple(hidden.shape) != (1, batch, self.proj_size):
                raise ValueError("Hidden state must have shape (1, batch, proj_size)")
            if tuple(cell.shape) != (1, batch, self.hidden_size):
                raise ValueError("Cell state must have shape (1, batch, hidden_size)")
            hidden, cell = hidden[0], cell[0]
        gates = F.linear(input[:, 0], self.weight_ih_l0, self.bias_ih_l0 if self.bias else None)
        gates = gates + F.linear(hidden, self.weight_hh_l0, self.bias_hh_l0 if self.bias else None)
        i, f, g, o = gates.chunk(4, dim=-1)
        cell = f.sigmoid() * cell + i.sigmoid() * g.tanh()
        unprojected = o.sigmoid() * cell.tanh()
        if steps > 1:
            weights = [self.weight_ih_l0, self.weight_hh_l0 @ self.weight_hr_l0]
            if self.bias:
                weights += [self.bias_ih_l0, self.bias_hh_l0]
            rest, _, final_cell = torch._VF.lstm(
                input[:, 1:].contiguous(),
                (unprojected.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous()),
                weights, self.bias, 1, 0.0, self.training, False, True,
            )
            unprojected = torch.cat((unprojected.unsqueeze(1), rest), dim=1)
        else:
            unprojected, final_cell = unprojected.unsqueeze(1), cell.unsqueeze(0)
        output = F.linear(unprojected, self.weight_hr_l0)
        return output, (output[:, -1].unsqueeze(0), final_cell)


def replace_projected_lstm_fused(module: nn.Module) -> nn.Module:
    """Patch native projected LSTMs using the optional fused implementation.

    Reuses the manual patcher's parameter, checkpoint, alias, and scope
    guarantees. Already adapted modules are left as-is; choose the desired
    implementation when patching the native model.
    """
    return _replace_projected_lstm(module, FusedProjectedLSTM)
