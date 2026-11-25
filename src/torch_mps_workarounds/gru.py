"""Weight-preserving nn.GRU built from ordinary operators.

Supports any number of layers, bidirectional or not, either batch layout,
bias on/off, and inter-layer dropout. Dense 3-D input only (no PackedSequence,
no unbatched input). Parameters are the original objects of the source GRU.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .projected_lstm import _replace_projected_lstm


class FallbackGRU(nn.Module):
    """Explicit time-step GRU; hidden is (num_layers * directions, batch, hidden)."""

    def __init__(self, source: nn.GRU):
        super().__init__()
        if not isinstance(source, nn.GRU):
            raise TypeError("source must be torch.nn.GRU")
        for name in (
            "input_size", "hidden_size", "num_layers", "bias",
            "batch_first", "dropout", "bidirectional", "mode",
        ):
            setattr(self, name, getattr(source, name))
        self._flat_weights_names = list(source._flat_weights_names)
        self._all_weights = [list(names) for names in source._all_weights]
        for name, parameter in source.named_parameters(recurse=False):
            self.register_parameter(name, parameter)
        self.train(source.training)

    @property
    def all_weights(self):
        return [[getattr(self, name) for name in names] for names in self._all_weights]

    @property
    def _flat_weights(self):
        return [getattr(self, name) for name in self._flat_weights_names]

    def flatten_parameters(self):
        """Compatibility with callers preparing native recurrent modules."""

    def _direction(self, x, h, layer, suffix):
        w_ih, w_hh = (getattr(self, f"weight_{k}_l{layer}{suffix}") for k in ("ih", "hh"))
        b_ih = getattr(self, f"bias_ih_l{layer}{suffix}") if self.bias else None
        b_hh = getattr(self, f"bias_hh_l{layer}{suffix}") if self.bias else None
        affine = F.linear(x, w_ih, b_ih).unbind(dim=1)  # (batch, 3H) per step
        order = range(len(affine) - 1, -1, -1) if suffix else range(len(affine))
        outputs = [None] * len(affine)
        for t in order:
            ir, iz, in_ = affine[t].chunk(3, dim=-1)
            hr, hz, hn = F.linear(h, w_hh, b_hh).chunk(3, dim=-1)
            r = (ir + hr).sigmoid()
            z = (iz + hz).sigmoid()
            n = (in_ + r * hn).tanh()
            h = (1 - z) * n + z * h
            outputs[t] = h
        return torch.stack(outputs, dim=1), h

    def forward(self, input, hx=None):
        if not isinstance(input, torch.Tensor) or input.ndim != 3:
            raise ValueError("FallbackGRU expects a dense 3-D tensor")
        x = input if self.batch_first else input.transpose(0, 1)
        batch, steps, width = x.shape
        if width != self.input_size or steps == 0:
            raise ValueError("Input must have the configured width and nonzero time length")
        directions = 2 if self.bidirectional else 1
        shape = (self.num_layers * directions, batch, self.hidden_size)
        if hx is None:
            hx = input.new_zeros(shape)
        elif tuple(hx.shape) != shape:
            raise ValueError(f"Hidden state must have shape {shape}")
        finals = []
        for layer in range(self.num_layers):
            if layer and self.dropout:
                x = F.dropout(x, self.dropout, self.training)
            pieces = []
            for d, suffix in enumerate(("", "_reverse")[:directions]):
                out, h = self._direction(x, hx[layer * directions + d], layer, suffix)
                pieces.append(out)
                finals.append(h)
            x = torch.cat(pieces, dim=-1) if directions == 2 else pieces[0]
        output = x if self.batch_first else x.transpose(0, 1)
        return output, torch.stack(finals, dim=0)


def replace_gru(module: nn.Module) -> nn.Module:
    """Replace every nn.GRU recursively with FallbackGRU; return the root.

    Same contract as replace_projected_lstm: parameter objects, names, order,
    checkpoint keys, and shared-module aliases are preserved; repeating is safe.
    Assign the return value, since the root may itself be a GRU. Patch before
    .to(...) and optimizer construction.
    """
    return _replace_projected_lstm(module, FallbackGRU, lambda m: isinstance(m, nn.GRU))
