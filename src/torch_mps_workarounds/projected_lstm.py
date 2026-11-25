"""Weight-preserving projected LSTM for PyTorch devices without a native kernel.

Only dense, one-layer, batch-first, unidirectional projected LSTMs are
supported. No checkpoints or normalizers are rewritten.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ProjectedLSTM(nn.Module):
    """An ordinary-operator implementation with the original parameter objects.

    Input is (batch, time, input_size); hidden is (1, batch, proj_size),
    and cell is (1, batch, hidden_size), matching the supported nn.LSTM.
    The input affine is computed for the entire chunk in one batched operation.
    """

    def __init__(self, source: nn.LSTM):
        super().__init__()
        if not isinstance(source, nn.LSTM):
            raise TypeError("source must be torch.nn.LSTM")
        if not (
            source.proj_size > 0
            and source.num_layers == 1
            and source.batch_first
            and not source.bidirectional
        ):
            raise ValueError(
                "ProjectedLSTM supports only a projected, one-layer, "
                "batch-first, unidirectional LSTM"
            )
        for name in (
            "input_size", "hidden_size", "proj_size", "num_layers", "bias",
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

    def forward(self, input, hx=None):
        if not isinstance(input, torch.Tensor) or input.ndim != 3:
            raise ValueError("ProjectedLSTM expects a dense (batch, time, input) tensor")
        batch, steps, width = input.shape
        if width != self.input_size or steps == 0:
            raise ValueError("Input must have the configured width and nonzero time length")
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
        inputs = F.linear(
            input, self.weight_ih_l0,
            self.bias_ih_l0 if self.bias else None,
        ).unbind(dim=1)
        outputs = []
        for affine in inputs:
            gates = affine + F.linear(
                hidden, self.weight_hh_l0,
                self.bias_hh_l0 if self.bias else None,
            )
            i, f, g, o = gates.chunk(4, dim=-1)
            cell = f.sigmoid() * cell + i.sigmoid() * g.tanh()
            hidden = F.linear(o.sigmoid() * cell.tanh(), self.weight_hr_l0)
            outputs.append(hidden)
        return torch.stack(outputs, dim=1), (hidden.unsqueeze(0), cell.unsqueeze(0))


def replace_projected_lstm(module: nn.Module) -> nn.Module:
    """Replace supported projected LSTMs recursively; return the resulting root.

    Parameter names, values, and objects remain unchanged. Nonprojected LSTMs
    and all other modules/buffers remain the same objects. Shared child modules
    remain shared. Assign the return value, since the root may itself be an
    LSTM. Prefer calling before .to(...) and optimizer construction. Unsupported
    projected variants raise before the module tree is changed.
    """
    return _replace_projected_lstm(module, ProjectedLSTM)


def _replace_projected_lstm(module: nn.Module, implementation, match=None) -> nn.Module:
    """Shared atomic, alias-preserving model patch for both implementations."""
    if not isinstance(module, nn.Module):
        raise TypeError("module must be torch.nn.Module")
    if match is None:
        def match(child):
            return isinstance(child, nn.LSTM) and child.proj_size
    # Validate and construct replacements before editing a parent. A shared
    # projected LSTM may appear at different depths in the module tree.
    replacements = {}
    for child in module.modules():
        if match(child):
            replacements[id(child)] = implementation(child)

    visited = set()

    def replace(current):
        if id(current) in replacements:
            return replacements[id(current)]
        if id(current) in visited:
            return current
        visited.add(id(current))
        # _modules includes aliases that named_children() would deduplicate.
        for name, child in list(current._modules.items()):
            if child is not None:
                current._modules[name] = replace(child)
        return current

    return replace(module)
