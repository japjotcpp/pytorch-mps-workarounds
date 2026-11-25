"""Synthetic projected-LSTM parity example; CPU by default, MPS explicit."""
import argparse
import copy
import json
import os

import torch
from torch import nn

from torch_mps_workarounds import replace_projected_lstm, replace_projected_lstm_fused


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--implementation', choices=('manual', 'fused'), default='manual')
    parser.add_argument('--show-native-error', action='store_true')
    args = parser.parse_args()
    if args.device == 'mps':
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0':
            parser.error('set PYTORCH_ENABLE_MPS_FALLBACK=0 before starting Python')
        if not torch.backends.mps.is_available():
            parser.error('MPS is unavailable; no GPU check ran')
    elif args.show_native_error:
        parser.error('--show-native-error requires --device mps')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(7)
    native = nn.LSTM(5, 12, proj_size=4, batch_first=True)
    reference = copy.deepcopy(native)
    original_parameters = dict(native.named_parameters())
    checkpoint = copy.deepcopy(native.state_dict())
    patch = replace_projected_lstm if args.implementation == 'manual' else replace_projected_lstm_fused
    adapted = patch(native)
    assert all(original_parameters[name] is p for name, p in adapted.named_parameters())
    adapted.load_state_dict(checkpoint, strict=True)
    adapted = adapted.to(args.device)
    x_cpu = torch.randn(2, 12, 5, requires_grad=True)
    x = x_cpu.detach().clone().to(args.device).requires_grad_(True)
    if args.show_native_error:
        try:
            native_mps = copy.deepcopy(reference).to('mps')
            with torch.no_grad():
                native_mps(x)
                torch.mps.synchronize()
            print('Native projected LSTM executed on MPS in this installation.')
        except RuntimeError as error:
            print(f'Native projected LSTM on MPS raised: {error}')
    expected, expected_state = reference(x_cpu)
    actual, actual_state = adapted(x)
    atol, rtol = (1e-5, 2e-3) if args.device == 'mps' else (3e-6, 3e-4)

    def close(got, want):
        torch.testing.assert_close(got.detach().cpu(), want.detach().cpu(), atol=atol, rtol=rtol)

    close(actual, expected)
    for got, want in zip(actual_state, expected_state):
        close(got, want)
    expected_loss = expected.square().mean() + sum(s.square().mean() for s in expected_state)
    actual_loss = actual.square().mean() + sum(s.square().mean() for s in actual_state)
    expected_loss.backward()
    actual_loss.backward()
    close(x.grad, x_cpu.grad)
    for (name, got), (other_name, want) in zip(adapted.named_parameters(), reference.named_parameters()):
        assert name == other_name
        close(got.grad, want.grad)
    with torch.no_grad():
        first, state = adapted(x[:, :5])
        last, final_state = adapted(x[:, 5:], state)
        close(torch.cat([first, last], dim=1), actual)
        for got, want in zip(final_state, actual_state):
            close(got, want)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
    optimizer = torch.optim.SGD(adapted.parameters(), lr=.01)
    reference_optimizer.step()
    optimizer.step()
    for got, want in zip(adapted.parameters(), reference.parameters()):
        close(got, want)
    print(json.dumps(dict(status='PASS', device=args.device, torch=torch.__version__,
                          implementation=args.implementation, shape=[2, 12, 5, 12, 4],
                          checks=['parameter identity before device move', 'strict checkpoint',
                                  'forward/state', 'input/parameter gradients', 'chunk carry',
                                  'optimizer update']), indent=2))


if __name__ == '__main__':
    main()
