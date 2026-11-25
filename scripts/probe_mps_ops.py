"""Probe which ops fail on MPS (fallback unset) or differ from CPU. Run: python scripts/probe_mps_ops.py"""
import os, warnings
os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
import torch
import torch.nn.functional as F

g = torch.Generator().manual_seed(0)
r = lambda *s, dt=torch.float32: torch.randn(*s, generator=g).to(dt)
i = lambda hi, *s: torch.randint(0, hi, s, generator=g)
C = {
 "cumsum": lambda: (r(4, 5).cumsum(1),),
 "cumprod": lambda: (r(4, 5).cumprod(1),),
 "cummax": lambda: (r(4, 5).cummax(1).values,),
 "cummin": lambda: (r(4, 5).cummin(1).values,),
 "logcumsumexp": lambda: (r(4, 5).logcumsumexp(1),),
 "unique": lambda: (torch.unique(i(5, 20)),),
 "unique_consecutive": lambda: (torch.unique_consecutive(i(3, 20)),),
 "sort": lambda: (r(4, 5).sort(1).values,),
 "topk": lambda: (r(4, 5).topk(2).values,),
 "nonzero": lambda: (r(4, 5).gt(0).nonzero(),),
 "masked_select": lambda: (r(4, 5).masked_select(r(4, 5).gt(0)),),
 "grid_sample": lambda: (F.grid_sample(r(1, 2, 4, 4), r(1, 3, 3, 2).clamp(-1, 1), align_corners=False),),
 "grid_sample_3d": lambda: (F.grid_sample(r(1, 2, 4, 4, 4), r(1, 3, 3, 3, 3).clamp(-1, 1), align_corners=False),),
 "affine_grid": lambda: (F.affine_grid(r(1, 2, 3), (1, 1, 4, 4), align_corners=False),),
 "ctc_loss": lambda: (F.ctc_loss(r(10, 2, 5).log_softmax(2), i(4, 2, 3) + 1, torch.tensor([10, 10]), torch.tensor([3, 3])),),
 "fft": lambda: (torch.fft.fft(r(8)).abs(),),
 "rfft": lambda: (torch.fft.rfft(r(2, 8)).abs(),),
 "fft2": lambda: (torch.fft.fft2(r(4, 4)).abs(),),
 "stft": lambda: (torch.stft(r(64), 16, return_complex=True).abs(),),
 "scatter_reduce_sum": lambda: (torch.zeros(5).scatter_reduce(0, i(5, 10), r(10), "sum"),),
 "scatter_reduce_amax": lambda: (torch.zeros(5).scatter_reduce(0, i(5, 10), r(10), "amax"),),
 "scatter_reduce_prod": lambda: (torch.ones(5).scatter_reduce(0, i(5, 10), r(10), "prod"),),
 "index_add": lambda: (torch.zeros(5).index_add(0, i(5, 10), r(10)),),
 "embedding_bag": lambda: (F.embedding_bag(i(10, 6), r(10, 4), torch.tensor([0, 3])),),
 "embedding_bag_max": lambda: (F.embedding_bag(i(10, 6), r(10, 4), torch.tensor([0, 3]), mode="max"),),
 "linalg.inv": lambda: (torch.linalg.inv(r(4, 4) + 4 * torch.eye(4)),),
 "linalg.solve": lambda: (torch.linalg.solve(r(4, 4) + 4 * torch.eye(4), r(4, 2)),),
 "linalg.cholesky": lambda: (torch.linalg.cholesky(torch.eye(4) * 2 + 0 * r(4, 4)),),
 "linalg.qr": lambda: (torch.linalg.qr(r(4, 4)).R.abs(),),
 "linalg.svd": lambda: (torch.linalg.svdvals(r(4, 4)),),
 "linalg.eigh": lambda: (torch.linalg.eigvalsh(r(4, 4) + r(4, 4).T),),
 "linalg.eig": lambda: (torch.linalg.eigvals(r(4, 4)).abs(),),
 "linalg.det": lambda: (torch.linalg.det(r(4, 4)),),
 "linalg.lstsq": lambda: (torch.linalg.lstsq(r(6, 4), r(6, 2)).solution,),
 "linalg.matrix_exp": lambda: (torch.linalg.matrix_exp(r(4, 4) * .1),),
 "linalg.norm_nuc": lambda: (torch.linalg.matrix_norm(r(4, 4), "nuc"),),
 "lu": lambda: (torch.linalg.lu_factor(r(4, 4) + 4 * torch.eye(4)).LU,),
 "triangular_solve": lambda: (torch.linalg.solve_triangular(torch.triu(r(4, 4)) + 4 * torch.eye(4), r(4, 2), upper=True),),
 "pinv": lambda: (torch.linalg.pinv(r(4, 3)),),
 "cdist": lambda: (torch.cdist(r(3, 4), r(5, 4)),),
 "bincount": lambda: (torch.bincount(i(5, 20)),),
 "histc": lambda: (torch.histc(r(100), 10, -3, 3),),
 "searchsorted": lambda: (torch.searchsorted(torch.arange(10.), r(5) * 3 + 5),),
 "bucketize": lambda: (torch.bucketize(r(5) * 3 + 5, torch.arange(10.)),),
 "median": lambda: (r(4, 5).median(1).values,),
 "kthvalue": lambda: (r(4, 5).kthvalue(2, 1).values,),
 "mode": lambda: (i(3, 4, 5).float().mode(1).values,),
 "multinomial": lambda: (None,),
 "interpolate_bicubic": lambda: (F.interpolate(r(1, 2, 4, 4), scale_factor=2, mode="bicubic"),),
 "interpolate_bilinear_aa": lambda: (F.interpolate(r(1, 2, 8, 8), size=4, mode="bilinear", antialias=True),),
 "interpolate_trilinear": lambda: (F.interpolate(r(1, 2, 4, 4, 4), scale_factor=2, mode="trilinear"),),
 "adaptive_max_pool2d": lambda: (F.adaptive_max_pool2d(r(1, 2, 7, 7), 3),),
 "adaptive_avg_pool2d": lambda: (F.adaptive_avg_pool2d(r(1, 2, 7, 7), 3),),
 "max_pool3d": lambda: (F.max_pool3d(r(1, 2, 4, 4, 4), 2),),
 "avg_pool3d": lambda: (F.avg_pool3d(r(1, 2, 4, 4, 4), 2),),
 "conv3d": lambda: (F.conv3d(r(1, 2, 4, 4, 4), r(3, 2, 3, 3, 3)),),
 "conv_transpose3d": lambda: (F.conv_transpose3d(r(1, 2, 4, 4, 4), r(2, 3, 3, 3, 3)),),
 "conv1d_float64": lambda: (None,),
 "unfold": lambda: (F.unfold(r(1, 2, 5, 5), 2),),
 "fold": lambda: (F.fold(r(1, 8, 16), (5, 5), 2),),
 "pixel_shuffle": lambda: (F.pixel_shuffle(r(1, 4, 3, 3), 2),),
 "roll": lambda: (r(4, 5).roll(1, 1),),
 "diff": lambda: (r(4, 5).diff(),),
 "trace": lambda: (r(4, 4).trace(),),
 "cross": lambda: (torch.linalg.cross(r(4, 3), r(4, 3)),),
 "erfinv": lambda: (r(5).clamp(-.9, .9).erfinv(),),
 "lgamma": lambda: (r(5).abs().lgamma(),),
 "digamma": lambda: (r(5).abs().digamma(),),
 "polygamma": lambda: (r(5).abs().polygamma(1),),
 "igamma": lambda: (torch.special.gammainc(r(5).abs() + 1, r(5).abs()),),
 "i0": lambda: (r(5).i0(),),
 "bessel_j0": lambda: (torch.special.bessel_j0(r(5)),),
 "complex_mul": lambda: ((torch.complex(r(4), r(4)) * torch.complex(r(4), r(4))).abs(),),
 "float64": lambda: (r(4, 4, dt=torch.float64) @ r(4, 4, dt=torch.float64),),
 "float16_matmul": lambda: (r(4, 4, dt=torch.float16) @ r(4, 4, dt=torch.float16),),
 "bfloat16_matmul": lambda: (r(4, 4, dt=torch.bfloat16) @ r(4, 4, dt=torch.bfloat16),),
 "int64_matmul": lambda: (None,),
 "nn.RNN": lambda: (None,),
 "nn.LSTM_proj": lambda: (None,),
 "nn.GRU": lambda: (None,),
 "nn.MultiheadAttention": lambda: (None,),
 "sdpa": lambda: (F.scaled_dot_product_attention(r(1, 2, 4, 8), r(1, 2, 4, 8), r(1, 2, 4, 8)),),
 "nn.Transformer": lambda: (None,),
 "index_put_acc": lambda: (torch.zeros(5).index_put((i(5, 10),), r(10), accumulate=True),),
 "put_": lambda: (torch.zeros(5).put_(i(5, 3), r(3)),),
 "take_along_dim": lambda: (r(4, 5).take_along_dim(i(5, 4, 2), 1),),
 "repeat_interleave": lambda: (r(4).repeat_interleave(torch.tensor([1, 2, 0, 3])),),
 "tril_indices": lambda: (torch.tril_indices(4, 4).float(),),
 "isin": lambda: (torch.isin(i(5, 10), torch.tensor([1, 2])).float(),),
 "nanmedian": lambda: (r(4, 5).nanmedian(),),
 "rrelu_train": lambda: (None,),
 "dropout": lambda: (None,),
 "nms_like_sort_int": lambda: (i(100, 20).sort().values,),
 "argsort_stable": lambda: (r(20).argsort(stable=True).float(),),
 "logdet": lambda: (torch.logdet(r(4, 4) + 4 * torch.eye(4)),),
 "kron": lambda: (torch.kron(r(2, 2), r(2, 2)),),
 "einsum": lambda: (torch.einsum("ij,jk->ik", r(3, 4), r(4, 5)),),
}

def special(name):
    if name == "nn.RNN": m = torch.nn.RNN(4, 8, batch_first=True); x = r(2, 5, 4)
    elif name == "nn.LSTM_proj": m = torch.nn.LSTM(4, 8, proj_size=2, batch_first=True); x = r(2, 5, 4)
    elif name == "nn.GRU": m = torch.nn.GRU(4, 8, 2, batch_first=True, bidirectional=True); x = r(2, 5, 4)
    elif name == "nn.MultiheadAttention": m = torch.nn.MultiheadAttention(8, 2, batch_first=True); x = r(2, 5, 8); return m, (x, x, x)
    elif name == "nn.Transformer": m = torch.nn.TransformerEncoderLayer(8, 2, 16, batch_first=True); x = r(2, 5, 8)
    elif name == "int64_matmul": return None
    else: return None
    return m, (x,)

def cpu_and_mps(name):
    if name in ("nn.RNN", "nn.LSTM_proj", "nn.GRU", "nn.MultiheadAttention", "nn.Transformer"):
        m, args = special(name); m.eval(); torch.manual_seed(0)
        with torch.no_grad():
            a = m(*args)[0]; m2 = type(m)
        import copy; mm = copy.deepcopy(m).to("mps")
        with torch.no_grad(): b = mm(*[t.to("mps") for t in args])[0]
        return a, b.cpu()
    if name == "multinomial":
        p = torch.rand(3, 5); return (torch.multinomial(p, 2).shape[0] * torch.ones(1)), torch.multinomial(p.to("mps"), 2).shape[0] * torch.ones(1)
    if name == "dropout":
        return torch.ones(1), F.dropout(torch.ones(100, device="mps"), .5).mean().cpu().clamp(max=1) * 0 + 1
    if name == "rrelu_train": return torch.ones(1), F.rrelu(torch.randn(10, device="mps"), training=True).cpu() * 0 + 1
    if name == "conv1d_float64": return None
    if name == "int64_matmul": return None
    # make function deterministic: rebuild generator per call
    g.manual_seed(1); a = C[name]()
    g.manual_seed(1)
    return a[0], None

def run(name):
    import inspect
    fn = C[name]
    if name in ("nn.RNN", "nn.LSTM_proj", "nn.GRU", "nn.MultiheadAttention", "nn.Transformer", "multinomial", "dropout", "rrelu_train"):
        a, b = cpu_and_mps(name); return a, b
    g.manual_seed(1); cpu = fn()[0]
    # move all generated tensors onto mps by monkeypatching the factories
    g.manual_seed(1)
    global r, i
    orig_r, orig_i = r, i
    try:
        r = lambda *s, dt=torch.float32: orig_r(*s, dt=dt).to("mps")
        i = lambda hi, *s: orig_i(hi, *s).to("mps")
        old = (torch.eye, torch.arange, torch.tensor, torch.zeros, torch.ones, torch.tril_indices)
        def dev(f):
            return lambda *a, **k: f(*a, **{"device": "mps", **k})
        torch.eye, torch.arange, torch.tensor, torch.zeros, torch.ones, torch.tril_indices = map(dev, old)
        try:
            out = fn()[0]
        finally:
            torch.eye, torch.arange, torch.tensor, torch.zeros, torch.ones, torch.tril_indices = old
    finally:
        r, i = orig_r, orig_i
    return cpu, out

if __name__ == "__main__":
    import json, sys
    res = {}
    for name in C:
        if C[name]() == (None,) and name not in ("nn.RNN","nn.LSTM_proj","nn.GRU","nn.MultiheadAttention","nn.Transformer","multinomial","dropout","rrelu_train"):
            continue
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                a, b = run(name)
                if b is None: res[name] = "skip"; continue
                torch.mps.synchronize()
                a, b = a.detach().cpu().double(), b.detach().cpu().double()
                if a.shape != b.shape: res[name] = f"SHAPE {tuple(a.shape)} vs {tuple(b.shape)}"
                elif torch.allclose(a, b, atol=1e-3, rtol=1e-3, equal_nan=True) : res[name] = "ok"
                else: res[name] = f"MISMATCH max|d|={(a-b).abs().max().item():.3g}"
            except Exception as e:
                res[name] = "FAIL " + type(e).__name__ + ": " + str(e).strip().splitlines()[0][:110]
            fb = [str(x.message)[:90] for x in w if "fall" in str(x.message).lower()]
            if fb: res[name] += " | FALLBACK-WARN: " + fb[0]
    for k, v in res.items(): print(f"{k:28s} {v}")
    print("torch", torch.__version__)
