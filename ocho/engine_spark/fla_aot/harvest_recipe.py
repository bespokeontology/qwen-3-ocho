# Harvest FLA's chunk_gated_delta_rule Triton launches on this GPU: cubin + entry + launch geometry +
# ordered runtime params (by name/kind) + constexprs + int specializations, per chunk length class.
import torch, math, json, os, sys, inspect
import triton
from triton.runtime.jit import JITFunction
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
OUT = 'fla_aot'
records = []
orig_run = JITFunction.run
def hooked_run(self, *args, grid, warmup, **kwargs):
    kernel = orig_run(self, *args, grid=grid, warmup=warmup, **kwargs)
    if kernel is None: return kernel
    # bind args to parameter names in signature order
    params = list(inspect.signature(self.fn).parameters.keys())
    bound = {}
    for i, a in enumerate(args): bound[params[i]] = a
    bound.update({k: v for k, v in kwargs.items() if k in params})
    meta = {k: v for k, v in kwargs.items() if k not in params}   # num_warps/num_stages etc
    # evaluate grid
    g = grid(dict(kwargs)) if callable(grid) else grid
    g = tuple(int(x) for x in (g if isinstance(g, (tuple, list)) else (g,)))
    sig = kernel.src.signature if hasattr(kernel, 'src') else {}
    consts = kernel.src.constants if hasattr(kernel, 'src') else {}
    plist = []
    for name in params:
        ty = sig.get(name, '?')
        v = bound.get(name, None)
        if ty == 'constexpr':
            cv = consts.get(name, v) if isinstance(consts, dict) else v
            plist.append({'name': name, 'kind': 'constexpr', 'value': (cv if isinstance(cv, (int, float, bool, str, type(None))) else str(cv))})
        else:
            if isinstance(v, torch.Tensor): kind = 'ptr'; val = str(v.dtype); 
            elif isinstance(v, bool): kind = 'i1'; val = int(v)
            elif isinstance(v, int): kind = 'int'; val = v
            elif isinstance(v, float): kind = 'fp32'; val = v
            elif v is None: kind = 'none'; val = None
            else: kind = 'other'; val = str(v)
            plist.append({'name': name, 'kind': kind, 'sigtype': ty, 'value': val})
    md = kernel.metadata
    ptx = kernel.asm['ptx']; ent = ptx[ptx.index('.visible .entry'):]; ent = ent[:ent.index('{')]
    rec = {'fn': self.fn.__name__, 'entry': md.name, 'num_warps': md.num_warps, 'shared': md.shared,
           'global_scratch': md.global_scratch_size, 'profile_scratch': getattr(md, 'profile_scratch_size', 0),
           'ptx_params': ent.count('.param'), 'grid': g, 'meta': {k: (v if isinstance(v, (int, float, bool, str)) else str(v)) for k, v in meta.items()},
           'params': plist, 'hash': kernel.hash}
    cubin = kernel.asm['cubin']
    fn_c = f"{OUT}/{md.name}_{kernel.hash[:12]}.cubin"
    if not os.path.exists(fn_c): open(fn_c, 'wb').write(cubin)
    rec['cubin'] = os.path.basename(fn_c)
    records.append(rec)
    return kernel
JITFunction.run = hooked_run
T0, HV, D = 2048, 48, 128
q = torch.randn(1, T0, HV, D, device='cuda').bfloat16(); k = torch.randn(1, T0, HV, D, device='cuda').bfloat16(); v = torch.randn(1, T0, HV, D, device='cuda').bfloat16()
g = -torch.rand(1, T0, HV, device='cuda'); beta = torch.rand(1, T0, HV, device='cuda'); h0 = torch.randn(1, HV, D, D, device='cuda')
Ts = [int(x) for x in sys.argv[1:]] or [2048, 1893, 256, 125, 64, 32, 16]
manifest = {}
for T in Ts:
    records.clear()
    o, S = chunk_gated_delta_rule(q[:, :T].contiguous(), k[:, :T].contiguous(), v[:, :T].contiguous(), g[:, :T].contiguous(), beta[:, :T].contiguous(), scale=1/math.sqrt(D), initial_state=h0, output_final_state=True)
    torch.cuda.synchronize()
    # keep only the LAST launch per kernel fn (autotuning may launch several configs at first)
    last = {}
    for r in records: last[r['fn']] = r
    manifest[str(T)] = [last[k] for k in last]
    print(f"T={T}: {len(records)} launches, kernels: {[ (r['fn'], r['grid'], r['num_warps'], r['shared']) for r in manifest[str(T)] ]}")
json.dump(manifest, open(f'{OUT}/manifest.json', 'w'), indent=1)
r = manifest[str(Ts[0])]
for rec in r:
    print("==", rec['fn'], "entry", rec['entry'], "ptx_params", rec['ptx_params'], "gscratch", rec['global_scratch'], "pscratch", rec['profile_scratch'], "meta", rec['meta'])
    print("   params:", [(p['name'], p['kind'], p.get('sigtype'), p.get('value')) for p in rec['params']])
