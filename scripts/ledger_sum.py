import json, sys
for f in sys.argv[1:]:
    d = json.load(open(f)); pr = d['probe']
    keys = [k for k in pr[0] if k not in ('t', 'ocean_cut_nodes')]
    S = {k: sum(r[k] for r in pr) for k in keys}
    print(f"== {f}: {len(pr)} Myr; all in 1e6 km^3 (=1e15 m^3)")
    for k in keys:
        if k == 'ocean_cut_area_frac':
            print(f"  {k:32s} mean {S[k]/len(pr):.4f}")
        else:
            print(f"  {k:32s} {S[k]/1e15:9.1f}   ({S[k]/1e9/len(pr)/1e6:6.2f} km3/yr)")
