import argparse
import json
import math
import sys
from .storage import load_state, save_state, export_tiff, verify_state, load_climate


# Max pooling keeps any land in a k x k block; larger blocks erase oceans.
MAX_POOL_LIMIT = 3


def cache_budgets(args):
    budgets = {name:getattr(args,name.replace('_bytes','_mib'))
               for name in ('patch_cache_bytes','blend_cache_bytes','decoder_cache_bytes')}
    if any(value < 0 for value in budgets.values()):
        raise ValueError("Cache budgets must be nonnegative")
    return {name:value*1024**2 for name,value in budgets.items()}


def confirm_region(bounds, width, height, n, allowed=False):
    from .region import estimate_decoder_patches
    count = estimate_decoder_patches(bounds,width,height,n,stop_after=100000)
    if count <= 2000:
        print(f"Regional decoder estimate: about {count:,} unique patches",file=sys.stderr,flush=True)
        return
    amount = f"about {count:,}" if count <= 100000 else "more than 100,000"
    print(f"Regional decoder estimate: {amount} unique patches",file=sys.stderr,flush=True)
    if allowed:
        return
    if not sys.stdin.isatty():
        raise ValueError("Large regional generation requires --allow-large-region in non-interactive use")
    if input("Generate this large region? [y/N] ").strip().lower() not in ("y","yes"):
        raise ValueError("Regional generation cancelled")


def main(argv=None, backend_factory=None):
    parser = argparse.ArgumentParser(description="Generate spherical elevation with Terrain Diffusion")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="Generate a globe or regional spherical GeoTIFF")
    gen.add_argument("--state", required=True)
    gen.add_argument("--output", required=True)
    gen.add_argument("--climate-output", help="Also export a five-band climate GeoTIFF (cube geometry)")
    gen.add_argument("--seed", default="random", help="Unsigned integer, or random (default); printed and saved")
    gen.add_argument("--draft", help="2:1 global PNG: brightness maps linearly between black and white elevations")
    gen.add_argument("--conditioning-dir", help="Folder of global elevation/climate conditioning TIFFs; alternative to --draft")
    gen.add_argument("--snr", help="Five comma-separated TIFF refinement values: elevation, temperature, T std, precipitation, P CV (default .2,.2,1,.2,1)")
    endpoints = gen.add_mutually_exclusive_group()
    endpoints.add_argument("--draft-black-metres", type=float, help="Elevation assigned to black (default -2000 m); grayscale is interpolated linearly")
    endpoints.add_argument("--draft-ocean-depth", type=float, default=0., help="Compatibility option: positive depth sets black to its negative; 0 uses -2000 m")
    gen.add_argument("--draft-white-metres", type=float, default=4000., help="Elevation assigned to white (default 4000 m)")
    gen.add_argument("--draft-refinement", type=float, default=.2, help="Upstream conditioning noise, 0.01..4; smaller follows guide more closely (default .2)")
    gen.add_argument("--coarse-height", type=int, help="Logical guide height, multiple of 4; regional default 1024 (90 m) or 2560 (30 m)")
    gen.add_argument("--coarse-pooling", default="1",
                     help="Run the coarse model on a K-times finer guide and pool KxK cells (upstream coarse_pooling). "
                          "auto picks K so coarse cells match the model's training size at --radius-metres. Default 1")
    gen.add_argument("--coarse-pool-mode", choices=["avg", "max"], default="avg",
                     help="avg averages all channels; max uses max elevation and min p5 (more extreme relief)")
    gen.add_argument("--continents", action="store_true",
                     help="Continent passes: run the coarse model on a small guide, then refine it through finer guides "
                          "up to the coarse stage, for planet-scale layout (instead of --draft or --conditioning-dir)")
    gen.add_argument("--continent-guide-height", type=int, default=48,
                     help="Guide height of the first (rough) pass, multiple of 4; smaller gives fewer, larger continents (default 48)")
    gen.add_argument("--continent-step", type=float, default=3.,
                     help="Largest guide-height ratio between continent passes, 1.5-16; smaller adds passes (default 3)")
    gen.add_argument("--continent-relief", type=float, default=1.,
                     help="Scale for final guide land elevation, 0-4 (default 1)")
    gen.add_argument("--continent-refinement", type=float,
                     help="Elevation conditioning noise for the real pass, 0.01-4 (default: model setting)")
    gen.add_argument("--coarse-only", action="store_true",
                     help="Stop after the coarse stage and export its whole-globe elevation (fast layout preview); "
                          "rerun without it using the same --checkpoint-dir to continue")
    gen.add_argument("--regional-only", action="store_true", help="Use sparse fixed-grid guides; requires --bounds")
    gen.add_argument('--preview',action='store_true',help='Choose a cheaper regional grid matching output pixel density')
    gen.add_argument("--allow-large-region", action="store_true", help="Allow a region estimated above 2,000 decoder patches")
    gen.add_argument("--coarse-steps", type=int, default=20)
    gen.add_argument("--latent-batch-size", type=int, default=1,
                     help="Cube latent patches per model call; larger batches use more memory (default: 1)")
    gen.add_argument("--height", type=int, help="Output pixel height; defaults to native height")
    gen.add_argument("--bounds", nargs=4, type=float, metavar=("WEST","SOUTH","EAST","NORTH"),
                     help="Regional degrees; east < west crosses the date line. Cube geometry only")
    gen.add_argument("--width", type=int, help="Regional output width in pixels (requires --bounds and --height)")
    gen.add_argument("--radius-metres", type=float, default=6371000.)
    gen.add_argument("--backend", choices=["terrain", "diagnostic"], default="terrain")
    gen.add_argument("--geometry", choices=["cube", "equirectangular"], default="cube",
                     help="cube generates on six spherical charts; equirectangular is legacy v2")
    gen.add_argument("--checkpoint-dir", help="Cube checkpoints (default: STATE.checkpoints); legacy decoder cache")
    gen.add_argument("--keep-checkpoints", action="store_true",
                     help="Keep whole-globe checkpoints after a successful run (debugging, saved-run queries). By default "
                          "they are used for resuming and then removed; coarse-only previews always keep their small ones")
    gen.add_argument("--upstream", default="upstream")
    from .model_presets import MODEL_90M
    gen.add_argument("--model", default=MODEL_90M,
                     help="Upstream model ID or local folder; 30 m and 90 m checkpoints are supported")
    gen.add_argument("--revision", help="HF revision (default: pinned revision for 30 m or 90 m model)")
    gen.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    gen.add_argument("--precision", choices=["auto", "float32", "tf32", "bfloat16"], default="auto",
                     help="Auto uses native CUDA bfloat16 when supported, otherwise float32")
    gen.add_argument("--patch-cache-mib", type=int, default=64)
    gen.add_argument("--blend-cache-mib", type=int, default=8)
    gen.add_argument("--decoder-cache-mib", type=int, default=64)
    gen.add_argument("--threads", type=int, default=4, help="CPU torch threads (default: 4)")
    tile = sub.add_parser("export", help="Export native-resolution tiles or another overview from saved state")
    tile.add_argument("--state", required=True)
    tile.add_argument("--output", required=True)
    tile.add_argument("--climate-output", help="Also export saved climate on the same grid as elevation")
    tile.add_argument("--height", type=int, help="Global grid height; default is native resolution")
    tile.add_argument("--window", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    query = sub.add_parser("query", help="Decode another region from saved cube guides and decoder cache")
    query.add_argument("--state", required=True, help="Existing completed cube state")
    query.add_argument("--checkpoint-dir", help="Generation checkpoints; inferred beside --state by default")
    query.add_argument("--bounds", nargs=4, required=True, type=float, metavar=("WEST","SOUTH","EAST","NORTH"))
    query.add_argument("--width", required=True, type=int)
    query.add_argument("--height", required=True, type=int)
    query.add_argument("--allow-large-region", action="store_true", help="Allow a region estimated above 2,000 decoder patches")
    query.add_argument("--output", required=True)
    query.add_argument("--climate-output", help="Also export five-band climate from the saved state")
    query.add_argument("--upstream", default="upstream")
    query.add_argument("--patch-cache-mib", type=int, default=64)
    query.add_argument("--blend-cache-mib", type=int, default=8)
    query.add_argument("--decoder-cache-mib", type=int, default=64)
    check = sub.add_parser("verify", help="Check saved state's sphere topology and integrity")
    check.add_argument("--state", required=True)
    sub.add_parser("gui", help="Open the desktop launcher")
    args = parser.parse_args(argv)
    try:
        if args.command in ('generate','query'):
            budgets = cache_budgets(args)
        if args.command == "gui":
            from .gui import main as gui_main
            return gui_main()
        if args.command in ("generate","export","query"):
            from pathlib import Path
            if args.climate_output:
                if Path(args.climate_output).resolve() in (Path(args.output).resolve(),Path(args.state).resolve()):
                    raise ValueError("Climate output must differ from elevation output and state paths")
                if Path(args.climate_output).exists():
                    raise ValueError("Climate output path must not already exist")
        if args.command == "query":
            from pathlib import Path
            from .backends import DiagnosticBackend, TerrainBackend
            make_backend = backend_factory or TerrainBackend
            from .query import load_query, query_loaded_region
            if Path(args.output).exists():
                raise ValueError("Output path must not already exist")
            saved = json.loads((Path(args.state)/"planet.json").read_text(encoding="utf-8"))
            confirm_region(args.bounds,args.width,args.height,saved["face_native_intervals"],
                           args.allow_large_region)
            loaded = load_query(args.state,args.checkpoint_dir,
                                with_climate=bool(args.climate_output))
            _, identity, _, _, _ = loaded
            if identity.get("backend") == DiagnosticBackend.name:
                backend = DiagnosticBackend()
            elif identity.get("backend") == TerrainBackend.name:
                import torch
                torch.set_num_threads(int(identity["torch_threads"]))
                backend = make_backend(args.upstream,identity["model"],
                    identity["model_revision"],identity["device"],
                    precision=identity.get("precision","float32"))
            else:
                raise ValueError("Unsupported checkpoint backend")
            if identity.get('backend') == TerrainBackend.name:
                print(f"Compute: {backend.metadata['device']}; precision: {backend.precision} (saved run)",
                      file=sys.stderr,flush=True)
            elevation, metadata, climate = query_loaded_region(
                backend,loaded,args.bounds,args.width,args.height,
                with_climate=bool(args.climate_output),
                cache_budgets=cache_budgets(args),
                progress=lambda s:print(s,file=sys.stderr,flush=True))
            export_tiff(args.output,elevation,metadata)
            if args.climate_output:
                export_tiff(args.climate_output,elevation,metadata,climate=climate)
            print(json.dumps({"bounds":metadata["bounds"],"width":args.width,
                              "height":args.height,"regional_execution":metadata["regional_execution"]},indent=2))
            return 0
        if args.command == "generate":
            from pathlib import Path
            from .backends import DiagnosticBackend, TerrainBackend
            from .generate import generate
            from .model_presets import default_revision, sparse_guide_height
            from .seeds import resolve_seed
            checkpoint_dir = args.checkpoint_dir or (args.state+".checkpoints" if args.geometry == "cube" else None)
            if args.preview:
                if not args.regional_only or args.bounds is None or args.coarse_height is not None:
                    raise ValueError('--preview requires --regional-only and --bounds; omit --coarse-height')
                from .region import preview_guide_height
                args.coarse_height = preview_guide_height(args.bounds,args.width,args.height)
            if args.coarse_height is None:
                args.coarse_height = sparse_guide_height(args.model) if args.regional_only else 8
            if args.regional_only and (args.geometry != "cube" or args.bounds is None):
                raise ValueError("--regional-only requires cube geometry and --bounds")
            from .model_presets import parse_coarse_pooling
            pooling = parse_coarse_pooling(args.coarse_pooling)
            if pooling != 1 and (args.regional_only or args.geometry != "cube"):
                raise ValueError("--coarse-pooling requires whole-globe cube generation without --regional-only")
            if (args.continents or args.coarse_only) and (args.regional_only or args.geometry != "cube"):
                raise ValueError("--continents and --coarse-only require cube generation without --regional-only")
            if args.continents and (args.draft or args.conditioning_dir):
                raise ValueError("--continents replaces --draft and --conditioning-dir; choose one")
            if args.coarse_only and (args.bounds is not None or args.climate_output):
                raise ValueError("--coarse-only exports whole-globe elevation; omit --bounds and --climate-output")
            from .model_presets import native_metres, auto_coarse_pooling, model_guide_height
            model_metres = native_metres(args.model)
            if pooling == "auto":
                pooling = auto_coarse_pooling(args.radius_metres,args.coarse_height,model_metres)
            if args.coarse_pool_mode == "max" and pooling > MAX_POOL_LIMIT:
                raise ValueError(f"--coarse-pool-mode max with pooling {pooling} keeps the highest of {pooling*pooling} "
                                 f"cells, which turns most ocean into land; use avg, or pooling <= {MAX_POOL_LIMIT}")
            args.seed = resolve_seed(args.seed,checkpoint_dir if args.geometry == "cube" else None)
            print(f"Seed: {args.seed}",file=sys.stderr,flush=True)
            if Path(args.state).exists() or Path(args.output).exists():
                raise ValueError("State and output paths must not already exist")
            if not math.isfinite(args.radius_metres) or args.radius_metres <= 0:
                raise ValueError("radius-metres must be positive and finite")
            if args.coarse_height < 2 or args.coarse_height % 2 or args.coarse_steps < 2 or not 0 <= args.seed < 2**64:
                raise ValueError("Use even coarse-height >= 2, coarse-steps >= 2 and unsigned 64-bit seed")
            if args.geometry == "cube" and (args.coarse_height < 4 or args.coarse_height % 4):
                raise ValueError("Cube geometry requires coarse-height divisible by 4 and >= 4")
            native = args.coarse_height*(pooling if args.coarse_only else 256)
            height = args.height if args.height is not None else native
            region = None
            if args.bounds is not None:
                from .region import validate_region
                if args.geometry != "cube" or args.width is None or args.height is None:
                    raise ValueError("Regional generation requires cube geometry, --width and --height")
                bounds = validate_region(args.bounds,args.width,args.height,native)
                region = (args.bounds,args.width,args.height)
                confirm_region(bounds,args.width,args.height,args.coarse_height*128,
                               args.allow_large_region)
            elif args.width is not None:
                raise ValueError("--width requires --bounds")
            elif not 2 <= height <= native:
                raise ValueError("height must be between 2 and the native height"
                                 + (" (coarse-height x pooling with --coarse-only)" if args.coarse_only else " (coarse-height*256)"))
            if args.threads < 1:
                raise ValueError("threads must be positive")
            if args.latent_batch_size < 1:
                raise ValueError("latent-batch-size must be positive")
            if args.geometry != "cube" and args.latent_batch_size != 1:
                raise ValueError("latent-batch-size requires cube geometry")
            if args.geometry != "cube" and args.climate_output:
                raise ValueError("Climate output requires cube geometry")
            conditioning = None
            if args.snr is not None and not args.conditioning_dir:
                raise ValueError("--snr requires --conditioning-dir")
            if args.conditioning_dir:
                if args.geometry != "cube" or args.draft:
                    raise ValueError("TIFF conditioning requires cube geometry and cannot be combined with --draft")
                from .conditioning import TiffConditioning, DEFAULT_SNR
                conditioning = TiffConditioning(args.conditioning_dir,args.snr if args.snr is not None else DEFAULT_SNR)
            draft = None
            if args.draft:
                if args.geometry != "cube":
                    raise ValueError("PNG draft import requires cube geometry")
                from .draft import Draft
                draft = Draft(args.draft,args.draft_ocean_depth,args.draft_white_metres,args.draft_refinement,
                              black_metres=args.draft_black_metres)
                if args.coarse_height < 16:
                    print("Draft note: use coarse-height 16 or 32 for more recognizable global structure; each guide cell produces 256 output pixels.",file=sys.stderr)
            if region:
                print(f"Regional bounds: {bounds}; export: {args.width} x {height}; global guide height: {args.coarse_height}",file=sys.stderr)
            else:
                print(f"{'Coarse-only' if args.coarse_only else 'Native'} state: {2*native} x {native}; "
                      f"export: {2*height} x {height}", file=sys.stderr)
            print(f"Equatorial native spacing: {math.pi*args.radius_metres/native:.2f} m; scales with latitude", file=sys.stderr)
            model_cell = 256*model_metres
            layout_cell = math.pi*args.radius_metres/(args.coarse_height*pooling)
            print(f"Coarse pooling {pooling} ({args.coarse_pool_mode}): coarse model guide height "
                  f"{args.coarse_height*pooling}; layout cells {layout_cell/1000:.1f} km vs model "
                  f"{model_cell/1000:.2f} km (x{layout_cell/model_cell:.2f}); detail scale "
                  f"x{layout_cell*pooling/model_cell:.2f} of model", file=sys.stderr)
            if pooling == 1 and layout_cell/model_cell > 2:
                ideal = model_guide_height(args.radius_metres,model_metres)
                print(f"Scale note: terrain is x{layout_cell/model_cell:.1f} the model's horizontal scale. "
                      f"Model-scale layout needs guide height ~{ideal:.0f}; try --coarse-pooling auto "
                      f"(or radius {args.coarse_height*model_cell/math.pi/1000:.0f} km for this guide).",
                      file=sys.stderr)
            if args.backend == "diagnostic":
                backend = DiagnosticBackend()
            else:
                import torch
                torch.set_num_threads(args.threads)
                backend = (backend_factory or TerrainBackend)(args.upstream, args.model,
                    args.revision or default_revision(args.model), args.device, precision=args.precision)
                print(f"Compute: {backend.metadata['device']}; precision: {backend.precision}",
                      file=sys.stderr,flush=True)
            progress = lambda s: print(s, file=sys.stderr, flush=True)
            continents = None
            if args.continents:
                from .continents import ContinentGuide
                continents = ContinentGuide(backend,args.continent_guide_height,args.continent_relief,
                                            args.continent_step,args.coarse_steps,args.continent_refinement,progress)
                rough_cell = math.pi*args.radius_metres/args.continent_guide_height
                print(f"Continent passes: rough guide {args.continent_guide_height} (cells {rough_cell/1000:.0f} km), "
                      f"step {args.continent_step}, relief {args.continent_relief}", file=sys.stderr)
            climate = None
            if args.geometry == "cube":
                if args.regional_only:
                    from .sparse import SparseWorld
                    world = SparseWorld(backend,args.seed,args.coarse_height,args.coarse_steps,
                        args.radius_metres,checkpoint_dir,source=conditioning or draft,
                        latent_batch_size=args.latent_batch_size, **cache_budgets(args))
                    a,metadata,climate = world.region(bounds,args.width,height,
                        with_climate=bool(args.climate_output),progress=progress)
                else:
                    from .cube_generate import generate_cube
                    generated = generate_cube(backend,args.seed,args.coarse_height//2,args.coarse_steps,
                        progress=progress,checkpoint_dir=checkpoint_dir,draft=draft,region=region,
                        latent_batch_size=args.latent_batch_size,
                        decoder_cache_bytes=budgets['decoder_cache_bytes'],
                        with_climate=bool(args.climate_output),conditioning=conditioning,
                        coarse_pooling=pooling,coarse_pool_mode=args.coarse_pool_mode,
                        continents=continents,coarse_only=args.coarse_only,
                        keep_checkpoints=args.keep_checkpoints)
                    if args.climate_output:
                        a, metadata, climate = generated
                    else:
                        a, metadata = generated
            else:
                a, metadata = generate(backend,args.seed,args.coarse_height,args.coarse_steps,
                    progress=progress,decoder_cache=args.checkpoint_dir)
            metadata["radius_metres"] = args.radius_metres
            save_state(args.state, a, metadata, climate=None if args.regional_only else climate)
            if draft is not None:
                (Path(args.state)/"draft.png").write_bytes(draft.png_bytes)
            if conditioning is not None:
                conditioning.snapshot(args.state)
            a, metadata = load_state(args.state)
            export_tiff(args.output, a, metadata, height)
            if args.climate_output:
                progress("Exporting five-band climate GeoTIFF")
                export_tiff(args.climate_output,a,metadata,height,climate=climate)
            result = verify_state(a, metadata)
            print(json.dumps(result, indent=2))
            if (result["passed"] and args.geometry == "cube" and not args.regional_only
                    and not args.coarse_only and not args.keep_checkpoints):
                from .cube_generate import remove_checkpoints
                freed = remove_checkpoints(checkpoint_dir)
                progress(f"Removed checkpoints ({freed/1024**2:,.0f} MiB); use --keep-checkpoints to keep them "
                         "for debugging or saved-run queries")
            return 0 if result["passed"] else 1
        else:
            a, metadata = load_state(args.state)
            if args.command == "export":
                climate = load_climate(args.state,metadata) if args.climate_output else None
                export_tiff(args.output, a, metadata, args.height, args.window)
                if args.climate_output:
                    export_tiff(args.climate_output,a,metadata,args.height,args.window,climate=climate)
            else:
                result = verify_state(a, metadata)
                print(json.dumps(result, indent=2))
                return 0 if result["passed"] else 1
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0
