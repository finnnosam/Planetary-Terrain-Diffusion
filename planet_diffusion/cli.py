import argparse
import json
import math
import sys
from .storage import load_state, save_state, export_tiff, verify_state, load_climate


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate spherical elevation with Terrain Diffusion")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="Generate a globe or regional spherical GeoTIFF")
    gen.add_argument("--state", required=True)
    gen.add_argument("--output", required=True)
    gen.add_argument("--climate-output", help="Also export a five-band climate GeoTIFF (cube geometry)")
    gen.add_argument("--seed", default="random", help="Unsigned integer, or random (default); printed and saved")
    gen.add_argument("--draft", help="2:1 global PNG: black ocean, lighter shades higher land")
    gen.add_argument("--conditioning-dir", help="Folder of global elevation/climate conditioning TIFFs; alternative to --draft")
    gen.add_argument("--snr", help="Five comma-separated TIFF refinement values: elevation, temperature, T std, precipitation, P CV (default .2,.2,1,.2,1)")
    gen.add_argument("--draft-ocean-depth", type=float, default=0., help="Ocean depth prior in metres; 0 = automatic learned bathymetry (default)")
    gen.add_argument("--draft-white-metres", type=float, default=4000., help="Elevation assigned to white (default 4000 m)")
    gen.add_argument("--draft-refinement", type=float, default=.2, help="Upstream conditioning noise, 0.01..4; smaller follows guide more closely (default .2)")
    gen.add_argument("--coarse-height", type=int, help="Logical guide height, multiple of 4; default 8 dense or 1024 regional-only")
    gen.add_argument("--regional-only", action="store_true", help="Use sparse fixed-grid guides; requires --bounds")
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
    gen.add_argument("--upstream", default="upstream")
    gen.add_argument("--model", default="xandergos/terrain-diffusion-90m")
    from .backends import MODEL_REVISION
    gen.add_argument("--revision", default=MODEL_REVISION, help="HF revision (default: pinned 90m checkpoint)")
    gen.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
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
    query.add_argument("--output", required=True)
    query.add_argument("--climate-output", help="Also export five-band climate from the saved state")
    query.add_argument("--upstream", default="upstream")
    check = sub.add_parser("verify", help="Check saved state's sphere topology and integrity")
    check.add_argument("--state", required=True)
    sub.add_parser("gui", help="Open the desktop launcher")
    args = parser.parse_args(argv)
    try:
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
            from .query import load_query, query_region
            if Path(args.output).exists():
                raise ValueError("Output path must not already exist")
            directory, identity, _, _, _ = load_query(args.state,args.checkpoint_dir,
                                                       with_climate=bool(args.climate_output))
            if identity.get("backend") == DiagnosticBackend.name:
                backend = DiagnosticBackend()
            elif identity.get("backend") == TerrainBackend.name:
                import torch
                torch.set_num_threads(int(identity["torch_threads"]))
                backend = TerrainBackend(args.upstream,identity["model"],
                                         identity["model_revision"],identity["device"])
            else:
                raise ValueError("Unsupported checkpoint backend")
            elevation, metadata, climate = query_region(
                backend,args.state,args.bounds,args.width,args.height,directory,
                with_climate=bool(args.climate_output),
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
            from .seeds import resolve_seed
            checkpoint_dir = args.checkpoint_dir or (args.state+".checkpoints" if args.geometry == "cube" else None)
            if args.coarse_height is None:
                args.coarse_height = 1024 if args.regional_only else 8
            if args.regional_only and (args.geometry != "cube" or args.bounds is None):
                raise ValueError("--regional-only requires cube geometry and --bounds")
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
            native = args.coarse_height*256
            height = args.height if args.height is not None else native
            region = None
            if args.bounds is not None:
                from .region import validate_region
                if args.geometry != "cube" or args.width is None or args.height is None:
                    raise ValueError("Regional generation requires cube geometry, --width and --height")
                bounds = validate_region(args.bounds,args.width,args.height,native)
                region = (args.bounds,args.width,args.height)
            elif args.width is not None:
                raise ValueError("--width requires --bounds")
            elif not 2 <= height <= native:
                raise ValueError("height must be between 2 and coarse-height*256")
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
                draft = Draft(args.draft,args.draft_ocean_depth,args.draft_white_metres,args.draft_refinement)
                if args.coarse_height < 16:
                    print("Draft note: use coarse-height 16 or 32 for more recognizable global structure; each guide cell produces 256 output pixels.",file=sys.stderr)
            if region:
                print(f"Regional bounds: {bounds}; export: {args.width} x {height}; global guide height: {args.coarse_height}",file=sys.stderr)
            else:
                print(f"Native state: {2*native} x {native}; export: {2*height} x {height}", file=sys.stderr)
            print(f"Equatorial native spacing: {math.pi*args.radius_metres/native:.2f} m; scales with latitude", file=sys.stderr)
            if args.backend == "diagnostic":
                backend = DiagnosticBackend()
            else:
                import torch
                torch.set_num_threads(args.threads)
                backend = TerrainBackend(args.upstream, args.model, args.revision, args.device)
            progress = lambda s: print(s, file=sys.stderr, flush=True)
            climate = None
            if args.geometry == "cube":
                if args.regional_only:
                    from .sparse import SparseWorld
                    world = SparseWorld(backend,args.seed,args.coarse_height,args.coarse_steps,
                        args.radius_metres,checkpoint_dir,source=conditioning or draft)
                    a,metadata,climate = world.region(bounds,args.width,height,
                        with_climate=bool(args.climate_output),progress=progress)
                else:
                    from .cube_generate import generate_cube
                    a, metadata, climate = generate_cube(backend,args.seed,args.coarse_height//2,args.coarse_steps,
                        progress=progress,checkpoint_dir=checkpoint_dir,draft=draft,region=region,
                        latent_batch_size=args.latent_batch_size,with_climate=True,conditioning=conditioning)
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
