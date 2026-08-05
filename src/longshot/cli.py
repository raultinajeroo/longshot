"""longshot command-line interface.

Exit codes: 0 ok; 2 usage error (argparse); 3 venue unavailable;
4 empty/invalid input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .analyze import format_digest, run_analysis
from .horizons import DEFAULT_HORIZONS, parse_horizon
from .simulate import MODES, simulate_markets
from .store import StoreError, load_jsonl, write_jsonl
from .venues import VenueUnavailableError, get_client

EXIT_OK = 0
EXIT_VENUE_UNAVAILABLE = 3
EXIT_BAD_INPUT = 4


def _repo_file(rel: str) -> Path:
    """Resolve a repo-relative data file (cwd first, then package-relative).

    The package-relative fallback assumes a src-layout checkout or editable
    install (repo/src/longshot/cli.py -> repo root is parents[2]).
    """
    cwd = Path.cwd() / rel
    if cwd.exists():
        return cwd
    return Path(__file__).resolve().parents[2] / rel


def _add_common_fetch(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-markets", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="longshot",
        description="Calibration and bias analytics for prediction markets.",
    )
    parser.add_argument("--version", action="version",
                        version=f"longshot {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "fetch", help="download resolved markets from a venue",
        epilog="example: longshot fetch --venue manifold --out "
               "data/manifold.jsonl --max-markets 250")
    p.add_argument("--venue", required=True,
                   choices=["manifold", "polymarket", "kalshi"])
    p.add_argument("--out", required=True)
    p.add_argument("--max-bets-per-market", type=int, default=4000)
    p.add_argument("--max-calls", type=int, default=5000,
                   help="hard cap on HTTP requests")
    _add_common_fetch(p)

    p = sub.add_parser(
        "analyze", help="compute calibration/bias statistics",
        epilog="example: longshot analyze --input data/bundled/"
               "manifold_resolved_sample.jsonl --out analysis.json")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--out")
    p.add_argument("--config",
                   help="pre-registered analysis.yaml; supplies defaults "
                   "for horizons/bins/min_per_bin/bootstrap/seed "
                   "(explicit flags override)")
    p.add_argument("--horizons", default=",".join(DEFAULT_HORIZONS))
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--min-per-bin", type=int, default=30)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser(
        "correct", help="fit bias corrections; evaluate out-of-sample",
        epilog="example: longshot correct --input data/bundled/"
               "manifold_resolved_sample.jsonl --method both "
               "--out correction.json")
    p.add_argument("--input", required=True)
    p.add_argument("--method", choices=["isotonic", "platt", "both"],
                   default="both")
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--out")
    p.add_argument("--config",
                   help="pre-registered analysis.yaml; supplies defaults "
                   "(explicit flags override)")
    p.add_argument("--horizons", default=",".join(DEFAULT_HORIZONS))
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--min-per-bin", type=int, default=30)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser(
        "report", help="render a self-contained HTML report",
        epilog="example: longshot report --analysis analysis.json "
               "--correction correction.json --out report.html")
    p.add_argument("--analysis", required=True)
    p.add_argument("--correction")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="longshot calibration report")

    p = sub.add_parser(
        "publish",
        help="write a publishable note: report.html, report.json, "
        "README-summary.md, provenance.json",
        epilog="example: longshot publish --analysis analysis.json "
               "--correction correction.json --out site/")
    p.add_argument("--analysis", required=True)
    p.add_argument("--correction")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--title", default="longshot calibration report")

    p = sub.add_parser(
        "venues",
        help="print the honest per-venue collector status table")

    p = sub.add_parser(
        "simulate", help="generate synthetic markets",
        epilog="example: longshot simulate --mode calibrated --markets 120 "
               "--seed 7 --out fixtures/calibrated_markets.jsonl")
    p.add_argument("--mode", choices=list(MODES), required=True)
    p.add_argument("--markets", type=int, default=120)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", required=True)

    p = sub.add_parser(
        "demo", help="offline end-to-end demo on bundled data",
        epilog="example: longshot demo --outdir examples/demo")
    p.add_argument("--outdir", default="examples/demo")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return parser


def _load_inputs(paths: list[str]) -> list:
    """Load and concatenate JSONL inputs; exits 4 on invalid/empty input."""
    markets = []
    for path in paths:
        try:
            markets.extend(load_jsonl(path))
        except StoreError as exc:
            print(f"longshot: {exc}", file=sys.stderr)
            raise SystemExit(EXIT_BAD_INPUT)
    if not markets:
        print("longshot: input contained no usable markets", file=sys.stderr)
        raise SystemExit(EXIT_BAD_INPUT)
    return markets


def _cmd_fetch(args: argparse.Namespace) -> int:
    client = get_client(args.venue, max_calls=args.max_calls)
    out = Path(args.out)
    n = 0

    def progress(msg: str) -> None:
        print(msg, file=sys.stderr)

    try:
        markets = client.fetch_resolved(
            max_markets=args.max_markets,
            max_bets_per_market=getattr(args, "max_bets_per_market", 4000),
            seed=args.seed,
            progress_cb=progress,
        )
        n = write_jsonl(out, markets)
    except VenueUnavailableError as exc:
        print(f"longshot: {exc}", file=sys.stderr)
        return EXIT_VENUE_UNAVAILABLE
    if n == 0:
        print("longshot: venue returned no usable markets", file=sys.stderr)
        return EXIT_BAD_INPUT
    print(f"wrote {n} markets to {out}")
    return EXIT_OK


def _horizon_args(args: argparse.Namespace,
                  config: dict | None = None) -> tuple[list[str], dict[str, int]]:
    raw = args.horizons
    if config and raw == ",".join(DEFAULT_HORIZONS) and config.get("horizons"):
        names = [str(h) for h in config["horizons"]]
    else:
        names = [h.strip() for h in raw.split(",") if h.strip()]
    if not names:
        print("longshot: --horizons must name at least one horizon",
              file=sys.stderr)
        raise SystemExit(EXIT_BAD_INPUT)
    try:
        seconds = {h: parse_horizon(h) for h in names}
    except ValueError as exc:
        print(f"longshot: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_BAD_INPUT)
    return names, seconds


def _load_config_or_exit(path) -> dict:
    from .prereg import AnalysisConfigError, load_analysis_config

    try:
        return load_analysis_config(path)
    except AnalysisConfigError as exc:
        print(f"longshot: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_BAD_INPUT)


def _cmd_analyze(args: argparse.Namespace) -> int:
    from .prereg import pick

    config = _load_config_or_exit(getattr(args, "config", None))
    markets = _load_inputs(args.input)
    horizons, _ = _horizon_args(args, config)
    analysis = run_analysis(
        markets, horizons=horizons,
        n_bins=pick(args.bins, 10, config, "bins"),
        min_per_bin=pick(args.min_per_bin, 30, config, "min_per_bin"),
        n_boot=pick(args.bootstrap, 1000, config, "bootstrap"),
        seed=pick(args.seed, 42, config, "seed"),
        label=", ".join(str(p) for p in args.input),
    )
    print(format_digest(analysis))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(analysis, indent=2) + "\n",
                                  encoding="utf-8")
        print(f"\nwrote {args.out}")
    return EXIT_OK


def _cmd_correct(args: argparse.Namespace) -> int:
    from .correct import run_correction
    from .prereg import pick

    config = _load_config_or_exit(getattr(args, "config", None))
    corr_cfg = config.get("correction", {}) if config else {}
    markets = _load_inputs([args.input])
    horizons, seconds = _horizon_args(args, config)
    methods = corr_cfg.get("methods", ["isotonic", "platt"])
    method = (
        args.method
        if args.method != "both" or not corr_cfg.get("methods")
        else ("both" if len(methods) > 1 else methods[0])
    )
    correction = run_correction(
        markets, horizons=horizons, horizon_seconds=seconds,
        method=method,
        train_frac=pick(args.train_frac, 0.6, corr_cfg, "train_frac"),
        n_bins=pick(args.bins, 10, config, "bins"),
        min_per_bin=pick(args.min_per_bin, 30, config, "min_per_bin"),
        n_boot=pick(args.bootstrap, 1000, config, "bootstrap"),
        seed=pick(args.seed, 42, config, "seed"),
    )
    # Print the verdict table via the digest formatter.
    print(format_digest(_minimal_analysis_for_digest(markets, args), correction))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(correction, indent=2) + "\n",
                                  encoding="utf-8")
        print(f"\nwrote {args.out}")
    return EXIT_OK


def _minimal_analysis_for_digest(markets, args) -> dict:
    return {
        "label": str(args.input),
        "dataset": {
            "n_markets": len(markets),
            "n_yes": sum(m.outcome for m in markets),
            "n_no": sum(1 - m.outcome for m in markets),
        },
        "params": {"horizons": []},
        "horizons": {},
    }


def _cmd_report(args: argparse.Namespace) -> int:
    from .report import write_report

    try:
        analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"longshot: cannot read analysis {args.analysis}: {exc}",
              file=sys.stderr)
        return EXIT_BAD_INPUT
    correction = None
    if args.correction:
        try:
            correction = json.loads(
                Path(args.correction).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"longshot: cannot read correction {args.correction}: {exc}",
                  file=sys.stderr)
            return EXIT_BAD_INPUT
    out = write_report(analysis, correction, args.out, title=args.title)
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.json')}")
    return EXIT_OK


def _cmd_publish(args: argparse.Namespace) -> int:
    from .publish import publish

    for path in (args.analysis, args.correction):
        if path and not Path(path).is_file():
            print(f"longshot: input not found: {path}", file=sys.stderr)
            return EXIT_BAD_INPUT
    try:
        written = publish(args.analysis, args.correction, args.out,
                          title=args.title)
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"longshot: cannot publish: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT
    for path in written:
        print(f"wrote {path}")
    return EXIT_OK


def _cmd_venues(args: argparse.Namespace) -> int:
    from .venues.status import venue_status

    print("longshot venue status (honest claims only):")
    for row in venue_status():
        print(f"  {row['venue']:<11} {row['status']}")
    return EXIT_OK


def _cmd_simulate(args: argparse.Namespace) -> int:
    markets = simulate_markets(args.mode, n_markets=args.markets, seed=args.seed)
    n = write_jsonl(args.out, markets)
    print(f"wrote {n} simulated markets ({args.mode}, seed {args.seed}) "
          f"to {args.out}")
    return EXIT_OK


def _cmd_demo(args: argparse.Namespace) -> int:
    from .correct import run_correction
    from .report import write_report

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    bundled = _repo_file("data/bundled/manifold_resolved_sample.jsonl")
    biased = _repo_file("fixtures/biased_markets.jsonl")
    for path in (bundled, biased):
        if not path.exists():
            print(f"longshot: demo input missing: {path}", file=sys.stderr)
            return EXIT_BAD_INPUT

    horizons, seconds = _horizon_args(
        argparse.Namespace(horizons=",".join(DEFAULT_HORIZONS)))

    print(f"longshot demo (offline) — longshot {__version__}")
    print(f"bundled data: {bundled}")
    markets = _load_inputs([str(bundled)])
    analysis = run_analysis(
        markets, horizons=horizons, n_boot=args.bootstrap, seed=args.seed,
        label="Manifold bundled resolved-market sample",
    )
    correction = run_correction(
        markets, horizons=horizons, horizon_seconds=seconds, method="both",
        n_boot=args.bootstrap, seed=args.seed,
    )
    (outdir / "analysis_bundled.json").write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    (outdir / "correction_bundled.json").write_text(
        json.dumps(correction, indent=2) + "\n", encoding="utf-8")
    write_report(analysis, correction, outdir / "report_bundled.html",
                 title="longshot: Manifold bundled sample")

    biased_markets = _load_inputs([str(biased)])
    # The 120-market fixture needs a lower min_per_bin than the bundled set.
    analysis_biased = run_analysis(
        biased_markets, horizons=horizons, min_per_bin=10,
        n_boot=args.bootstrap, seed=args.seed,
        label="simulated compressed-bias fixture (seed 11)",
    )
    (outdir / "analysis_biased_fixture.json").write_text(
        json.dumps(analysis_biased, indent=2) + "\n", encoding="utf-8")

    print()
    print(format_digest(analysis, correction))
    print()
    h_long = horizons[0]  # longest horizon: least terminal drift
    c = analysis_biased["horizons"][h_long]["compression"]
    print(f"biased fixture (planted compression 0.55): slope at {h_long} = "
          f"{c['slope']:.2f} [{c['ci_lo']:.2f}, {c['ci_hi']:.2f}] "
          f"(expect ~1.8 = 1/0.55)")
    print()
    for f in ("analysis_bundled.json", "correction_bundled.json",
              "report_bundled.html", "report_bundled.json",
              "analysis_biased_fixture.json"):
        print(f"wrote {outdir / f}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    handler = {
        "fetch": _cmd_fetch,
        "analyze": _cmd_analyze,
        "correct": _cmd_correct,
        "report": _cmd_report,
        "publish": _cmd_publish,
        "venues": _cmd_venues,
        "simulate": _cmd_simulate,
        "demo": _cmd_demo,
    }[args.command]
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
