"""CLI: process FocusFix exports, compare models and export Core ML."""
import argparse
import json
from pathlib import Path

from focusfix_data import FEATURE_VERSION, FEATURES, prepare
from modeling import train_models
from synthetic import generate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train"):
        command = commands.add_parser(name)
        command.add_argument("source", type=Path, help="Session folder, recordings tree or exported research ZIP")
        identity = command.add_mutually_exclusive_group(required=True)
        identity.add_argument("--participant-id", help="Pseudonym for all recordings from ONE person")
        identity.add_argument("--manifest", type=Path, help="CSV mapping session_id,participant_id")
        command.add_argument("--skip-invalid", action="store_true", help="Exclude invalid/older/unfinished sessions, recording reasons in ingestion.json")
        command.add_argument("--window-seconds", type=float, default=300)
        command.add_argument("--output", type=Path, required=True)
    training = commands.choices["train"]
    training.add_argument("--group-by", choices=["participant", "session"], default="participant")
    training.add_argument("--models", nargs="+", help="Candidate names; mean_baseline always included")
    training.add_argument("--no-coreml", action="store_true", help="Train only, without exporting")
    demo = commands.add_parser("demo", help="Verify the entire pipeline using synthetic recordings")
    demo.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            if args.output.exists() and any(args.output.iterdir()):
                raise ValueError("Demo output must be new or empty")
            manifest = generate(args.output / "recordings")
            table, ingest = prepare(args.output / "recordings", manifest=manifest)
            report = train_models(table, args.output / "models", synthetic=True)
            print(json.dumps(report, indent=2))
            return
        table, ingest = prepare(args.source, args.participant_id, args.manifest, args.window_seconds, args.skip_invalid)
        if args.command == "prepare":
            if args.output.exists() and any(args.output.iterdir()):
                raise ValueError("Output must be new or empty")
            args.output.mkdir(parents=True, exist_ok=True)
            table.to_csv(args.output / "training_windows.csv", index=False)
            (args.output / "ingestion.json").write_text(json.dumps({"feature_version": FEATURE_VERSION, "features": FEATURES, "window_seconds": args.window_seconds, "sessions": ingest}, indent=2) + "\n")
            print(f"Prepared {len(table)} labelled windows from {len(ingest)} sessions in {args.output}")
        else:
            report = train_models(table, args.output, args.group_by, args.models,
                                  not args.no_coreml, args.window_seconds, synthetic=any(item.get("synthetic", False) for item in ingest))
            (args.output / "ingestion.json").write_text(json.dumps(ingest, indent=2) + "\n")
            for target, result in report["targets"].items():
                print(f"{target}: {result['selected_model']}; test error {result['test_mae']:.3f}/10; average-only error {result['baseline_test_mae']:.3f}/10")
            print(f"Experimental artifacts saved to {args.output}; see report.json")
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
