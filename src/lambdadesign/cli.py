"""Command line entry points for the inference package."""

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


def parser():
    root = argparse.ArgumentParser(prog="lambdadesign", description="LambdaDesign HeLa/mRNA inference")
    root.add_argument("--version", action="version", version="%(prog)s 1.0.1")
    commands = root.add_subparsers(dest="command", required=True)
    for name, help_text in (("score", "Predict activity from SMILES"),
                            ("generate", "Generate candidates from activity-filtered latent vectors"),
                            ("inspect", "Inspect checkpoint architecture and statistics")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--checkpoint", required=True, type=Path, help="LambdaDesign design checkpoint (external file)")
        command.add_argument("--device", default="auto", help="auto, cpu, or cuda:N")
        command.add_argument("--threads", default=4, type=int, help="PyTorch CPU threads")
        if name == "inspect":
            command.add_argument("--molclr-checkpoint", type=Path)
            continue
        command.add_argument("--output", required=True, type=Path, help="Output CSV; a JSON run summary is saved alongside it")
        command.add_argument("--overwrite", action="store_true")
        if name == "score":
            command.add_argument("--molclr-checkpoint", required=True, type=Path)
            command.add_argument("--input", required=True, type=Path)
            command.add_argument("--smiles-column", default="smiles")
            command.add_argument("--batch-size", default=32, type=int)
            command.add_argument("--canonicalize", action="store_true", help="Canonicalize atom order before encoding; default preserves input order")
        else:
            command.add_argument("--config", type=Path, help="JSON with GenerationOptions fields")
            command.add_argument("--num-molecules", type=int)
            command.add_argument("--max-latents", type=int)
            command.add_argument("--latent-batch-size", type=int)
            command.add_argument("--decode-batch-size", type=int)
            command.add_argument("--samples-per-latent", type=int)
            command.add_argument("--max-tokens", type=int)
            command.add_argument("--seed", type=int)
            command.add_argument("--temperature", type=float)
            command.add_argument("--top-k", type=int)
            command.add_argument("--activity-threshold", help="p75, p90, p95, mean, a number, or none")
    return root


def environment():
    return {"python":platform.python_version(), "platform":platform.platform(),
            "packages":{name:importlib.metadata.version(name)
                        for name in ("torch", "rdkit", "selfies", "numpy", "lambdadesign-inference")}}


def output_paths(args):
    output = args.output.expanduser()
    if output.suffix.lower() != ".csv":
        raise ValueError("--output must have a .csv extension")
    summary = output.with_suffix(".json")
    if getattr(args, "input", None) and output.resolve() == args.input.expanduser().resolve():
        raise ValueError("Input and output must be different files")
    for path in (output, summary):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {path}; choose another name or pass --overwrite")
    return output, summary


def write_results(output, summary_path, rows, columns, summary):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    summary["environment"] = environment()
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def run(args):
    import torch
    from .inference import LambdaDesign
    from .generation import GenerationOptions

    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command != "inspect":
        output, summary_path = output_paths(args)
    model = LambdaDesign.from_checkpoint(args.checkpoint, getattr(args, "molclr_checkpoint", None), args.device)
    if args.command == "inspect":
        print(json.dumps(model.describe(), indent=2, allow_nan=False)); return 0
    if args.command == "score":
        with args.input.expanduser().open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames
            if not columns or args.smiles_column not in columns:
                raise ValueError(f"Input CSV must contain column {args.smiles_column!r}")
            if len(columns) != len(set(columns)):
                raise ValueError("Input CSV has duplicate column names")
            reserved = {"canonical_smiles", "activity_score", "predicted_activity", "status", "error"}
            if reserved.intersection(columns):
                raise ValueError(f"Input uses reserved output columns: {sorted(reserved.intersection(columns))}")
            records = list(reader)
        if not records:
            raise ValueError("Input CSV has no data rows")
        if any(None in row or any(v is None for v in row.values()) for row in records):
            raise ValueError("Input CSV contains rows with the wrong number of fields")
        predictions = model.score([r[args.smiles_column] for r in records], args.batch_size, args.canonicalize)
        rows = [{**row, **{k:v for k,v in prediction.items() if k != "input_smiles"}}
                for row,prediction in zip(records, predictions)]
        counts = dict(Counter(p["status"] for p in predictions))
        summary = dict(model.describe(), input_rows=len(rows), status_counts=counts,
                       complete=counts.get("ok",0)==len(rows),
                       score_definition="activity_predictor(posterior_mean)",
                       canonicalize=args.canonicalize, batch_size=args.batch_size)
        write_results(output, summary_path, rows,
                      columns+["canonical_smiles", "activity_score", "predicted_activity", "status", "error"], summary)
        print(json.dumps({"output":str(output), "summary":str(summary_path), "status_counts":counts}))
        return 0 if counts.get("ok",0)>0 else 3
    values = {}
    if args.config:
        values = json.loads(args.config.expanduser().read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Generation config must be a JSON object")
    allowed = set(asdict(GenerationOptions()))
    unknown = set(values)-allowed
    if unknown:
        raise ValueError(f"Unknown generation config fields: {sorted(unknown)}")
    for key in allowed:
        if getattr(args, key, None) is not None:
            values[key] = getattr(args, key)
    def progress(counts):
        print(f"sampled={counts['sampled_latents']} decoded={counts['decoded_attempts']} "
              f"unique_eligible={counts['unique_eligible_molecules']}", file=sys.stderr)
    result = model.generate(GenerationOptions(**values), progress=progress)
    columns = ["rank", "smiles", "selfies", "latent_activity_score", "latent_index", "draw_index",
               "eos_terminated", "aliphatic_carbon_counts"]
    write_results(output, summary_path, result.molecules, columns, result.summary)
    print(json.dumps({"output":str(output), "summary":str(summary_path),
                      "retained_molecules":len(result.molecules), "complete":result.summary["complete"]}))
    if not result.summary["complete"]:
        print("Sampling budget exhausted. Partial results and exact counts were saved.", file=sys.stderr)
        return 3
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, TypeError, RuntimeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
