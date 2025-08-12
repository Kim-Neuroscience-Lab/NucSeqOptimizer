#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import yaml

DNA_COMPLEMENT = str.maketrans({"A": "T", "T": "A", "C": "G", "G": "C"})


def reverse_complement(seq: str) -> str:
    return seq.upper().translate(DNA_COMPLEMENT)[::-1]


def parse_yaml(yaml_path: Path) -> List[Dict]:
    data = yaml.safe_load(yaml_path.read_text())
    molecules = data.get("molecules", [])
    if not molecules:
        raise ValueError("No molecules provided in YAML")
    return molecules


def to_fasta_header(mol_type: str, chain_id: str, length: int | None = None) -> str:
    # Chai-1 expects headers formatted like: ">protein|name=A"
    # Only a single label part is supported after the entity type
    # (see chai_lab.data.dataset.inference_dataset.read_inputs)
    return f">{mol_type}|name={chain_id}"


def generate_fasta(molecules: List[Dict], out_fasta: Path) -> None:
    lines: List[str] = []
    for mol in molecules:
        mol_type = mol["type"].lower()
        copies = int(mol.get("copies", 1))
        chain_ids = list(mol.get("chain_ids", []))
        sequence = mol["sequence"].strip().upper()
        include_rc = (
            bool(mol.get("include_reverse_complement", False))
            if mol_type == "dna"
            else False
        )

        for copy_idx in range(copies):
            if chain_ids:
                chain_id = chain_ids.pop(0)
            else:
                chain_id = chr(ord("A") + copy_idx)

            header = to_fasta_header(mol_type, str(chain_id), len(sequence))
            lines.append(header)
            lines.append(sequence)

            if mol_type == "dna" and include_rc:
                rc_header = to_fasta_header(mol_type, f"{chain_id}_RC", len(sequence))
                lines.append(rc_header)
                lines.append(reverse_complement(sequence))

    out_fasta.write_text("\n".join(lines) + "\n")


def chai_cli_available() -> bool:
    return shutil.which("chai-lab") is not None


def run_or_print_chai(
    fasta: Path,
    out_dir: Path,
    use_msa_server: bool = False,
    msa_server_url: str | None = None,
    device: str | None = None,
    disable_esm: bool = False,
    num_trunk_recycles: int | None = None,
    num_diffn_timesteps: int | None = None,
    num_diffn_samples: int | None = None,
) -> int:
    cmd = ["chai-lab", "fold"]
    if use_msa_server:
        cmd.extend(["--use-msa-server", "--use-templates-server"])
        if msa_server_url:
            cmd.extend(["--msa-server-url", msa_server_url])
    if device:
        cmd.extend(["--device", device])
    if disable_esm:
        cmd.append("--no-use-esm-embeddings")
    if num_trunk_recycles is not None:
        cmd.extend(["--num-trunk-recycles", str(num_trunk_recycles)])
    if num_diffn_timesteps is not None:
        cmd.extend(["--num-diffn-timesteps", str(num_diffn_timesteps)])
    if num_diffn_samples is not None:
        cmd.extend(["--num-diffn-samples", str(num_diffn_samples)])
    cmd.extend([str(fasta), str(out_dir)])

    if not chai_cli_available():
        print(
            "chai-lab CLI not available on this system. Install chai_lab in your environment:"
        )
        print(" ")
        print(" ".join(cmd))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    proc = subprocess.run(cmd, env=env, check=False)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate FASTA from YAML and run or print chai-lab command"
    )
    parser.add_argument("yaml", type=Path, help="YAML file describing molecules")
    parser.add_argument(
        "out_dir", type=Path, help="Output directory for fasta and results"
    )
    parser.add_argument(
        "--fasta-name", default="input.fasta", help="Output FASTA filename"
    )
    parser.add_argument(
        "--use-msa-server",
        action="store_true",
        help="Use ColabFold MMseqs2 server for MSAs and templates",
    )
    parser.add_argument("--msa-server-url", default=None, help="Custom MSA server URL")
    parser.add_argument(
        "--device",
        default=None,
        help="Device for chai-lab (e.g., cuda:0, mps, cpu). Default: mps on macOS if available, else cpu if no CUDA.",
    )
    parser.add_argument(
        "--no-esm",
        action="store_true",
        help="Disable ESM embeddings for faster and lighter runs.",
    )
    parser.add_argument(
        "--trunk-recycles",
        type=int,
        default=None,
        help="Override number of trunk recycles (default from chai-lab is 3).",
    )
    parser.add_argument(
        "--diffn-timesteps",
        type=int,
        default=None,
        help="Override number of diffusion timesteps (default from chai-lab is 200).",
    )
    parser.add_argument(
        "--diffn-samples",
        type=int,
        default=None,
        help="Override number of diffusion samples (default from chai-lab is 5).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = args.out_dir / args.fasta_name

    molecules = parse_yaml(args.yaml)
    generate_fasta(molecules, fasta_path)

    # Pick sensible default device if not explicitly provided
    default_device: str | None = None
    try:
        import torch  # type: ignore

        if args.device:
            default_device = args.device
        else:
            if platform.system() == "Darwin" and torch.backends.mps.is_available():
                default_device = "mps"
            elif torch.cuda.is_available():
                default_device = "cuda:0"
            else:
                default_device = "cpu"
    except Exception:
        # torch not available; fall back to CPU
        default_device = args.device or "cpu"

    # Use a fresh run directory per invocation to satisfy chai-lab's requirement
    # that the output directory does not exist or is empty.
    from datetime import datetime

    run_dir = args.out_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    code = run_or_print_chai(
        fasta_path,
        run_dir,
        args.use_msa_server,
        args.msa_server_url,
        device=default_device,
        disable_esm=args.no_esm,
        num_trunk_recycles=args.trunk_recycles,
        num_diffn_timesteps=args.diffn_timesteps,
        num_diffn_samples=args.diffn_samples,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
