"""Reset optimizer state (and optionally preprocessors) in a SKRL checkpoint.

Usage:
    python scripts/reset_checkpoint.py <checkpoint_path> [--output <output_path>] [--reset-preprocessors]

This keeps all learned model weights but removes optimizer state (including LR,
momentum, etc.). When SKRL loads this checkpoint, it will initialize a fresh
optimizer with the LR from the YAML config (e.g. 3e-4).

With --reset-preprocessors, also removes RunningStandardScaler state so the
preprocessors recalibrate to new observation/reward distributions (useful when
changing environment configuration like number of obstacles).
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="Reset optimizer state in SKRL checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint file")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: overwrites input)")
    parser.add_argument("--reset-preprocessors", action="store_true",
                        help="Also reset RunningStandardScaler preprocessors")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # SKRL MAPPO structure: {agent_name: {policy, value, optimizer, state_preprocessor, ...}}
    removed = []
    for agent_name in list(ckpt.keys()):
        agent_data = ckpt[agent_name]
        if not isinstance(agent_data, dict):
            continue

        # Remove optimizer
        if "optimizer" in agent_data:
            del agent_data["optimizer"]
            removed.append(f"{agent_name}.optimizer")
            print(f"  Removed: {agent_name}.optimizer")

        # Optionally remove preprocessors
        if args.reset_preprocessors:
            for key in list(agent_data.keys()):
                if "preprocessor" in key:
                    del agent_data[key]
                    removed.append(f"{agent_name}.{key}")
                    print(f"  Removed: {agent_name}.{key}")

        # Show what's kept
        for k in agent_data:
            print(f"  Kept:    {agent_name}.{k}")

    if not removed:
        print("WARNING: Nothing removed!")
    else:
        print(f"\nRemoved {len(removed)} entries: {removed}")

    output_path = args.output or args.checkpoint
    print(f"Saving to: {output_path}")
    torch.save(ckpt, output_path)
    print("Done! Training will start with fresh LR from YAML config.")


if __name__ == "__main__":
    main()
