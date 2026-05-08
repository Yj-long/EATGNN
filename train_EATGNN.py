import argparse
from pathlib import Path

from eatgnn.config_utils import load_config
from eatgnn.trainer import run_training


def parse_args():
    parser = argparse.ArgumentParser(description="Train EATGNN from a JSON config file.")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to the JSON configuration file. Defaults to config.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    config = load_config(config_path)
    run_training(config)


if __name__ == "__main__":
    main()
