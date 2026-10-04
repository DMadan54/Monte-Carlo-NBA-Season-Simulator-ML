"""Generate a version-linked awards artifact without changing season outputs."""
import argparse
from pathlib import Path
from src.models.awards import save_awards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = save_awards(args.source, args.output)
    print(f"{result['forecast_version']}: {result['status']}; saved {args.output}")


if __name__ == '__main__':
    main()
