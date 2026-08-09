"""Backward-compatible wrapper for the new explanation workflow."""

from explain import explain, parse_args

if __name__ == "__main__":
    explain(parse_args())
