#!/usr/bin/env python3
"""Build a Windows wheel in staging; expose only a complete checked artifact.

The build job has no signing credentials. Protected release workflows may pass
the resulting unsigned artifact through sign_wheel.py before testing/uploading.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from sign_wheel import context_from_env, prepare_output, publish_unsigned, WheelError


def build_wheel(output_dir: Path) -> None:
    prepare_output(output_dir)
    with tempfile.TemporaryDirectory(
        prefix="wheel-build-", dir=output_dir.parent
    ) as temp:
        staging = Path(temp)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(staging),
            ],
            check=True,
        )
        wheels = list(staging.glob("*.whl"))
        if len(wheels) != 1:
            raise WheelError("Expected exactly one completed Windows wheel")
        publish_unsigned(wheels[0], output_dir, context_from_env())
    print(
        "Unsigned wheel staged and RECORD verified; signing is not configured in build jobs"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    build_wheel(args.output_dir)


if __name__ == "__main__":
    main()
