"""Random name generator."""

import secrets
import string
from pathlib import Path


def generate_id(length: int = 8) -> str:
    """Generate a random base-36 string of `length` digits.

    This method is taken from the wandb SDK.

    There are ~2.8T base-36 8-digit strings. Generating 210k ids will have a ~1% chance of collision.

    Args:
        length (int, optional): The length of the string. Defaults to 8.

    Returns:
        str: A random base-36 string of `length` digits.

    """
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def generate_name() -> str:
    """Generate a random name.

    Returns:
        str: The final name.

    """
    from names_generator import generate_name as _generate_name

    return _generate_name(style="hyphen")


def generate_counted_name(artifact_dir: Path) -> str:
    """Generate a random name with a count attached.

    The count is calculated by the number of existing directories in the specified artifact directory.
    The final name is in the format '{somename}-{somesecondname}-{count+1}'.

    Args:
        artifact_dir (Path): The directory of existing runs.

    Returns:
        str: The final name.

    """
    from names_generator import generate_name as _generate_name

    run_name = _generate_name(style="hyphen")
    # Count the number of existing runs in the artifact_dir, increase the number by one and append it to the name
    run_count = sum(1 for p in artifact_dir.glob("*") if p.is_dir())
    run_name = f"{run_name}-{run_count + 1}"
    return run_name
