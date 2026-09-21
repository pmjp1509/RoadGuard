"""Resolve model weights from a local path, falling back to the Hugging Face Hub.

Local development keeps its weights under `runs/` and `models/weights/`, both of which
are too large for git and are not shipped in the container image. In deployment the same
files come from a Hub model repo and land in the Hub cache, so nothing has to be baked
into the image.
"""
import os
from pathlib import Path
from typing import Optional

from src.utils.logger import setup_logger

logger = setup_logger("Weights")

ROOT_DIR = Path(__file__).parent.parent.parent.absolute()

# Set INFRASIGHT_WEIGHTS_REPO to override the repo named in config.yaml, which is what
# the deployment does rather than editing the file.
REPO_ENV_VAR = "INFRASIGHT_WEIGHTS_REPO"


def _is_missing(exc: Exception) -> bool:
    """True when the Hub says the repo or file does not exist, rather than that the
    download itself failed. huggingface_hub raises typed errors for the first kind."""
    try:
        from huggingface_hub.errors import (
            EntryNotFoundError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ImportError:
        return False
    return isinstance(
        exc, (EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError)
    )


def resolve(
    local_paths: list,
    hub_repo: Optional[str] = None,
    hub_file: Optional[str] = None,
) -> str:
    """Return a usable path to a weights file.

    Args:
        local_paths: candidate paths, relative to the project root, tried in order.
        hub_repo: Hugging Face model repo id, used when no local path exists.
        hub_file: filename inside that repo.

    Raises:
        FileNotFoundError: nothing local matched and the Hub was not configured or the
            download failed.
    """
    for candidate in local_paths:
        if not candidate:
            continue
        full = ROOT_DIR / candidate
        if full.exists():
            logger.info(f"Using local weights: {full.name}")
            return str(full)

    repo = os.environ.get(REPO_ENV_VAR) or hub_repo
    # config.yaml ships a CHANGE-ME placeholder. Treating it as a real repo id would turn
    # "you have not configured this yet" into an HTTP 401 from the Hub.
    if repo and repo.startswith("CHANGE-ME"):
        repo = None

    if not repo or not hub_file:
        raise FileNotFoundError(
            f"No weights found at {local_paths} and no Hub repo configured. "
            f"Set {REPO_ENV_VAR} or fill in hub_repo and hub_file in config/config.yaml."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise FileNotFoundError(
            f"No local weights at {local_paths}, and huggingface-hub is not installed "
            "so they cannot be fetched. Install it with: pip install huggingface-hub"
        ) from exc

    logger.info(f"Fetching {hub_file} from {repo}...")
    try:
        path = hf_hub_download(repo_id=repo, filename=hub_file)
    except Exception as exc:
        # A download failure is not the same as "these weights do not exist". Callers
        # degrade gracefully on FileNotFoundError, and a credentials or disk problem
        # quietly changing the reported result is worse than stopping.
        if _is_missing(exc):
            raise FileNotFoundError(
                f"{hub_file} is not in {repo}: {exc}"
            ) from exc
        raise RuntimeError(
            f"Could not download {hub_file} from {repo}: {exc}"
        ) from exc

    logger.info(f"Weights ready: {path}")
    return path


def resolve_from_config(section: dict, path_keys: tuple = ("weights_path",)) -> str:
    """Resolve weights described by one `models.*` block of config.yaml."""
    return resolve(
        [section.get(key) for key in path_keys],
        section.get("hub_repo"),
        section.get("hub_file"),
    )


if __name__ == "__main__":
    # Warm the cache so the first request does not pay for the download.
    import sys
    import yaml

    with open(ROOT_DIR / "config" / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    failures = []
    for name, keys in [
        ("yolo", ("weights_path", "weights_fallback", "weights_legacy")),
        ("material", ("weights_path",)),
    ]:
        try:
            print(f"{name}: {resolve_from_config(config['models'][name], keys)}")
        except FileNotFoundError as exc:
            failures.append(f"{name}: {exc}")

    for line in failures:
        print(line, file=sys.stderr)
    sys.exit(1 if failures else 0)
