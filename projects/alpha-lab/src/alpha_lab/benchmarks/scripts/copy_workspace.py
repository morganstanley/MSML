"""Copy a benchmark workspace from a source directory to a destination.

Default behavior: full copy of every top-level entry in the source workspace.
``data/`` and any directories listed in the source's ``workspace_includes``
can be symlinked instead of copied via ``--symlink-data`` and
``--symlink-includes``.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def copy_workspace(
    source: Path,
    output_dir: Path,
    *,
    name: str | None = None,
    symlink_data: bool = False,
    symlink_includes: bool = False,
    config: dict[str, Any] | None = None,
) -> Path:
    """Copy a workspace from ``source`` into ``output_dir``.

    Args:
        source: Source workspace directory.
        output_dir: Parent directory for the new workspace.
        name: Destination directory name. Defaults to source directory name.
        symlink_data: If True, symlink the source's ``data/`` directory
            (or each entry inside it) into the new workspace instead of
            copying. Default copies.
        symlink_includes: If True, symlink each ``workspace_includes`` entry
            into the new workspace instead of copying. Default copies.
        config: If provided, use this dict as ``config.json`` for the new
            workspace instead of reading from the source's ``config.json``.

    Returns:
        Path to the created workspace directory.

    Raises:
        FileNotFoundError: If source does not exist, ``config.json`` is
            missing (when no external config is provided), or a declared
            ``workspace_includes`` entry is missing from source.
        FileExistsError: If the destination workspace already exists.
    """
    source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source workspace does not exist: {source}")

    dst_name = name or source.name
    dst = output_dir.resolve() / dst_name
    if dst.exists():
        raise FileExistsError(f"Destination workspace already exists: {dst}")

    if config is not None:
        cfg = config
    else:
        config_path = source / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"config.json not found in source workspace: {source}"
            )
        cfg = json.loads(config_path.read_text())

    includes: list[str] = list(cfg.get("workspace_includes", []) or [])
    # ``data/`` and the includes are handled with their own copy/symlink modes;
    # everything else gets a plain recursive copy.
    handled = {"data", "config.json"} | set(includes)

    dst.mkdir(parents=True)

    for entry in source.iterdir():
        if entry.name in handled:
            continue
        target = dst / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target)
        else:
            shutil.copy2(entry, target)

    _copy_data(source, dst, cfg, symlink=symlink_data)

    for entry_name in includes:
        src = source / entry_name
        if not src.exists():
            raise FileNotFoundError(
                f"workspace_includes entry {entry_name!r} missing from "
                f"source workspace: {source}"
            )
        target = dst / entry_name
        if symlink_includes:
            target.symlink_to(src, target_is_directory=src.is_dir())
        elif src.is_dir():
            shutil.copytree(src, target)
        else:
            shutil.copy2(src, target)

    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    LOGGER.info("[copied] %s -> %s", source.name, dst)
    return dst


def _copy_data(
    source: Path, dst: Path, cfg: dict[str, Any], *, symlink: bool
) -> None:
    """Copy or symlink the source's ``data/`` into ``dst/data/``.

    Updates ``cfg["data_path"]`` to point at the new location.
    """
    raw_data = Path(cfg["data_path"])
    abs_data = raw_data if raw_data.is_absolute() else (source / raw_data).resolve()
    if not abs_data.exists():
        raise FileNotFoundError(f"data_path not found: {abs_data}")

    dst_data_dir = dst / "data"
    dst_data_dir.mkdir(parents=True)

    if symlink:
        if abs_data.is_dir():
            for entry in abs_data.iterdir():
                (dst_data_dir / entry.name).symlink_to(
                    entry, target_is_directory=entry.is_dir()
                )
            cfg["data_path"] = str(dst_data_dir.resolve())
        else:
            link = dst_data_dir / abs_data.name
            link.symlink_to(abs_data)
            cfg["data_path"] = str(link)
    else:
        if abs_data.is_dir():
            for entry in abs_data.iterdir():
                target = dst_data_dir / entry.name
                if entry.is_dir():
                    shutil.copytree(entry, target)
                else:
                    shutil.copy2(entry, target)
            cfg["data_path"] = str(dst_data_dir.resolve())
        else:
            target = dst_data_dir / abs_data.name
            shutil.copy2(abs_data, target)
            cfg["data_path"] = str(target.resolve())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Copy a benchmark workspace from a source directory to a destination."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Source workspace directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Parent directory for the new workspace.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Override destination directory name (defaults to source dir name).",
    )
    parser.add_argument(
        "--symlink-data",
        action="store_true",
        help="Symlink the source's data/ into the new workspace instead of copying.",
    )
    parser.add_argument(
        "--symlink-includes",
        action="store_true",
        help=(
            "Symlink each workspace_includes entry instead of copying. "
            "Default is to copy."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    try:
        copy_workspace(
            source=args.source.resolve(),
            output_dir=args.output_dir,
            name=args.name,
            symlink_data=args.symlink_data,
            symlink_includes=args.symlink_includes,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2
    return 0


def cli(argv: list[str] | None = None) -> int:
    """Simplified CLI: ``--src WS --dest DIR [--name NAME] [--symlink-data] [--symlink-includes]``."""
    parser = argparse.ArgumentParser(
        description="Copy a workspace from a source directory to a destination."
    )
    parser.add_argument("--src", type=Path, required=True,
                        help="Source workspace directory.")
    parser.add_argument("--dest", type=Path, required=True,
                        help="Parent directory for the new workspace.")
    parser.add_argument("--name", default=None,
                        help="Destination dir name (default: source dir name).")
    parser.add_argument("--symlink-data", action="store_true")
    parser.add_argument("--symlink-includes", action="store_true")
    args = parser.parse_args(argv)

    inner = [str(args.src), "--output-dir", str(args.dest)]
    if args.name is not None:
        inner += ["--name", args.name]
    if args.symlink_data:
        inner.append("--symlink-data")
    if args.symlink_includes:
        inner.append("--symlink-includes")
    return main(inner)


if __name__ == "__main__":
    raise SystemExit(main())
