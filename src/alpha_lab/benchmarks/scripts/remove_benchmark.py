"""Remove benchmarks from a suite by deleting DB rows and workspace directories."""

import argparse
import logging
import shutil
from pathlib import Path

from alpha_lab.benchmarks.registry.store import connect_registry, ensure_schema

LOGGER = logging.getLogger(__name__)

SUITE_DB_NAME = "suite.db"
WORKSPACES_SUBDIR = "workspaces"


def remove_benchmarks(
    suite_dir: Path,
    benchmark_ids: list[str],
) -> list[str]:
    """Remove benchmarks from a suite.

    Args:
        suite_dir: Suite root directory containing suite.db and workspaces/.
        benchmark_ids: Benchmark IDs to remove.

    Returns:
        List of IDs that were successfully removed.

    Raises:
        FileNotFoundError: If suite_dir or suite.db does not exist.
    """
    suite_dir = suite_dir.resolve()
    suite_db = suite_dir / SUITE_DB_NAME
    if not suite_db.is_file():
        raise FileNotFoundError(f"Suite DB not found: {suite_db}")

    removed: list[str] = []
    conn = connect_registry(suite_db)
    try:
        ensure_schema(conn)
        for bid in benchmark_ids:
            db_deleted = False
            fs_deleted = False

            cursor = conn.execute(
                "DELETE FROM benchmarks WHERE id = ?", (bid,)
            )
            if cursor.rowcount == 0:
                LOGGER.warning("Benchmark ID not found in DB: %s", bid)
            else:
                LOGGER.info("[db] removed %s", bid)
                db_deleted = True

            ws_dir = suite_dir / WORKSPACES_SUBDIR / bid
            if ws_dir.is_dir():
                shutil.rmtree(ws_dir)
                LOGGER.info("[fs] removed %s", ws_dir)
                fs_deleted = True
            else:
                LOGGER.warning("Workspace directory not found: %s", ws_dir)

            if db_deleted or fs_deleted:
                removed.append(bid)

        conn.commit()
    finally:
        conn.close()

    return removed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Remove benchmarks from a suite."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Suite root directory (contains suite.db and workspaces/).",
    )
    parser.add_argument(
        "--id",
        nargs="+",
        required=True,
        dest="benchmark_ids",
        metavar="ID",
        help="Benchmark IDs to remove.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    try:
        removed = remove_benchmarks(args.output_dir, args.benchmark_ids)
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 2
    if not removed:
        LOGGER.error("No benchmarks were removed.")
        return 2
    LOGGER.info("Removed %d benchmark(s): %s", len(removed), ", ".join(removed))
    return 0


def cli(argv: list[str] | None = None) -> int:
    """Simplified CLI: ``--dest DIR --id ID1 [ID2 ...]``."""
    parser = argparse.ArgumentParser(
        description="Remove benchmarks from a suite (DB rows + workspaces/)."
    )
    parser.add_argument("--dest", type=Path, required=True,
                        help="Suite directory (contains suite.db and workspaces/).")
    parser.add_argument("--id", nargs="+", required=True, dest="benchmark_ids",
                        metavar="ID", help="Benchmark IDs to remove.")
    args = parser.parse_args(argv)

    inner = ["--output-dir", str(args.dest), "--id", *args.benchmark_ids]
    return main(inner)


if __name__ == "__main__":
    raise SystemExit(main())
