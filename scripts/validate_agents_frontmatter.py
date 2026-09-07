#!/usr/bin/env python3
"""Validate `.md` files against the Anthropic SKILLS.md open standard.

Recursively scans a directory and checks every `.md` file's YAML frontmatter
for compliance with the reserved-key contract: only `name`, `description`,
`allowed-tools`, `license`, and `metadata` are permitted at the top level;
everything else belongs under `metadata`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import yaml

RESERVED_TOP_LEVEL_KEYS = frozenset({
    "name",
    "description",
    "allowed-tools",
    "license",
    "metadata",
})

REQUIRED_KEYS = ("name", "description")

FRONTMATTER_OPEN = "---\n"
FRONTMATTER_CLOSE = "\n---\n"


def _split_frontmatter_from_config_body(text: str) -> tuple[str, str]:
    if not text.startswith(FRONTMATTER_OPEN):
        raise ValueError("missing YAML frontmatter opening")
    if (end := text.find(FRONTMATTER_CLOSE, len(FRONTMATTER_OPEN))) == -1:
        raise ValueError("missing YAML frontmatter closing")
    return text[len(FRONTMATTER_OPEN):end], text[end + len(FRONTMATTER_CLOSE):]


def _validate_frontmatter_mapping(frontmatter: dict) -> list[str]:
    violations: list[str] = []

    for required_key in REQUIRED_KEYS:
        value = frontmatter.get(required_key)
        if value is None:
            violations.append(f"missing required '{required_key}'")
        elif not isinstance(value, str) or not value.strip():
            violations.append(f"'{required_key}' must be a non-empty string")

    for top_key in frontmatter.keys():
        if top_key not in RESERVED_TOP_LEVEL_KEYS:
            violations.append(
                f"unexpected top-level key '{top_key}' (should be under metadata)"
            )

    if "allowed-tools" in frontmatter:
        allowed_tools = frontmatter["allowed-tools"]
        if not isinstance(allowed_tools, list) or not all(
            isinstance(tool, str) for tool in allowed_tools
        ):
            violations.append("'allowed-tools' must be a list of strings")

    if "license" in frontmatter and not isinstance(frontmatter["license"], str):
        violations.append("'license' must be a string")

    if "metadata" in frontmatter and not isinstance(frontmatter["metadata"], dict):
        violations.append("'metadata' must be a mapping")

    return violations


def _classify_markdown_file(path: Path) -> tuple[str, list[str]]:
    """Return (status, violations) for a single markdown file.

    status is one of: "valid", "invalid", "no-frontmatter".
    """
    text = path.read_text()

    if not text.startswith(FRONTMATTER_OPEN):
        return "no-frontmatter", []

    try:
        frontmatter_text, _body = _split_frontmatter_from_config_body(text)
    except ValueError as split_error:
        return "invalid", [str(split_error)]

    try:
        frontmatter = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as parse_error:
        return "invalid", [f"invalid YAML: {parse_error}"]

    if not isinstance(frontmatter, dict):
        return "invalid", ["frontmatter must be a YAML mapping"]

    violations = _validate_frontmatter_mapping(frontmatter)
    return ("invalid" if violations else "valid"), violations


@click.command()
@click.argument(
    "directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Treat files without any frontmatter as invalid.",
)
def main(directory: Path, strict: bool) -> None:
    """Validate `.md` files under DIRECTORY against the SKILLS.md open standard."""
    root = directory.resolve()
    markdown_files = sorted(root.rglob("*.md"))

    valid_count = 0
    invalid_count = 0
    no_frontmatter_count = 0
    invalid_lines: list[str] = []

    for markdown_path in markdown_files:
        status, violations = _classify_markdown_file(markdown_path)
        relative_path = markdown_path.relative_to(root)

        if status == "valid":
            valid_count += 1
        elif status == "no-frontmatter":
            if strict:
                invalid_count += 1
                invalid_lines.append(f"{relative_path}: no frontmatter")
            no_frontmatter_count += 1
        else:
            invalid_count += 1
            invalid_lines.append(f"{relative_path}: {'; '.join(violations)}")

    for line in invalid_lines:
        click.echo(line)

    if invalid_lines:
        click.echo("")
    click.echo(f"scanned:         {len(markdown_files)}")
    click.echo(f"valid:           {valid_count}")
    click.echo(f"invalid:         {invalid_count}")
    click.echo(f"no-frontmatter:  {no_frontmatter_count}")

    sys.exit(1 if invalid_count > 0 else 0)


if __name__ == "__main__":
    main()
