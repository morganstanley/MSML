"""Package the review machinery as a standalone, harness-neutral project.

The review machinery (`src/alpha_lab/benchmarks/runcmp/`) audits FINISHED runs
from their preserved artifacts. Nothing in it depends on the harness that
produced the runs — by design it must judge any harness, including external
evidence-contract submissions — and its only import from the surrounding
codebase is the layer that talks to models (`alpha_lab.client.get_provider`
plus the provider classes behind it).

This script emits a self-contained tree:

    README.md, pyproject.toml
    runcmp/            the machinery, imports rewritten to the new package name
    runcmp/llm/        the model-talking layer, DUPLICATED from the harness on
                       purpose: the package must attach to no harness, so the
                       one shared piece is vendored rather than imported
    tests/             the machinery's tests (harness-only tests dropped)

The canonical source of truth stays in this directory; the standalone tree is
generated output, reproducible with:

    python scripts/build_runcmp_standalone.py --out <target>

so the packaged line on GitHub can always be rebuilt from what is here.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src/alpha_lab"

# Rewrites applied to every copied text file (.py/.md/.json/.txt/.sh), in
# order (longest first). Not just imports: the manual, DESIGN.md and
# lineup.json shipped verbatim for a while, telling package users to run
# `PYTHONPATH=src python -m alpha_lab...` — commands that only work in the
# harness tree (found 2026-08-11).
# Site dataset defaults must not ship as personal-directory paths: the
# lineup gets placeholders, and docs/site/msml-internal.md (generated from
# THIS mapping, so the two cannot drift) gives teammates who can read the
# msml shared directories the real locations plus a one-command apply.
SITE_DATA = {
    "<YOUR-SITE-DATA>/etf_rfq_research_dataset":
        "/v/global/user/y/yu/yuriyn/PycharmProjects5/mstech-alphalab"
        "/etf_rfq_research_dataset",
    "<YOUR-SITE-DATA>/payup/dataFullNoDups.csv":
        "/v/region/na/appl/spg/shared/data/spgrisk/agency/x42/specpricer"
        "/dailyDump/v9/prod/dataFullNoDups.csv",
}

REWRITES = [
    # the one shipped-content path in lineup.json (the d6 task dir travels
    # WITH the package); dataset paths stay deployment-local by design
    ("/v/campus/ny/appl/msml/workspace/data/yuriyn/mstech-alphalab-cond"
     "/src/alpha_lab/benchmarks/runcmp", "runcmp"),
] + [(real, ph) for ph, real in SITE_DATA.items()] + [
    ("alpha_lab.benchmarks.runcmp", "runcmp"),
    ("alpha_lab.client", "runcmp.llm.client"),
    ("alpha_lab.provider_openai", "runcmp.llm.provider_openai"),
    ("alpha_lab.provider_anthropic", "runcmp.llm.provider_anthropic"),
    ("alpha_lab.provider_bedrock", "runcmp.llm.provider_bedrock"),
    ("alpha_lab.provider_chat", "runcmp.llm.provider_chat"),
    ("alpha_lab.provider_grok", "runcmp.llm.provider_grok"),
    ("alpha_lab.provider", "runcmp.llm.provider"),
    # prose occurrences (usage strings, docstrings)
    ("python -m alpha_lab.benchmarks.runcmp", "python -m runcmp"),
    # slash-path references (one test reads gate_policy.json relative to the
    # project root; imports don't catch it)
    ("src/alpha_lab/benchmarks/runcmp/", "runcmp/"),
    # the package root IS the import root; the harness needed src/
    ('export PYTHONPATH="$REPO/src"', 'export PYTHONPATH="$REPO"'),
    ("PYTHONPATH=src", "PYTHONPATH=."),
    # neutral env-var name for the wrapper scripts' python override
    ("ALPHALAB_PYTHON", "RUNCMP_PYTHON"),
]

# text suffixes that go through the rewrites; everything else copies verbatim
TEXT_SUFFIXES = {".py", ".md", ".json", ".txt", ".sh"}

# wrapper scripts the docs point at: the full deterministic chain with
# post-condition checks, the one-command PR gate, and the on-prem token
# bootstrap the vendored auth layer's error messages name. All of them are
# pure stage-drivers / stdlib+site-lib code — nothing imports the harness.
SCRIPTS = ["runcmp_chain.sh", "runcmp_gate.sh", "auth_setup.sh",
           "prefetch_token.py"]

LLM_FILES = ["client.py", "provider.py", "provider_openai.py",
             "provider_anthropic.py", "provider_bedrock.py",
             "provider_chat.py", "provider_grok.py",
             "scalar_2_sample_setup.py"]

# tests of harness code that rides along in the shared test file; the
# standalone package has no harness, so these classes are cut whole
DROP_TEST_CLASSES = ["TestLeaderboardDirection"]

README = (pathlib.Path(__file__).parent
          / "runcmp_standalone_README.md").read_text()

PYPROJECT = """\
[project]
name = "runcmp"
version = "0.1.0"
description = "Harness-neutral post-hoc comparison and LLM review of autonomous research runs"
requires-python = ">=3.11"
dependencies = ["openai>=1.0"]

[project.optional-dependencies]
anthropic = ["anthropic>=0.34"]
bedrock = ["boto3>=1.34"]
html = ["markdown>=3.5"]
mlflow = ["mlflow>=2.12"]
test = ["pytest>=8"]

[tool.setuptools.packages.find]
include = ["runcmp*"]
"""


def rewrite(text: str) -> str:
    for old, new in REWRITES:
        text = text.replace(old, new)
    return text


def copy_py(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.write_text(rewrite(src.read_text()))


def drop_classes(text: str, names: list[str]) -> str:
    for name in names:
        m = re.search(rf"^class {name}\b.*?(?=^class |\Z)", text,
                      re.MULTILINE | re.DOTALL)
        if m:
            text = text[:m.start()] + text[m.end():]
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()
    out: pathlib.Path = args.out
    pkg = out / "runcmp"
    if out.exists():
        raise SystemExit(f"{out} already exists — give a fresh target")
    (pkg / "llm").mkdir(parents=True)

    # the machinery itself — every text file goes through the rewrites (the
    # manual and lineup.json once shipped verbatim with harness-tree commands)
    src_pkg = SRC / "benchmarks/runcmp"
    for p in sorted(src_pkg.rglob("*")):
        if "__pycache__" in p.parts or not p.is_file():
            continue
        dst = pkg / p.relative_to(src_pkg)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix in TEXT_SUFFIXES:
            dst.write_text(rewrite(p.read_text()))
        else:
            shutil.copy2(p, dst)

    # the wrapper scripts the docs name (a shipped manual once pointed at
    # scripts/ that existed only in the harness tree)
    scripts_dir = out / "scripts"
    scripts_dir.mkdir()
    for name in SCRIPTS:
        dst = scripts_dir / name
        dst.write_text(rewrite((ROOT / "scripts" / name).read_text()))
        dst.chmod(0o755)

    # the vendored model layer
    (pkg / "llm/__init__.py").write_text(
        '"""The model-talking layer, vendored so the package attaches to no '
        'harness."""\nfrom runcmp.llm.client import get_provider  # noqa: F401\n')
    for name in LLM_FILES:
        copy_py(SRC / name, pkg / "llm" / name)

    # complete, real mission files as samples — nothing left to imagination.
    # These are generated artifacts (comparison_out is not tracked), so the
    # generator names exactly what it needs and fails loudly if one is absent.
    SAMPLES = {
        "comparison_out/grand/team_opus/mission.md": "union.md",
        "comparison_out/review_reasonfix/team_opus/mission.md": "harness.md",
        "comparison_out/review_reasonfix/treatment_opus/mission.md": "treatment.md",
        "comparison_out/twoday/review_variability/mission.md": "variability.md",
        "comparison_out/grand/mission_notes.txt": "notes.txt",
    }
    samples = pkg / "missions/samples"
    samples.mkdir(parents=True, exist_ok=True)
    for src_rel, dst_name in SAMPLES.items():
        src = ROOT / src_rel
        if not src.is_file():
            raise SystemExit(f"sample mission source missing: {src} — supply a "
                             "real generated mission or drop it from SAMPLES")
        (samples / dst_name).write_text(src.read_text())
    (samples / "README.md").write_text(
        "# Complete, real mission files\n\n"
        "Every focus has a full sample here, taken verbatim from production "
        "batteries — not sketches:\n\n"
        "- `union.md` — the all-encompassing report over a whole campaign\n"
        "- `harness.md` — which system to run, per task and overall\n"
        "- `treatment.md` — before/after one deliberate change\n"
        "- `variability.md` — run-to-run spread over replication groups\n"
        "- `notes.txt` — operator-declared facts passed with --note @notes.txt\n\n"
        "Regenerate for YOUR corpus with `python -m runcmp mission --corpus ... "
        "--focus <name>`; never splice an old mission's corpus description "
        "forward.\n")

    # four real end-product reports, shipped INSIDE the package so the README
    # can link them RELATIVELY — GitHub strips file:// links at render time,
    # so an in-tree link is the only kind guaranteed clickable (2026-08-10).
    # .md renders readably on GitHub; .html is for a local browser.
    EXAMPLES = {
        "comparison_out/grand/team_opus/REPORT": "union_campaign_report",
        "comparison_out/review_reasonfix/team_opus/REPORT.superseded_20260810T192259":
            "harness_battery_report",
        "comparison_out/review_d4rep/team_opus/REPORT": "replication_battery_report",
        "comparison_out/review_d7/team_opus/REPORT": "single_domain_battery_report",
    }
    ex = out / "examples"
    ex.mkdir()
    for src_stem, dst_stem in EXAMPLES.items():
        for suffix in (".md", ".html"):
            src = ROOT / (src_stem + suffix)
            if not src.is_file():
                raise SystemExit(f"example report missing: {src}")
            (ex / (dst_stem + suffix)).write_text(src.read_text())

    # the MLflow navigation guide's screenshots, and the demo submission the
    # README walks through — both live in the maintaining directory and ship
    # with the package so nothing in the docs points at thin air
    shots_src = ROOT / "scripts/mlflow_guide_shots"
    docs = out / "docs/mlflow"
    docs.mkdir(parents=True)
    for p in sorted(shots_src.glob("*.png")):
        (docs / p.name).write_bytes(p.read_bytes())
    if not any(docs.iterdir()):
        raise SystemExit("no MLflow guide screenshots in scripts/mlflow_guide_shots")
    demo_src = ROOT / "scripts/demo_submission"
    if not demo_src.is_dir():
        raise SystemExit("scripts/demo_submission missing")
    shutil.copytree(demo_src, ex / "demo_submission")

    # the internal site page: real dataset locations for people who can
    # read the msml shared directories, generated from SITE_DATA
    site = out / "docs/site"
    site.mkdir(parents=True)
    seds = "; ".join(f"s|{ph}|{real}|g" for ph, real in SITE_DATA.items())
    (site / "msml-internal.md").write_text(
        "# msml-team data locations (internal)\n\n"
        "The shipped `runcmp/lineup.json` uses `<YOUR-SITE-DATA>` "
        "placeholders so the\npackage carries no personal or site "
        "directories. If you can read the msml\nshared directories "
        "(most of the maintainer's team can), the real locations:\n\n"
        + "".join(f"- `{ph}`\n  is really\n  `{real}`\n"
                  for ph, real in SITE_DATA.items())
        + "\nPoint the packaged lineup at them in one command:\n\n"
        "```bash\n"
        f"sed -i '{seds}' runcmp/lineup.json\n"
        "```\n\n"
        "The finished showcase reports (with charts) are on the same "
        "share —\nthe top README's \"end product\" section carries the "
        "Explorer paths.\n")

    # tests, minus harness-only classes
    (out / "tests").mkdir()
    t = rewrite((ROOT / "tests/test_runcmp.py").read_text())
    (out / "tests/test_runcmp.py").write_text(drop_classes(t, DROP_TEST_CLASSES))

    (out / "README.md").write_text(README)
    (out / "pyproject.toml").write_text(PYPROJECT)
    # the vendored auth module drops credentials beside itself when imported;
    # those must never enter the packaged tree
    (out / ".gitignore").write_text(
        "__pycache__/\n*.pyc\nsecrets.env\n.token_cache.json\n")

    # hard guarantee of standalone-ness: no import of the harness anywhere,
    # and no instruction anywhere that only works in the harness tree
    leaks = []
    for p in out.rglob("*.py"):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if re.match(r"\s*(from|import)\s+alpha_lab", line):
                leaks.append(f"{p}:{i}: {line.strip()}")
    for p in out.rglob("*"):
        if not p.is_file() or p.suffix not in TEXT_SUFFIXES:
            continue
        text = p.read_text(errors="replace")
        for bad in ("alpha_lab.benchmarks", "python -m alpha_lab",
                    "PYTHONPATH=src", "mstech-alphalab-cond/src"):
            if bad in text:
                leaks.append(f"{p}: contains {bad!r}")
        # personal-directory paths never ship, except the top README's
        # deliberate internal-share block and quoted evidence in examples
        rel = p.relative_to(out)
        if rel.parts[0] != "examples" and str(rel) != "README.md" \
                and "site" not in rel.parts:
            for bad in ("/v/global/user", "PycharmProjects"):
                if bad in text:
                    leaks.append(f"{p}: contains personal path {bad!r}")
    if leaks:
        raise SystemExit("harness references leaked into the standalone tree:\n"
                         + "\n".join(leaks))
    n = sum(1 for _ in out.rglob("*") if _.is_file())
    print(f"standalone tree written: {out} ({n} files, no alpha_lab imports)")


if __name__ == "__main__":
    main()
