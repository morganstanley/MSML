#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
from swebench.harness.constants import (  # noqa: E402
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_INSTANCE,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR,
    UTF8,
)
from swebench.harness.docker_build import close_logger, setup_logger  # noqa: E402
from swebench.harness.grading import get_eval_report  # noqa: E402
from swebench.harness.reporting import make_run_report  # noqa: E402
from swebench.harness.run_evaluation import get_dataset_from_preds  # noqa: E402
from swebench.harness.test_spec.test_spec import make_test_spec  # noqa: E402
from swebench.harness.utils import get_predictions_from_file, load_swebench_dataset  # noqa: E402


GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]

_IMAGE_PULL_LOCKS: dict[str, threading.Lock] = {}
_IMAGE_PULL_LOCKS_GUARD = threading.Lock()


def get_pull_lock(key: str) -> threading.Lock:
    with _IMAGE_PULL_LOCKS_GUARD:
        if key not in _IMAGE_PULL_LOCKS:
            _IMAGE_PULL_LOCKS[key] = threading.Lock()
        return _IMAGE_PULL_LOCKS[key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SWE-Bench local evaluation using Apptainer/Singularity instead of Docker."
    )
    parser.add_argument(
        "--dataset_name",
        default="princeton-nlp/SWE-Bench_Verified",
        help="Dataset name or local dataset path.",
    )
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        default=None,
        help="Optional explicit instance IDs to evaluate.",
    )
    parser.add_argument(
        "--predictions_path",
        required=True,
        help="Path to predictions file (JSON or JSONL supported by swebench).",
    )
    parser.add_argument("--max_workers", type=int, default=1, help="Parallel workers.")
    parser.add_argument("--run_id", required=True, help="Run identifier.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Per-instance eval timeout in seconds for /eval.sh.",
    )
    parser.add_argument(
        "--namespace",
        default="swebench",
        help='Remote image namespace. Must not be "none" for this Apptainer path.',
    )
    parser.add_argument(
        "--instance_image_tag",
        default="latest",
        help="Remote instance image tag.",
    )
    parser.add_argument(
        "--env_image_tag",
        default="latest",
        help="Kept for CLI compatibility; not used when consuming remote instance images.",
    )
    parser.add_argument(
        "--report_dir",
        default=".",
        help="Directory for the final summary report JSON.",
    )
    parser.add_argument(
        "--apptainer_executable",
        default=os.getenv("MSWEA_SINGULARITY_EXECUTABLE", "apptainer"),
        help="Apptainer/Singularity executable to use.",
    )
    parser.add_argument(
        "--apptainer_cache_dir",
        default=os.getenv("APPTAINER_CACHEDIR"),
        help="Apptainer cache directory. Example: /project/luofeng/jingjiw/apptainer-cache",
    )
    parser.add_argument(
        "--apptainer_tmp_dir",
        default=os.getenv("APPTAINER_TMPDIR") or os.getenv("TMPDIR"),
        help="Apptainer tmp directory. Example: /project/luofeng/jingjiw/apptainer-tmp",
    )
    parser.add_argument(
        "--overlay_size_mb",
        type=int,
        default=2048,
        help="Size of the writable Apptainer overlay in MiB.",
    )
    return parser.parse_args()


def configure_apptainer_env(args: argparse.Namespace) -> None:
    if args.apptainer_cache_dir:
        Path(args.apptainer_cache_dir).mkdir(parents=True, exist_ok=True)
        os.environ["APPTAINER_CACHEDIR"] = args.apptainer_cache_dir
        os.environ["SINGULARITY_CACHEDIR"] = args.apptainer_cache_dir
    if args.apptainer_tmp_dir:
        Path(args.apptainer_tmp_dir).mkdir(parents=True, exist_ok=True)
        os.environ["APPTAINER_TMPDIR"] = args.apptainer_tmp_dir
        os.environ["SINGULARITY_TMPDIR"] = args.apptainer_tmp_dir
        os.environ["TMPDIR"] = args.apptainer_tmp_dir


def to_apptainer_image_uri(image_ref: str) -> str:
    if image_ref.startswith("docker://"):
        return image_ref
    first = image_ref.split("/", 1)[0]
    has_registry = "." in first or ":" in first or first == "localhost"
    if not has_registry:
        image_ref = f"docker.io/{image_ref}"
    return f"docker://{image_ref}"


class CachedApptainerEnvironment:
    def __init__(
        self,
        *,
        image: str,
        cwd: str,
        executable: str,
        cache_dir: str | None,
        tmp_dir: str | None,
        overlay_size_mb: int,
        logger,
    ):
        self.image = image
        self.cwd = cwd
        self.executable = executable
        self.logger = logger
        self.overlay_size_mb = overlay_size_mb
        self.cache_dir = Path(cache_dir or tempfile.gettempdir())
        self.tmp_dir = Path(tmp_dir or tempfile.gettempdir())
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.image_path = self._ensure_image()
        self.work_root = self.tmp_dir / f"swebench-apptainer-{uuid.uuid4().hex[:8]}"
        self.host_bind_dir = self.work_root / "host-bind"
        self.workdir_dir = self.work_root / "workdir"
        self.home_dir = self.work_root / "home"
        self.overlay_path = self.work_root / "overlay.img"
        self.host_bind_dir.mkdir(parents=True, exist_ok=True)
        self.workdir_dir.mkdir(parents=True, exist_ok=True)
        self.home_dir.mkdir(parents=True, exist_ok=True)
        self._create_overlay()

    def _safe_image_name(self) -> str:
        digest = hashlib.sha1(self.image.encode("utf-8")).hexdigest()[:12]
        stem = self.image.replace("docker://", "").replace("/", "_").replace(":", "__")
        return f"{stem}-{digest}.sif"

    def _ensure_image(self) -> Path:
        image_path = self.cache_dir / self._safe_image_name()
        if image_path.exists():
            return image_path
        lock = get_pull_lock(str(image_path))
        with lock:
            if image_path.exists():
                return image_path
            self.logger.info("Pulling Apptainer image %s to %s", self.image, image_path)
            result = subprocess.run(
                [self.executable, "pull", str(image_path), self.image],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding=UTF8,
                errors="replace",
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"apptainer pull failed for {self.image} -> {image_path}\n{result.stdout}"
                )
        return image_path

    def _create_overlay(self) -> None:
        result = subprocess.run(
            [
                self.executable,
                "overlay",
                "create",
                "--size",
                str(self.overlay_size_mb),
                "--sparse",
                str(self.overlay_path),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding=UTF8,
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"apptainer overlay create failed for {self.overlay_path}\n{result.stdout}"
            )

    def write_host_file(self, name: str, content: str) -> str:
        path = self.host_bind_dir / name
        path.write_text(content, encoding=UTF8)
        return f"/host-run/{name}"

    def execute(self, command: str, *, cwd: str = "", timeout: int | None = None) -> dict[str, object]:
        work_dir = cwd or self.cwd
        cmd = [
            self.executable,
            "exec",
            "--contain",
            "--cleanenv",
            "--overlay",
            str(self.overlay_path),
            "--workdir",
            str(self.workdir_dir),
            "--home",
            f"{self.home_dir}:/root",
            "--bind",
            f"{self.host_bind_dir}:/host-run",
        ]
        if work_dir and work_dir != "/":
            cmd.extend(["--pwd", work_dir])
        cmd.extend([str(self.image_path), "bash", "-lc", command])
        result = subprocess.run(
            cmd,
            text=True,
            timeout=timeout,
            encoding=UTF8,
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": result.stdout, "returncode": result.returncode}

    def cleanup(self) -> None:
        shutil.rmtree(self.work_root, ignore_errors=True)


def exec_env(env: CachedApptainerEnvironment, command: str, *, cwd: str, timeout: int | None) -> tuple[str, int, bool]:
    try:
        result = env.execute(command, cwd=cwd, timeout=timeout)
        return result["output"], result["returncode"], False
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(UTF8, errors="replace")
        return output, 124, True


def run_instance_apptainer(
    instance: dict,
    prediction: dict,
    args: argparse.Namespace,
) -> dict[str, bool]:
    test_spec = make_test_spec(
        instance,
        namespace=args.namespace,
        instance_image_tag=args.instance_image_tag,
        env_image_tag=args.env_image_tag,
    )
    instance_id = test_spec.instance_id
    model_name_or_path = prediction.get(KEY_MODEL, "None").replace("/", "__")
    log_dir = RUN_EVALUATION_LOG_DIR / args.run_id / model_name_or_path / instance_id
    report_path = log_dir / LOG_REPORT
    if report_path.exists():
        report = json.loads(report_path.read_text())
        return {"completed": True, "resolved": report[instance_id]["resolved"]}

    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(instance_id, log_dir / LOG_INSTANCE)
    env: CachedApptainerEnvironment | None = None
    report = {}
    eval_completed = False
    try:
        image_uri = to_apptainer_image_uri(test_spec.instance_image_key)
        logger.info("Using Apptainer image %s", image_uri)
        env = CachedApptainerEnvironment(
            image=image_uri,
            cwd="/testbed",
            executable=args.apptainer_executable,
            cache_dir=args.apptainer_cache_dir,
            tmp_dir=args.apptainer_tmp_dir,
            overlay_size_mb=args.overlay_size_mb,
            logger=logger,
        )

        patch_path = env.write_host_file("patch.diff", prediction[KEY_PREDICTION] or "")
        eval_path = env.write_host_file("eval.sh", test_spec.eval_script)

        applied_patch = False
        last_output = ""
        for git_apply_cmd in GIT_APPLY_CMDS:
            output, returncode, _ = exec_env(
                env,
                f"{git_apply_cmd} {patch_path}",
                cwd="/testbed",
                timeout=120,
            )
            last_output = output
            if returncode == 0:
                logger.info("%s:\n%s", APPLY_PATCH_PASS, output)
                applied_patch = True
                break
            logger.info("Failed to apply patch with %s", git_apply_cmd)

        if not applied_patch:
            logger.info("%s:\n%s", APPLY_PATCH_FAIL, last_output)
            raise RuntimeError(f"{APPLY_PATCH_FAIL}:\n{last_output}")

        git_diff_before, _, _ = exec_env(
            env,
            "git -c core.fileMode=false diff",
            cwd="/testbed",
            timeout=120,
        )
        logger.info("Git diff before:\n%s", git_diff_before.strip())

        chmod_out, chmod_rc, _ = exec_env(
            env,
            f"chmod +x {eval_path}",
            cwd="/testbed",
            timeout=60,
        )
        if chmod_rc != 0:
            raise RuntimeError(f"Failed to chmod eval script:\n{chmod_out}")

        test_output, _, timed_out = exec_env(
            env,
            f"/bin/bash {shlex.quote(eval_path)}",
            cwd="/testbed",
            timeout=args.timeout,
        )
        test_output_path = log_dir / LOG_TEST_OUTPUT
        with open(test_output_path, "w", encoding=UTF8) as fh:
            fh.write(test_output)
            if timed_out:
                fh.write(f"\n\nTimeout error: {args.timeout} seconds exceeded.")
        if timed_out:
            raise RuntimeError(f"Test timed out after {args.timeout} seconds.")

        git_diff_after, _, _ = exec_env(
            env,
            "git -c core.fileMode=false diff",
            cwd="/testbed",
            timeout=120,
        )
        logger.info("Git diff after:\n%s", git_diff_after.strip())
        if git_diff_after != git_diff_before:
            logger.info("Git diff changed after running eval script")

        report = get_eval_report(
            test_spec=test_spec,
            prediction=prediction,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
        logger.info(
            "report: %s\nResult for %s: resolved: %s",
            report,
            instance_id,
            report[instance_id]["resolved"],
        )
        report_path.write_text(json.dumps(report, indent=4), encoding=UTF8)
        eval_completed = True
    except Exception as exc:
        logger.error(
            "Error in evaluating %s: %s\n%s",
            instance_id,
            exc,
            traceback.format_exc(),
        )
    finally:
        if env is not None:
            env.cleanup()
        close_logger(logger)
    return {
        "completed": eval_completed,
        "resolved": report.get(instance_id, {}).get("resolved", False),
    }


def main() -> int:
    args = parse_args()
    if args.namespace in {None, "", "none"}:
        raise ValueError(
            "This Apptainer evaluator only supports remote prebuilt instance images. "
            'Use a namespace like "swebench".'
        )

    configure_apptainer_env(args)
    predictions = get_predictions_from_file(args.predictions_path, args.dataset_name, args.split)
    predictions = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}
    dataset = get_dataset_from_preds(
        args.dataset_name,
        args.split,
        args.instance_ids,
        predictions,
        args.run_id,
        rewrite_reports=False,
    )
    full_dataset = load_swebench_dataset(args.dataset_name, args.split, args.instance_ids)
    if not dataset:
        print("No instances to run.")
    else:
        print(
            f"Running {len(dataset)} instances with {args.apptainer_executable} "
            f"(workers={args.max_workers})"
        )
        stats = {"resolved": 0, "unresolved": 0, "error": 0}
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(run_instance_apptainer, instance, predictions[instance[KEY_INSTANCE_ID]], args): instance[
                    KEY_INSTANCE_ID
                ]
                for instance in dataset
            }
            for future in as_completed(futures):
                instance_id = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    with lock:
                        stats["error"] += 1
                    print(f"[error] {instance_id}: {exc}")
                    continue

                with lock:
                    if result["completed"]:
                        if result["resolved"]:
                            stats["resolved"] += 1
                            label = "resolved"
                        else:
                            stats["unresolved"] += 1
                            label = "unresolved"
                    else:
                        stats["error"] += 1
                        label = "error"
                print(
                    f"[{label}] {instance_id} | "
                    f"resolved={stats['resolved']} unresolved={stats['unresolved']} error={stats['error']}"
                )

    report_path = make_run_report(
        predictions,
        full_dataset,
        args.run_id,
        client=None,
        namespace=args.namespace,
        instance_image_tag=args.instance_image_tag,
        env_image_tag=args.env_image_tag,
    )
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    final_report = report_dir / report_path.name
    if report_path.resolve() != final_report.resolve():
        shutil.move(str(report_path), final_report)
    print(f"Final report written to {final_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
