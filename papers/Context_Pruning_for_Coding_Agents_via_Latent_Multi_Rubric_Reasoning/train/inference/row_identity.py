import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Set


def normalize_identity_mask(raw_kept_frags: Any) -> List[int]:
    if not isinstance(raw_kept_frags, list):
        return []

    kept = set()
    for line_no in raw_kept_frags:
        try:
            kept.add(int(line_no))
        except Exception:
            continue
    return sorted(kept)


def normalize_identity_score(raw_score: Any):
    try:
        return round(float(raw_score), 12)
    except Exception:
        return None


def build_row_identity_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    if "original_kept_frags" in item:
        kept_frags = normalize_identity_mask(item.get("original_kept_frags"))
    elif "kept_frags" in item:
        kept_frags = normalize_identity_mask(item.get("kept_frags"))
    elif "final_kept_frags" in item:
        kept_frags = normalize_identity_mask(item.get("final_kept_frags"))
    elif "repaired_kept_frags" in item:
        kept_frags = normalize_identity_mask(item.get("repaired_kept_frags"))
    else:
        kept_frags = []

    return {
        "query": item.get("query"),
        "code": item.get("code"),
        "score": normalize_identity_score(item.get("score")),
        "seed_kept_frags": kept_frags,
    }


def build_row_identity(item: Dict[str, Any]) -> str:
    payload = build_row_identity_payload(item)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def load_processed_row_ids(output_jsonl: Path) -> Set[str]:
    processed_row_ids: Set[str] = set()
    if not output_jsonl.exists():
        return processed_row_ids

    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                processed_row_ids.add(build_row_identity(item))

    return processed_row_ids
