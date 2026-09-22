"""
Deterministic cached demo-feed builder.

All displayed classifications come from hierarchy-corrected production positive
decisions (labels after ``run_hierarchy_correction``). No ground-truth category
columns, simulated probabilities, or independent 0.5 cutoff are used.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from app.services.errors import ModelServiceError
from app.utils.feed_display import (
    _safe_float_prob,
    _safe_text_value,
    calculate_severity,
    genre_to_source,
    to_display_name,
)
from app.utils.hierarchy_helpers import run_hierarchy_correction

SCHEMA_VERSION = 1
MAX_CLASSIFICATIONS = 10
MAX_CATEGORY_NAMES = 3


def load_message_ids(path: Path | str) -> list[int]:
    """Load pinned message IDs from a JSON array. Fail closed on missing/invalid."""
    ids_path = Path(path)
    if not ids_path.is_file():
        raise FileNotFoundError(f"Demo feed message IDs file not found: {ids_path}")

    try:
        with open(ids_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in message IDs file {ids_path}: {exc}") from exc

    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Message IDs file must be a non-empty JSON array: {ids_path}")

    message_ids: list[int] = []
    for index, raw in enumerate(payload):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(
                f"Message IDs[{index}] must be an integer, got {raw!r} ({type(raw).__name__})"
            )
        if isinstance(raw, float) and (math.isnan(raw) or math.isinf(raw) or raw != int(raw)):
            raise ValueError(f"Message IDs[{index}] must be an integer, got {raw!r}")
        message_ids.append(int(raw))
    return message_ids


def select_initial_message_ids(df: pd.DataFrame, n: int = 50) -> list[int]:
    """
    Select the first ``n`` message IDs by ascending id with non-empty message text.

    Intended only for ``--init-ids``. Ignores all category/label columns.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if "id" not in df.columns:
        raise ValueError("DataFrame must include an 'id' column to select message IDs")
    if "message" not in df.columns:
        raise ValueError("DataFrame must include a 'message' column to select message IDs")

    working = df[["id", "message"]].copy()
    working["id"] = pd.to_numeric(working["id"], errors="coerce")
    working = working.dropna(subset=["id"])
    working["id"] = working["id"].astype(int)
    working["message"] = working["message"].map(_safe_text_value).str.strip()
    working = working[working["message"] != ""]
    working = working.sort_values("id", ascending=True, kind="mergesort")
    selected = working["id"].drop_duplicates().head(n).tolist()
    if len(selected) < n:
        raise ValueError(
            f"Only found {len(selected)} non-empty messages with ids; need {n}"
        )
    return selected


def _canonicalize_text(value: Any) -> str:
    """Canonicalize null/NaN to empty string for stable hashing and display."""
    return _safe_text_value(value)


def hash_input_rows(rows_in_id_order: Sequence[Mapping[str, Any]]) -> str:
    """
    SHA-256 over ordered id/message/original/genre fields.

    Null/NaN text values canonicalize to empty string. Serialization is stable
    (sorted keys, compact separators, UTF-8).
    """
    digest = hashlib.sha256()
    for row in rows_in_id_order:
        payload = {
            "id": int(row["id"]),
            "message": _canonicalize_text(row.get("message")),
            "original": _canonicalize_text(row.get("original")),
            "genre": _canonicalize_text(row.get("genre")),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def _round_prob_map(probabilities: Mapping[str, Any], digits: int = 6) -> dict[str, float]:
    return {
        str(key): round(_safe_float_prob(value), digits)
        for key, value in probabilities.items()
    }


def _label_map(labels: Mapping[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, value in labels.items():
        if value is None:
            result[str(key)] = 0
            continue
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            result[str(key)] = 0
            continue
        try:
            result[str(key)] = int(value)
        except (TypeError, ValueError, OverflowError):
            result[str(key)] = 0
    return result


def _positive_pairs(fixed_labels: Mapping[str, int], fixed_probs: Mapping[str, float]):
    """Hierarchy positives excluding related, sorted by confidence descending."""
    pairs = [
        (internal, _safe_float_prob(fixed_probs.get(internal, 0.0)))
        for internal, label in fixed_labels.items()
        if internal != "related" and int(label) == 1
    ]
    pairs.sort(key=lambda item: (-item[1], item[0]))
    return pairs


def _build_feed_item(
    *,
    message_id: int,
    row: Mapping[str, Any],
    raw_probs: Mapping[str, Any],
    raw_labels: Mapping[str, Any],
    fixed_probs: Mapping[str, float],
    fixed_labels: Mapping[str, int],
) -> dict[str, Any]:
    message = _canonicalize_text(row.get("message")).strip()
    original_raw = _canonicalize_text(row.get("original")).strip()
    original = original_raw or None
    is_translated = bool(original and original != message)
    positive = _positive_pairs(fixed_labels, fixed_probs)
    classifications = [
        {
            "category": to_display_name(internal),
            "confidence": round(confidence, 2),
        }
        for internal, confidence in positive[:MAX_CLASSIFICATIONS]
    ]
    categories = [to_display_name(internal) for internal, _ in positive[:MAX_CATEGORY_NAMES]]
    return {
        "id": f"SIG-{message_id}",
        "source": genre_to_source(_canonicalize_text(row.get("genre"))),
        "content": message,
        "originalContent": original if is_translated else None,
        "language": "en",
        "riskLevel": calculate_severity(fixed_probs),
        "categories": categories,
        "classifications": classifications,
        "isTranslated": is_translated,
        "raw": {
            "probabilities": _round_prob_map(raw_probs),
            "labels": _label_map(raw_labels),
        },
        "fixed": {
            "probabilities": _round_prob_map(fixed_probs),
            "labels": _label_map(fixed_labels),
        },
    }


def _provenance_fields(artifacts: Any) -> dict[str, str]:
    model_path = artifacts.paths.model_path
    model_stem = model_path.stem
    model_info = artifacts.model_info or {}
    model_version = model_info.get("version")
    if not isinstance(model_version, str) or not model_version:
        model_version = model_stem
    return {
        "model_version": model_version,
        "model_stem": model_stem,
        "model_sha256": artifacts.model_sha256,
        "thresholds_sha256": artifacts.thresholds_sha256,
        "labels_sha256": artifacts.labels_sha256,
    }


def build_demo_feed(
    *,
    model_service: Any,
    rows_by_id: Mapping[int, Mapping[str, Any]],
    message_ids: Sequence[int],
    generated_at: str,
) -> dict[str, Any]:
    """
    Build a schema_version=1 demo feed from production predictions.

    Fails closed on ``ModelServiceError``. Display fields derive only from
    hierarchy-corrected labels/probabilities.
    """
    if not generated_at or not isinstance(generated_at, str):
        raise ValueError("generated_at must be a non-empty ISO-8601 string")
    if not message_ids:
        raise ValueError("message_ids must be a non-empty sequence")

    try:
        artifacts = model_service.get_production_artifacts()
        thresholds_map = model_service.get_thresholds_map()
    except ModelServiceError:
        raise
    except Exception as error:
        raise ModelServiceError(
            "Model unavailable: production artifact provenance failed."
        ) from error

    ordered_rows: list[Mapping[str, Any]] = []
    items: list[dict[str, Any]] = []

    for message_id in message_ids:
        row = rows_by_id.get(int(message_id))
        if row is None:
            raise KeyError(f"Requested message id {message_id} is missing from input rows")
        message = _canonicalize_text(row.get("message")).strip()
        if not message:
            raise ValueError(f"Message id {message_id} has empty message text")
        ordered_rows.append(row)

        try:
            prediction = model_service.predict(message)
            raw_probs = prediction.get("probabilities") or {}
            raw_labels = prediction.get("labels") or {}
            fixed_probs, fixed_labels = run_hierarchy_correction(
                dict(raw_probs), dict(thresholds_map)
            )
        except ModelServiceError:
            raise
        except Exception as error:
            raise ModelServiceError(
                f"Prediction failed for message id {message_id}."
            ) from error

        items.append(
            _build_feed_item(
                message_id=int(message_id),
                row=row,
                raw_probs=raw_probs,
                raw_labels=raw_labels,
                fixed_probs=fixed_probs,
                fixed_labels=fixed_labels,
            )
        )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        **_provenance_fields(artifacts),
        "input_message_ids": [int(message_id) for message_id in message_ids],
        "input_rows_sha256": hash_input_rows(ordered_rows),
        "items": items,
    }
    return payload


def serialize_demo_feed(payload: Mapping[str, Any]) -> bytes:
    """Serialize demo feed JSON with sorted keys and compact separators."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def load_rows_frame(
    *,
    database_path: Path | None = None,
    csv_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load staging messages with an ``id`` column.

    Staging SQLite ``stg_disaster_response`` currently lacks ``id`` (CSV sibling
    ``stg_disaster_messages.csv`` has ids). Prefer CSV when the DB has no ``id``;
    fail closed if neither source yields ids.
    """
    frame: pd.DataFrame | None = None
    if database_path is not None and Path(database_path).is_file():
        try:
            with sqlite3.connect(str(database_path)) as conn:
                frame = pd.read_sql_query("SELECT * FROM stg_disaster_response", conn)
        except (OSError, pd.errors.DatabaseError, sqlite3.Error, ValueError) as error:
            raise ValueError(f"Failed to read staging database {database_path}: {error}") from error

    if frame is not None and "id" in frame.columns:
        return frame

    resolved_csv = Path(csv_path) if csv_path is not None else None
    if resolved_csv is None and database_path is not None:
        sibling = Path(database_path).with_name("stg_disaster_messages.csv")
        if sibling.is_file():
            resolved_csv = sibling

    if resolved_csv is not None and resolved_csv.is_file():
        csv_frame = pd.read_csv(resolved_csv, low_memory=False)
        if "id" not in csv_frame.columns:
            raise ValueError(f"Staging CSV missing required 'id' column: {resolved_csv}")
        return csv_frame

    if frame is not None:
        raise ValueError(
            "Staging database has no 'id' column and no sibling "
            "stg_disaster_messages.csv was found. Rebuild the DB from CSV or "
            "point --database / CSV paths at sources that include message ids."
        )
    raise FileNotFoundError(
        "No staging database or CSV available to load message rows with ids."
    )


def rows_by_id_from_frame(df: pd.DataFrame, message_ids: Sequence[int]) -> dict[int, dict[str, Any]]:
    """Index required rows by id; fail if any id is missing or has empty message."""
    if "id" not in df.columns:
        raise ValueError("DataFrame must include an 'id' column")

    columns = ["id", "message"]
    if "original" in df.columns:
        columns.append("original")
    if "genre" in df.columns:
        columns.append("genre")

    working = df.loc[:, columns].copy()
    working["id"] = pd.to_numeric(working["id"], errors="coerce")
    working = working.dropna(subset=["id"])
    working["id"] = working["id"].astype(int)
    working = working.drop_duplicates(subset=["id"], keep="first").set_index("id")

    rows: dict[int, dict[str, Any]] = {}
    missing: list[int] = []
    empty: list[int] = []
    for message_id in message_ids:
        mid = int(message_id)
        if mid not in working.index:
            missing.append(mid)
            continue
        record = working.loc[mid]
        row = {
            "id": mid,
            "message": _canonicalize_text(
                record["message"] if "message" in working.columns else ""
            ),
            "original": _canonicalize_text(
                record["original"] if "original" in working.columns else ""
            ),
            "genre": _canonicalize_text(
                record["genre"] if "genre" in working.columns else ""
            ),
        }
        if not row["message"].strip():
            empty.append(mid)
        rows[mid] = row

    if missing:
        preview = ", ".join(str(item) for item in missing[:8])
        suffix = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        raise KeyError(f"Missing message id(s): {preview}{suffix}")
    if empty:
        preview = ", ".join(str(item) for item in empty[:8])
        suffix = f" (+{len(empty) - 8} more)" if len(empty) > 8 else ""
        raise ValueError(f"Empty message text for id(s): {preview}{suffix}")
    return rows
