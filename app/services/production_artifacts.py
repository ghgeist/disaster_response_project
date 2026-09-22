"""
Strict production model/threshold/label provenance resolution.

Single shared contract for inference and (later) dashboard/export consumers.
No legacy ``thresholds.json`` / ``label_order.json`` fallbacks.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from disasterproject.utils.config import TARGET_COLUMNS

logger = logging.getLogger(__name__)


class ProductionArtifactError(ValueError):
    """Raised when production artifacts fail resolution or provenance checks."""


@dataclass(frozen=True)
class ProductionArtifactPaths:
    """Stem-bound production artifact locations (no legacy names)."""

    model_path: Path
    thresholds_path: Path
    labels_path: Path
    model_info_path: Path


@dataclass(frozen=True)
class ProductionArtifacts:
    """Validated production operating-point artifacts and provenance hashes."""

    paths: ProductionArtifactPaths
    thresholds: Dict[str, float]
    label_order: List[str]
    model_sha256: str
    thresholds_sha256: str
    labels_sha256: str
    model_info: Mapping[str, Any]


def compute_file_sha256(path: Path) -> str:
    """Return lowercase hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stem_bound_thresholds_path(model_path: Path) -> Path:
    """Return ``{stem}_thresholds.json`` next to the model pickle."""
    return model_path.parent / f"{model_path.stem}_thresholds.json"


def stem_bound_labels_path(model_path: Path) -> Path:
    """Return ``{stem}_labels.json`` next to the model pickle."""
    return model_path.parent / f"{model_path.stem}_labels.json"


def model_info_path_for(model_path: Path) -> Path:
    """Return ``MODEL_INFO.json`` in the model directory."""
    return model_path.parent / "MODEL_INFO.json"


def resolve_production_artifact_paths(model_path: Path) -> ProductionArtifactPaths:
    """
    Resolve stem-bound production companion paths.

    Raises:
        ProductionArtifactError: If the model path or required companions are missing.
    """
    resolved_model = Path(model_path)
    if not resolved_model.is_file():
        raise ProductionArtifactError(f"Production model file not found: {resolved_model}")

    thresholds_path = stem_bound_thresholds_path(resolved_model)
    labels_path = stem_bound_labels_path(resolved_model)
    info_path = model_info_path_for(resolved_model)

    if not thresholds_path.is_file():
        raise ProductionArtifactError(
            f"Required stem-bound thresholds artifact missing: {thresholds_path.name} "
            "(legacy thresholds.json is not accepted in production)"
        )
    if not labels_path.is_file():
        raise ProductionArtifactError(
            f"Required stem-bound labels artifact missing: {labels_path.name} "
            "(legacy label_order.json is not accepted in production)"
        )
    if not info_path.is_file():
        raise ProductionArtifactError(f"Required MODEL_INFO.json missing: {info_path}")

    return ProductionArtifactPaths(
        model_path=resolved_model,
        thresholds_path=thresholds_path,
        labels_path=labels_path,
        model_info_path=info_path,
    )


def validate_thresholds_map(
    raw_payload: Any,
    expected_labels: Sequence[str],
) -> Dict[str, float]:
    """
    Validate a thresholds artifact payload and return the per-label map.

    Accepts either ``{"thresholds": {...}}`` or a bare label->float mapping.
    Every expected label must be present with a finite value in ``[0, 1]``.
    """
    if not isinstance(raw_payload, dict):
        raise ProductionArtifactError(
            f"Thresholds artifact must be a JSON object, got {type(raw_payload).__name__}"
        )

    if "thresholds" in raw_payload:
        thresholds = raw_payload["thresholds"]
    else:
        thresholds = raw_payload

    if not isinstance(thresholds, dict):
        raise ProductionArtifactError(
            "Thresholds artifact missing deployable 'thresholds' object"
        )

    missing = [label for label in expected_labels if label not in thresholds]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        raise ProductionArtifactError(
            f"Thresholds map incomplete; missing {len(missing)} label(s): {preview}{suffix}"
        )

    validated: Dict[str, float] = {}
    invalid: List[str] = []
    for label in expected_labels:
        raw_value = thresholds[label]
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            invalid.append(f"{label}={raw_value!r} (not numeric)")
            continue
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            invalid.append(f"{label}={raw_value!r} (need finite value in [0, 1])")
            continue
        validated[label] = value

    if invalid:
        preview = ", ".join(invalid[:8])
        suffix = f" (+{len(invalid) - 8} more)" if len(invalid) > 8 else ""
        raise ProductionArtifactError(
            f"Thresholds map has {len(invalid)} out-of-range or invalid value(s): "
            f"{preview}{suffix}"
        )

    return validated


def validate_label_order(
    raw_payload: Any,
    expected_labels: Sequence[str],
) -> List[str]:
    """Require exact label-order coverage matching the production contract."""
    if not isinstance(raw_payload, list):
        raise ProductionArtifactError(
            f"Labels artifact must be a JSON array, got {type(raw_payload).__name__}"
        )

    labels = [str(item) for item in raw_payload]
    expected = list(expected_labels)

    if labels != expected:
        if set(labels) != set(expected):
            missing = [label for label in expected if label not in labels]
            extra = [label for label in labels if label not in expected]
            raise ProductionArtifactError(
                "Labels artifact coverage mismatch with production contract: "
                f"missing={missing[:8]!r} extra={extra[:8]!r}"
            )
        raise ProductionArtifactError(
            "Labels artifact order mismatch with production contract "
            f"(expected {len(expected)} labels in TARGET_COLUMNS order)"
        )

    return labels


def _load_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ProductionArtifactError(
            f"Corrupt JSON in production artifact {path.name}: {exc}"
        ) from exc
    except OSError as exc:
        raise ProductionArtifactError(
            f"Unable to read production artifact {path.name}: {exc}"
        ) from exc


def _require_sha256_field(model_info: Mapping[str, Any], field: str) -> str:
    value = model_info.get(field)
    normalized = value.lower() if isinstance(value, str) else ""
    if (
        not isinstance(value, str)
        or len(normalized) != 64
        or not all(ch in "0123456789abcdef" for ch in normalized)
    ):
        raise ProductionArtifactError(
            f"MODEL_INFO.json missing valid {field} (64-char hex SHA-256)"
        )
    return normalized


def resolve_production_artifacts(
    model_path: Path,
    *,
    model_info_path: Optional[Path] = None,
    expected_labels: Optional[Sequence[str]] = None,
) -> ProductionArtifacts:
    """
    Resolve and validate the production model/threshold/label operating point.

    Verifies:
    - Stem-bound thresholds and labels exist (no legacy fallbacks)
    - Threshold map completeness and value ranges
    - Label-order exact match to the production contract
    - Pickle / thresholds / labels SHA-256 match MODEL_INFO provenance
    """
    paths = resolve_production_artifact_paths(model_path)
    info_path = Path(model_info_path) if model_info_path is not None else paths.model_info_path
    if not info_path.is_file():
        raise ProductionArtifactError(f"Required MODEL_INFO.json missing: {info_path}")

    labels_contract = list(expected_labels) if expected_labels is not None else list(TARGET_COLUMNS)

    model_info_raw = _load_json(info_path)
    if not isinstance(model_info_raw, dict):
        raise ProductionArtifactError("MODEL_INFO.json must be a JSON object")

    thresholds_payload = _load_json(paths.thresholds_path)
    labels_payload = _load_json(paths.labels_path)

    thresholds = validate_thresholds_map(thresholds_payload, labels_contract)
    label_order = validate_label_order(labels_payload, labels_contract)

    expected_model_sha = _require_sha256_field(model_info_raw, "sha256")
    expected_thresholds_sha = _require_sha256_field(model_info_raw, "thresholds_sha256")
    expected_labels_sha = _require_sha256_field(model_info_raw, "labels_sha256")

    actual_model_sha = compute_file_sha256(paths.model_path)
    actual_thresholds_sha = compute_file_sha256(paths.thresholds_path)
    actual_labels_sha = compute_file_sha256(paths.labels_path)

    if actual_model_sha != expected_model_sha:
        raise ProductionArtifactError(
            "Production model SHA-256 mismatch vs MODEL_INFO.sha256 "
            f"(actual={actual_model_sha[:16]}... expected={expected_model_sha[:16]}...)"
        )
    if actual_thresholds_sha != expected_thresholds_sha:
        raise ProductionArtifactError(
            "Production thresholds SHA-256 mismatch vs MODEL_INFO.thresholds_sha256 "
            f"(actual={actual_thresholds_sha[:16]}... expected={expected_thresholds_sha[:16]}...)"
        )
    if actual_labels_sha != expected_labels_sha:
        raise ProductionArtifactError(
            "Production labels SHA-256 mismatch vs MODEL_INFO.labels_sha256 "
            f"(actual={actual_labels_sha[:16]}... expected={expected_labels_sha[:16]}...)"
        )

    logger.info(
        "Production artifacts verified for %s (model=%s... thresholds=%s... labels=%s...)",
        paths.model_path.name,
        actual_model_sha[:12],
        actual_thresholds_sha[:12],
        actual_labels_sha[:12],
    )

    return ProductionArtifacts(
        paths=ProductionArtifactPaths(
            model_path=paths.model_path,
            thresholds_path=paths.thresholds_path,
            labels_path=paths.labels_path,
            model_info_path=info_path,
        ),
        thresholds=thresholds,
        label_order=label_order,
        model_sha256=actual_model_sha,
        thresholds_sha256=actual_thresholds_sha,
        labels_sha256=actual_labels_sha,
        model_info=model_info_raw,
    )
