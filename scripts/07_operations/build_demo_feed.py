#!/usr/bin/env python3
"""
Build a deterministic cached demo feed from production model predictions.

Uses hierarchy-corrected positive decisions only — never ground-truth category
columns or simulated probabilities.

Regeneration after model promotion
----------------------------------
After promoting a new production model, re-run this script **without**
``--init-ids`` so the pinned message ID list stays fixed and only predictions /
provenance hashes refresh:

  python scripts/07_operations/build_demo_feed.py \\
    --generated-at "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

First-time ID pinning (creates ``app/data/demo_feed_message_ids.json``):

  python scripts/07_operations/build_demo_feed.py --init-ids \\
    --generated-at "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

Staging note: SQLite ``stg_disaster_response`` may lack an ``id`` column. This
CLI falls back to sibling ``stg_disaster_messages.csv`` when the DB has no ids.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo root on sys.path so ``app`` and ``disasterproject`` imports resolve when
# invoked as ``python scripts/07_operations/build_demo_feed.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from app.config import Config  # noqa: E402
from app.services.demo_feed import (  # noqa: E402
    build_demo_feed,
    load_message_ids,
    load_rows_frame,
    rows_by_id_from_frame,
    select_initial_message_ids,
    serialize_demo_feed,
)
from app.services.errors import ModelServiceError  # noqa: E402
from app.services.model_service import ModelService  # noqa: E402

# ruff: noqa: I001  — imports follow sys.path bootstrap above


DEFAULT_OUTPUT = PROJECT_ROOT / "app" / "data" / "demo_feed.json"
DEFAULT_IDS_FILE = PROJECT_ROOT / "app" / "data" / "demo_feed_message_ids.json"


def _resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


def _write_json_bytes(path: Path, payload_bytes: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload_bytes)


def _write_ids_file(path: Path, message_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(message_ids, handle, indent=2)
        handle.write("\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic cached demo feed from production predictions"
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--ids-file",
        default=str(DEFAULT_IDS_FILE),
        help=f"Pinned message IDs JSON path (default: {DEFAULT_IDS_FILE})",
    )
    parser.add_argument(
        "--init-ids",
        action="store_true",
        help="Create the IDs file when absent (errors if the file already exists)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of IDs to pin with --init-ids (default: 50; requires --init-ids)",
    )
    parser.add_argument(
        "--generated-at",
        required=True,
        help="ISO-8601 timestamp embedded in the artifact (required for byte-stable regen)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override production model pickle path (default: Config.MODEL_PATH)",
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Override staging SQLite path (default: Config.DATABASE_PATH)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional staging CSV with id column (default: sibling of --database)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # --limit is only meaningful with --init-ids
    argv_tokens = argv if argv is not None else sys.argv[1:]
    limit_explicit = any(
        arg == "--limit" or arg.startswith("--limit=") for arg in argv_tokens
    )
    if limit_explicit and not args.init_ids:
        print("❌ --limit requires --init-ids")
        return 1
    if args.limit < 1:
        print(f"❌ --limit must be >= 1, got {args.limit}")
        return 1

    output_path = _resolve_path(args.output)
    ids_path = _resolve_path(args.ids_file)
    if args.model_path:
        model_path = _resolve_path(args.model_path)
    else:
        model_path = Path(Config.MODEL_PATH).resolve()
    database_path = (
        _resolve_path(args.database)
        if args.database
        else Path(Config.DATABASE_PATH).resolve()
    )
    csv_path = _resolve_path(args.csv) if args.csv else None

    try:
        print(f"📦 Loading production model: {model_path}")
        model_service = ModelService(model_path)
        model_service.load_model()
        artifacts = model_service.get_production_artifacts()
        print(
            f"✅ Provenance verified: stem={artifacts.paths.model_path.stem} "
            f"model={artifacts.model_sha256[:12]}..."
        )

        print("📄 Loading staging rows (CSV fallback when DB lacks id)...")
        frame = load_rows_frame(database_path=database_path, csv_path=csv_path)
        if "id" not in frame.columns:
            raise ValueError("Loaded frame has no id column")

        if args.init_ids:
            if ids_path.exists():
                print(
                    f"❌ IDs file already exists: {ids_path}\n"
                    "   Refusing to overwrite. Re-run without --init-ids to regenerate "
                    "the feed for the pinned IDs."
                )
                return 1
            message_ids = select_initial_message_ids(frame, n=args.limit)
            _write_ids_file(ids_path, message_ids)
            print(f"📌 Wrote {len(message_ids)} pinned IDs -> {ids_path}")
        else:
            if not ids_path.is_file():
                print(
                    f"❌ IDs file not found: {ids_path}\n"
                    "   Pass --init-ids once to create a pinned ID list."
                )
                return 1
            message_ids = load_message_ids(ids_path)
            print(f"📌 Loaded {len(message_ids)} pinned IDs from {ids_path}")

        rows_by_id = rows_by_id_from_frame(frame, message_ids)
        payload = build_demo_feed(
            model_service=model_service,
            rows_by_id=rows_by_id,
            message_ids=message_ids,
            generated_at=args.generated_at,
        )
        payload_bytes = serialize_demo_feed(payload)
        _write_json_bytes(output_path, payload_bytes)
        print(
            f"✅ Wrote demo feed ({len(payload['items'])} items, "
            f"{len(payload_bytes)} bytes) -> {output_path}"
        )
        print(f"   input_rows_sha256={payload['provenance']['input_rows_sha256']}")
        return 0
    except (FileNotFoundError, KeyError, ValueError, ModelServiceError) as error:
        print(f"❌ {error}")
        return 1
    except Exception as error:  # pragma: no cover - unexpected
        print(f"❌ Unexpected failure: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
