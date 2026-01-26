"""
Model artifact loading utilities (thresholds, label order).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ModelArtifactLoader:
    """Load model artifacts from the model directory."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path

    def load_artifacts(self) -> Tuple[Optional[Dict[str, float]], Optional[List[str]]]:
        """Load thresholds and label order from disk if present."""
        try:
            thresholds = self._load_thresholds()
            label_order = self._load_label_order()
            return thresholds, label_order
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Failed loading model artifacts (thresholds/label_order): %s", exc)
            return None, None

    def _load_thresholds(self) -> Optional[Dict[str, float]]:
        model_dir = self.model_path.parent
        model_stem = self.model_path.stem
        thresholds_candidates = [
            model_dir / f"{model_stem}_thresholds.json",
            model_dir / "optimized_critical_thresholds.json",
            model_dir / "optimized_all_thresholds.json",
            model_dir / "thresholds.json",
        ]

        for thresholds_path in thresholds_candidates:
            if thresholds_path.exists():
                with open(thresholds_path, "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)

                if isinstance(loaded_data, dict) and "thresholds" in loaded_data:
                    thresholds = loaded_data["thresholds"]
                    logger.info(
                        "Loaded optimized thresholds from %s (target recall: %s)",
                        thresholds_path.name,
                        loaded_data.get("metadata", {}).get("target_recall", "unknown"),
                    )
                    return thresholds

                logger.info("Loaded thresholds from %s", thresholds_path.name)
                return loaded_data

        return None

    def _load_label_order(self) -> Optional[List[str]]:
        model_dir = self.model_path.parent
        model_stem = self.model_path.stem
        label_order_candidates = [
            model_dir / f"{model_stem}_labels.json",
            model_dir / "label_order.json",
        ]

        for label_order_path in label_order_candidates:
            if label_order_path.exists():
                with open(label_order_path, "r", encoding="utf-8") as f:
                    label_order = json.load(f)
                logger.info("Loaded label order from %s", label_order_path.name)
                return label_order

        return None
