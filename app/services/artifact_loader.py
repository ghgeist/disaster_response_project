"""
Model artifact loading utilities (thresholds, label order).

Production loading delegates to the shared strict provenance resolver.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .production_artifacts import (
    ProductionArtifactError,
    ProductionArtifacts,
    resolve_production_artifacts,
)

logger = logging.getLogger(__name__)


class ModelArtifactLoader:
    """Load and validate production model companion artifacts."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._resolved: Optional[ProductionArtifacts] = None

    def load_artifacts(self) -> Tuple[Dict[str, float], List[str]]:
        """
        Load production thresholds and label order with provenance checks.

        Raises:
            ProductionArtifactError: On missing, malformed, incomplete, or
                hash-mismatched production artifacts.
        """
        artifacts = self.resolve()
        return artifacts.thresholds, artifacts.label_order

    def resolve(self) -> ProductionArtifacts:
        """Return the validated production artifact bundle (cached)."""
        if self._resolved is not None:
            return self._resolved
        try:
            self._resolved = resolve_production_artifacts(self.model_path)
        except ProductionArtifactError:
            logger.error(
                "Production artifact provenance failed for %s",
                self.model_path,
                exc_info=False,
            )
            raise
        return self._resolved
