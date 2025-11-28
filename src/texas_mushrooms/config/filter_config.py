"""Mushroom taxonomy filter configuration.

Provides a YAML-configurable filter for selecting "interesting" mushrooms
based on genus-level taxonomy, explicit species lists, and caption keywords.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Default config path relative to project root
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "mushroom_filter.yaml"


@dataclass
class MushroomFilter:
    """Configuration for filtering mushroom observations by taxonomy.
    
    Attributes:
        exclude_groups: Morphological group names (documentation only).
        exclude_genera: Genera to exclude (e.g., Trametes, Stereum).
        exclude_species: Specific species to exclude.
        include_species: Species to include even if genus is excluded.
        exclude_caption_keywords: Keywords in captions that trigger exclusion.
    """
    
    exclude_groups: list[str] = field(default_factory=list)
    exclude_genera: list[str] = field(default_factory=list)
    exclude_species: list[str] = field(default_factory=list)
    include_species: list[str] = field(default_factory=list)
    exclude_caption_keywords: list[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Normalize lists for case-insensitive matching."""
        # Handle None values from YAML (empty keys parse as None, not [])
        if self.exclude_groups is None:
            self.exclude_groups = []
        if self.exclude_genera is None:
            self.exclude_genera = []
        if self.exclude_species is None:
            self.exclude_species = []
        if self.include_species is None:
            self.include_species = []
        if self.exclude_caption_keywords is None:
            self.exclude_caption_keywords = []
        
        # Normalize genera to title case for consistent matching
        self._exclude_genera_set = {g.strip().title() for g in self.exclude_genera}
        self._exclude_species_set = {s.strip().lower() for s in self.exclude_species}
        self._include_species_set = {s.strip().lower() for s in self.include_species}
        self._exclude_keywords_lower = [kw.lower() for kw in self.exclude_caption_keywords]
    
    @classmethod
    def from_yaml(cls, path: Optional[Path] = None) -> "MushroomFilter":
        """Load filter configuration from a YAML file.
        
        Args:
            path: Path to YAML config file. Uses default if not provided.
            
        Returns:
            Configured MushroomFilter instance.
            
        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If YAML parsing fails.
        """
        config_path = path or DEFAULT_CONFIG_PATH
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        logger.info(f"Loading mushroom filter config from {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        filter_config = data.get("filter", {})
        
        return cls(
            exclude_groups=filter_config.get("exclude_groups", []),
            exclude_genera=filter_config.get("exclude_genera", []),
            exclude_species=filter_config.get("exclude_species", []),
            include_species=filter_config.get("include_species", []),
            exclude_caption_keywords=filter_config.get("exclude_caption_keywords", []),
        )
    
    @classmethod
    def empty(cls) -> "MushroomFilter":
        """Create an empty filter that includes everything."""
        return cls()
    
    def _extract_genus(self, species_name: str) -> str:
        """Extract genus (first word) from a species name.
        
        Args:
            species_name: Full species name (e.g., "Amanita muscaria").
            
        Returns:
            Genus name in title case, or empty string if invalid.
        """
        if not species_name or not isinstance(species_name, str):
            return ""
        parts = species_name.strip().split()
        return parts[0].title() if parts else ""
    
    def should_include(
        self,
        species_name: str,
        caption: str = "",
    ) -> bool:
        """Determine if a mushroom observation should be included.
        
        Filtering logic (in order):
        1. If species is in include_species whitelist -> INCLUDE
        2. If species is in exclude_species blacklist -> EXCLUDE
        3. If genus is in exclude_genera blacklist -> EXCLUDE
        4. If caption contains any exclude_caption_keywords -> EXCLUDE
        5. Otherwise -> INCLUDE
        
        Args:
            species_name: Latin binomial species name.
            caption: Photo caption text (optional).
            
        Returns:
            True if observation should be included, False otherwise.
        """
        species_lower = species_name.strip().lower() if species_name else ""
        
        # 1. Explicit include takes priority
        if species_lower in self._include_species_set:
            return True
        
        # 2. Explicit species exclude
        if species_lower in self._exclude_species_set:
            return False
        
        # 3. Genus-level exclude
        genus = self._extract_genus(species_name)
        if genus in self._exclude_genera_set:
            return False
        
        # 4. Caption keyword exclude
        if caption:
            caption_lower = caption.lower()
            for keyword in self._exclude_keywords_lower:
                if keyword in caption_lower:
                    return False
        
        # 5. Default: include
        return True
    
    def get_exclusion_reason(
        self,
        species_name: str,
        caption: str = "",
    ) -> Optional[str]:
        """Get the reason why a species would be excluded, if any.
        
        Useful for debugging/understanding filter behavior.
        
        Args:
            species_name: Latin binomial species name.
            caption: Photo caption text (optional).
            
        Returns:
            Reason string if excluded, None if included.
        """
        species_lower = species_name.strip().lower() if species_name else ""
        
        if species_lower in self._include_species_set:
            return None  # Explicitly included
        
        if species_lower in self._exclude_species_set:
            return f"species in exclude list: {species_name}"
        
        genus = self._extract_genus(species_name)
        if genus in self._exclude_genera_set:
            return f"genus excluded: {genus}"
        
        if caption:
            caption_lower = caption.lower()
            for keyword in self._exclude_keywords_lower:
                if keyword in caption_lower:
                    return f"caption keyword: '{keyword}'"
        
        return None  # Not excluded
    
    def filter_species_list(self, species_list: list[str]) -> list[str]:
        """Filter a list of species names.
        
        Args:
            species_list: List of species names to filter.
            
        Returns:
            List containing only included species.
        """
        return [sp for sp in species_list if self.should_include(sp)]
    
    def summary(self) -> str:
        """Return a summary of the filter configuration."""
        return (
            f"MushroomFilter(\n"
            f"  exclude_groups={len(self.exclude_groups)},\n"
            f"  exclude_genera={len(self.exclude_genera)},\n"
            f"  exclude_species={len(self.exclude_species)},\n"
            f"  include_species={len(self.include_species)},\n"
            f"  exclude_keywords={len(self.exclude_caption_keywords)}\n"
            f")"
        )
    
    def __repr__(self) -> str:
        return self.summary()
