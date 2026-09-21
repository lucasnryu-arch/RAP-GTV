"""Small immutable identities for foundation-level provenance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class IdentityRecord:
    """Identity of one input, configuration, or frozen specification."""

    kind: Literal["input", "config", "spec"]
    name: str
    sha256: str
    version: str | None = None
    path: Path | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        digest = self.sha256.lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("sha256 must be a 64-character hexadecimal digest")
        object.__setattr__(self, "sha256", digest)
        if self.path is not None:
            object.__setattr__(self, "path", Path(self.path))

