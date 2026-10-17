"""Immutable prompt versions. Promotion moves the active pointer. Text never changes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptVersion:
    name: str
    version: int
    template: str
    note: str


class RegistryError(ValueError):
    pass


class PromptRegistry:
    def __init__(self) -> None:
        self._versions: dict[str, list[PromptVersion]] = {}
        self._active: dict[str, int] = {}

    def add(self, name: str, template: str, note: str = "") -> PromptVersion:
        if not name.strip():
            raise RegistryError("name is blank")
        if not template.strip():
            raise RegistryError("template is blank")
        existing = self._versions.get(name, [])
        for item in existing:
            if item.template == template:
                return item
        version = len(existing) + 1
        item = PromptVersion(name=name, version=version, template=template, note=note)
        self._versions.setdefault(name, []).append(item)
        self._active.setdefault(name, version)
        return item

    def get(self, name: str, version: int | None = None) -> PromptVersion:
        rows = self._versions.get(name)
        if not rows:
            raise RegistryError(f"unknown prompt {name}")
        wanted = self._active[name] if version is None else version
        for item in rows:
            if item.version == wanted:
                return item
        raise RegistryError(f"unknown version {wanted} for {name}")

    def promote(self, name: str, version: int) -> PromptVersion:
        item = self.get(name, version)
        self._active[name] = item.version
        return item

    def active_version(self, name: str) -> int:
        if name not in self._active:
            raise RegistryError(f"unknown prompt {name}")
        return self._active[name]

    def versions(self, name: str) -> list[int]:
        return [item.version for item in self._versions.get(name, [])]
