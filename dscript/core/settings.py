"""
Settings from a YAML file and command-line flags, generated from one pydantic
model per command. Each option is declared once, as a field.

Precedence, lowest to highest: defaults < YAML file (``--config``) < flags
typed on the command line. YAML keys may use ``hidden_dim`` or ``hidden-dim``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, CliSettingsSource, SettingsConfigDict

S = TypeVar("S", bound="Settings")


class Settings(BaseSettings):
    """Base for every command's settings. Flat: one field per flag."""

    model_config = SettingsConfigDict(
        cli_kebab_case=True,  # --hidden-dim, not --hidden_dim
        cli_implicit_flags=True,  # --weighting / --no-weighting
        extra="forbid",  # unknown YAML keys are an error, not ignored
    )

    config: Path | None = Field(None, description="YAML file of settings; flags override it")

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, *args, **kwargs):
        # Only explicit values: no environment variables or .env files, which
        # would silently change a run.
        return (init_settings,)

    @classmethod
    def parse(cls: type[S], argv: list[str], prog: str) -> S:
        """Build from command-line arguments (without the program name)."""
        flags = CliSettingsSource(cls, cli_parse_args=argv, cli_prog_name=prog)()  # only flags typed
        values: dict[str, Any] = {}
        if flags.get("config"):
            values.update(read_yaml(Path(flags["config"])))
        values.update(flags)
        return cls(**values)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json", exclude={"config"}), sort_keys=False)


def read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping of setting names to values")
    return {str(k).replace("-", "_"): v for k, v in data.items()}
