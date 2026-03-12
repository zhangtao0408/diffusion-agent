"""Scenario definitions and base interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from diffusion_agent.agents.state import AgentState


class Scenario(str, Enum):
    CHECK = "check"
    ADAPT = "adapt"
    ANALYZE = "analyze"
    OPTIMIZE = "optimize"
    VERIFY = "verify"


class ScenarioBase(ABC):
    """Base interface for all scenario implementations."""

    @abstractmethod
    def plan(self, state: AgentState) -> list[dict[str, Any]]:
        """Break the scenario into a list of features for the feature-list."""

    @abstractmethod
    def execute(self, state: AgentState) -> AgentState:
        """Execute the current feature within this scenario."""
