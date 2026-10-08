"""finclaim: a financial research agent whose every claim is checked against its sources."""
from .agent import Agent, AgentConfig
from .state import Checkpointer, RunState
from .tools import FaultInjector, FixtureBackend, LiveBackend, ToolRegistry

__all__ = ["Agent", "AgentConfig", "Checkpointer", "RunState", "ToolRegistry",
           "LiveBackend", "FixtureBackend", "FaultInjector"]
__version__ = "0.1.0"
