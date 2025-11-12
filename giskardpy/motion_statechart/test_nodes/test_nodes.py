from dataclasses import dataclass
from typing import Optional

from giskardpy.motion_statechart.context import ExecutionContext
from giskardpy.motion_statechart.graph_node import MotionStatechartNode


@dataclass
class ChangeStateOnEvents(MotionStatechartNode):
    state: Optional[str] = None

    def on_start(self, context: ExecutionContext):
        self.state = "on_start"

    def on_pause(self, context: ExecutionContext):
        self.state = "on_pause"

    def on_unpause(self, context: ExecutionContext):
        self.state = "on_unpause"

    def on_end(self, context: ExecutionContext):
        self.state = "on_end"

    def on_reset(self, context: ExecutionContext):
        self.state = "on_reset"
