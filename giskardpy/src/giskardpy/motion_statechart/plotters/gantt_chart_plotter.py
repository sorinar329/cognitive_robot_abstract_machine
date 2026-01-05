from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, TYPE_CHECKING

import numpy as np

from giskardpy.middleware import get_middleware
from giskardpy.motion_statechart.plotters.styles import (
    LiftCycleStateToColor,
    ObservationStateToColor,
)
from giskardpy.utils.utils import create_path

if TYPE_CHECKING:  # avoid circular import at runtime
    from giskardpy.motion_statechart.motion_statechart import MotionStatechart


@dataclass
class HistoryGanttChartPlotter:
    """
    Takes the History of a MotionStatechart and plots a Gantt chart of the node states over control cycles.
    """

    motion_statechart: MotionStatechart

    def plot_gantt_chart(self, file_name: str) -> None:
        """
        Plot a Gantt-style chart of node states over control cycles using StateHistory.
        """
        import matplotlib.pyplot as plt

        nodes = self.motion_statechart.nodes
        if len(nodes) == 0:
            get_middleware().logwarn(
                "Gantt chart skipped: no nodes in motion statechart."
            )
            return

        history = self.motion_statechart.history.history
        if len(history) == 0:
            get_middleware().logwarn("Gantt chart skipped: empty StateHistory.")
            return

        # Build y-axis mapping
        names = [n.name[:50] for n in nodes]
        positions: Dict[str, int] = {name: i for i, name in enumerate(names)}

        # Figure sizing heuristics based on number of nodes and duration
        last_cycle = max(item.control_cycle for item in history)
        num_bars = len(names)
        figure_height = 0.7 + num_bars * 0.25
        figure_width = max(4.0, 0.5 * float(last_cycle + 1))

        plt.figure(figsize=(figure_width, figure_height))
        plt.grid(True, axis="x", zorder=-1)

        # Initialize per-node tracking from the first history item
        start_cycle = history[0].control_cycle
        current_life = [history[0].life_cycle_state[n] for n in nodes]
        current_obs = [history[0].observation_state[n] for n in nodes]
        segment_start = [start_cycle for _ in nodes]

        def flush_segments(upto_cycle: int) -> None:
            bar_height = 0.8
            for idx, node in enumerate(nodes):
                y = positions[node.name[:50]]
                lc = current_life[idx]
                oc = current_obs[idx]
                x0 = segment_start[idx]
                width = upto_cycle - x0
                if width <= 0:
                    continue
                # Top half: life cycle
                plt.barh(
                    y + bar_height / 4,
                    width,
                    height=bar_height / 2,
                    left=x0,
                    color=LiftCycleStateToColor[lc],
                    zorder=2,
                )
                # Bottom half: observation
                plt.barh(
                    y - bar_height / 4,
                    width,
                    height=bar_height / 2,
                    left=x0,
                    color=ObservationStateToColor[oc],
                    zorder=2,
                )

        # Iterate history and draw when something changes
        for item in history[1:]:
            next_cycle = item.control_cycle
            changed = False
            for i, node in enumerate(nodes):
                new_life = item.life_cycle_state[node]
                new_obs = item.observation_state[node]
                if new_life != current_life[i] or new_obs != current_obs[i]:
                    changed = True
            if changed:
                flush_segments(upto_cycle=next_cycle)
                # Update state and segment starts for changed nodes
                for i, node in enumerate(nodes):
                    new_life = item.life_cycle_state[node]
                    new_obs = item.observation_state[node]
                    if new_life != current_life[i] or new_obs != current_obs[i]:
                        current_life[i] = new_life
                        current_obs[i] = new_obs
                        segment_start[i] = next_cycle

        # Flush until last+1 to terminate bars
        flush_segments(upto_cycle=last_cycle + 1)

        # Axes formatting
        plt.xlabel("Control cycle")
        plt.xlim(start_cycle, last_cycle + 1)
        plt.xticks(
            np.arange(
                start_cycle,
                last_cycle + 2,
                max(1, (last_cycle - start_cycle + 1) // 10),
            )
        )
        plt.ylabel("Nodes")
        plt.ylim(-0.8, num_bars - 1 + 0.8)
        plt.yticks([positions[name] for name in names], names)
        plt.gca().yaxis.tick_right()
        plt.tight_layout()

        create_path(file_name)
        plt.savefig(file_name)
        plt.close()
        get_middleware().loginfo(f"Saved gantt chart to {file_name}.")
