import datetime
import threading
from abc import ABC
from dataclasses import field, dataclass
from os.path import dirname
from pathlib import Path
from typing import Optional, List, Dict, Type

from giskardpy.executor import Pacer, SimulationPacer
from krrood.symbolic_math.symbolic_math import FloatVariable
from segmind import logger, set_logger_level, LogLevel
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import FileMesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body, Agent
from .datastructures.events import EventUnion, ContactEvent, Event, CloseContactEvent
from .datastructures.object_tracker import ObjectTracker
from .detectors.atomic_event_detectors_nodes import SegmindContext
from .detectors.base import DetectorStateChart
from .episode_player import EpisodePlayer
from .event_logger import EventLogger

set_logger_level(LogLevel.DEBUG)


@dataclass
class EpisodeSegmenterExecutor:
    context: SegmindContext
    player: EpisodePlayer | None = None
    pacer: Pacer = field(default_factory=SimulationPacer)
    statechart: DetectorStateChart = field(init=False)
    _control_cycle_index: int = field(init=False)
    _time_variable: FloatVariable = field(init=False)




    def start(self):
        if self.player:
            self.player.start()


    def compile(self, statechart: DetectorStateChart):
        self.statechart = statechart
        self.control_cycles = 0
        self.statechart.compile(self.context)
        # self.context.collision_manager.update_collision_matrix()
        # do one tick to immediately active nodes whose start condition is constant true.
        self.fill_holes()
        self.statechart.tick(self.context)
        if self.player:
            self.player.start()


    def fill_holes(self):
        for o in self.context.world.bodies:
            if "hole" in o.name.name:
                self.context.holes.append(o)

    def tick(self):
        #self.player.pause()
        #self.control_cycles += 1
        if hasattr(self.context, "tick_count"):
            self.context.tick_count += 1
        self.statechart.tick(self.context)
        #self.player.resume()
        # ToDo: Here we need to add the state model updates.

    def tick_until_end(self, timeout: int = 1_000):
        """
        Calls tick until is_end_motion() returns True.
        :param timeout: Max number of ticks to perform.
        #ToDo: So in the Dataplayer thread we can add an EndMotion Node and that will trigger the end.
        """
        try:
            for i in range(timeout):
                if self.player.is_alive():
                    self.tick()
                    self.pacer.sleep()
                else:
                    return
            raise TimeoutError("Timeout reached while waiting for end of motion.")
        finally:
            self.statechart.cleanup_nodes(context=self.context)
            self.context.cleanup()

    def spawn_scene(self, models_dir):
        directory = Path(models_dir)
        urdf_files = [f.name for f in directory.glob("*.urdf")]
        stl_files = [f.name for f in directory.glob("*.stl")]
        if urdf_files:
            for file in urdf_files:
                file_path = models_dir + file
                obj_name = Path(file).stem

                if obj_name == "iCub":
                    continue
                try:
                    if obj_name == "scene":
                        obj_world = URDFParser.from_file(file_path).parse()
                        world_C_scene = FixedConnection(
                            parent=self.context.world.root, child=obj_world.root
                        )
                        with self.context.world.modify_world():
                            self.context.world.merge_world(obj_world, world_C_scene)
                    else:
                        obj_world = URDFParser.from_file(file_path).parse()
                        with self.context.world.modify_world():
                            self.context.world.merge_world(obj_world)

                except Exception as e:
                    # import pdb
                    # pdb.set_trace()

                    continue

        'obj_000001'
        if stl_files:
            for file in stl_files:
                file_path = models_dir + file
                obj_name = Path(file).stem


                new_body = Body(name=PrefixedName(obj_name), visual=ShapeCollection([FileMesh.from_file(file_path)]), collision=ShapeCollection([FileMesh.from_file(file_path)]))
                with self.context.world.modify_world():
                    new_body_C_root = Connection6DoF.create_with_dofs(world=self.context.world,
                                                                  parent=self.context.world.root, child=new_body)
                    self.context.world.add_connection(new_body_C_root)
