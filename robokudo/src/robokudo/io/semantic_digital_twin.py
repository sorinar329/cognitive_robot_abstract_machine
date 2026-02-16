import copy
import dataclasses
import importlib
import uuid

from ament_index_python import get_packages_with_prefixes
from random_events.utils import recursive_subclasses

from robokudo.types.annotation import BoundingBox3DAnnotation
from robokudo.utils.annotator_helper import transform_pose_from_cam_to_world
from robokudo.utils.comparators import (
    TranslationComparator,
    ClassnameComparator,
    HistogramComparator,
    SemanticColorComparator,
    RoiComparator,
    AdditionalDataComparator,
    BboxComparator,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import TransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.geometry import Shape, Color, Scale, Box
from semantic_digital_twin.world_description.world_entity import Body, SemanticAnnotation
from semantic_digital_twin.world_description.world_modification import (
    AddConnectionModification,
    AddDegreeOfFreedomModification,
    AddSemanticAnnotationModification,
    RemoveSemanticAnnotationModification,
    RemoveBodyModification,
    RemoveConnectionModification,
    RemoveDegreeOfFreedomModification, AddKinematicStructureEntityModification,
)


@dataclasses.dataclass
class Object:
    data: dict


@dataclasses.dataclass
class TrackedObject:
    obj: Object
    body: Body
    semantic_annotations: list
    conns: list
    uid: uuid.UUID = dataclasses.field(default_factory=lambda: uuid.uuid4())


class UpdateSemanticAnnotationCommand:
    def __init__(self, adapter: "SemanticDigitalTwinAdapter", old_semantic_annotation: SemanticAnnotation,
                 new_semantic_annotation: SemanticAnnotation):
        self.adapter = adapter
        self.semantic_annotation = old_semantic_annotation
        self.old_semantic_annotation = copy.deepcopy(old_semantic_annotation)
        self.new_semantic_annotation = new_semantic_annotation

    def apply(self, world):
        # with self.adapter.world.modify_world():
        raise NotImplementedError("semantic_annotation updates are currently not implemented")

    def undo(self, world):
        # with self.adapter.world.modify_world():
        raise NotImplementedError("semantic_annotation updates are currently not implemented")


class AddCollisionCommand:
    def __init__(
            self, adapter: "SemanticDigitalTwinAdapter", body: Body, new_collision: Shape
    ):
        self.adapter = adapter
        self.body = body
        self.new_collision = new_collision

    def apply(self, world):
        self.body.collision.append(self.new_collision)

    def undo(self, world):
        self.body.collision.remove(self.new_collision)


class UpdateCollisionCommand:
    def __init__(
            self,
            adapter: "SemanticDigitalTwinAdapter",
            old_collision: Shape,
            new_collision: Shape,
    ):
        self.adapter = adapter
        self.collision = old_collision
        self.old_collision = copy.deepcopy(old_collision)
        self.new_collision = new_collision

        if type(self.old_collision) != type(self.new_collision):
            raise ValueError(
                "cannot update collision to different shape type"
            )

    def apply(self, world):
        old_fields = dataclasses.fields(self.old_collision)
        for field in old_fields:
            old_value = getattr(self.old_collision, field.name)
            new_value = getattr(self.new_collision, field.name)

            if old_value != new_value:
                setattr(self.collision, field.name, new_value)

    def undo(self, world):
        old_fields = dataclasses.fields(self.old_collision)
        for field in old_fields:
            old_value = getattr(self.old_collision, field.name)
            new_value = getattr(self.new_collision, field.name)

            if old_value != new_value:
                setattr(self.collision, field.name, old_value)


class RemoveCollisionCommand:
    def __init__(
            self, adapter: "SemanticDigitalTwinAdapter", body: Body, old_collision: Shape
    ):
        self.adapter = adapter
        self.body = body
        self.old_collision = old_collision

    def apply(self, world):
        self.body.collision.remove(self.old_collision)

    def undo(self, world):
        self.body.collision.append(self.old_collision)


class AddObjectDiff:
    def __init__(self, adapter: "SemanticDigitalTwinAdapter", new_object: Object):
        self.adapter = adapter
        self.new_object = new_object
        self.tracked_object = self.adapter.object_to_tracked_object(new_object)

        self.commands = list()

        self.commands.append(
            AddKinematicStructureEntityModification(
                kinematic_structure_entity=self.tracked_object.body
            )
        )

        conn_name = PrefixedName(
            name=f"{self.adapter.root.name.name}_{self.tracked_object.body.name.name}"
        )

        dofs = {}
        for name in ["x", "y", "z", "qx", "qy", "qz", "qw"]:
            prefixed_name = PrefixedName(name=name, prefix=conn_name.name)
            dof = DegreeOfFreedom(name=prefixed_name)

            dofs[name] = dof

            self.commands.append(AddDegreeOfFreedomModification(dof=dof))

        conn = Connection6DoF(
            parent=self.adapter.root,
            child=self.tracked_object.body,
            x_name=dofs["x"].name,
            y_name=dofs["y"].name,
            z_name=dofs["z"].name,
            qx_name=dofs["qx"].name,
            qy_name=dofs["qy"].name,
            qz_name=dofs["qz"].name,
            qw_name=dofs["qw"].name,
        )
        self.commands.append(AddConnectionModification(connection=conn))

        for semantic_annotation in self.tracked_object.semantic_annotations:
            self.commands.append(AddSemanticAnnotationModification(semantic_annotation=semantic_annotation))

    def apply(self):
        with self.adapter.world.modify_world():
            for command in self.commands:
                command.apply(self.adapter.world)
        self.adapter.tracked_objects.append(self.tracked_object)

    def undo(self):
        with self.adapter.world.modify_world():
            for command in self.commands:
                command.undo(self.adapter.world)
        self.adapter.tracked_objects.remove(self.tracked_object)


class UpdateObjectDiff:
    def __init__(
            self,
            adapter: "SemanticDigitalTwinAdapter",
            old_object: TrackedObject,
            new_object: Object,
    ):
        self.adapter = adapter
        self.old_object = old_object
        self.new_object = new_object
        self.tracked_object = self.adapter.object_to_tracked_object(new_object)

        self.commands = list()

        # Assume a single simple body as collision for perceived, dynamic objects
        if (
                len(self.old_object.body.collision) == 0
                and len(self.tracked_object.body.collision) == 1
        ):
            # Add the collision
            self.commands.append(
                AddCollisionCommand(
                    self.adapter,
                    self.old_object.body,
                    self.tracked_object.body.collision[0],
                )
            )
        elif (
                len(self.old_object.body.collision) == 1
                and len(self.tracked_object.body.collision) == 0
        ):
            # Remove the collision
            self.commands.append(
                RemoveCollisionCommand(
                    self.adapter,
                    self.old_object.body,
                    self.old_object.body.collision[0],
                )
            )
        elif (
                len(self.old_object.body.collision) == 1
                and len(self.tracked_object.body.collision) == 1
        ):
            # Update the collision
            self.commands.append(
                UpdateCollisionCommand(
                    self.adapter,
                    self.old_object.body.collision[0],
                    self.tracked_object.body.collision[0],
                )
            )

    def apply(self):
        with self.adapter.world.modify_world():
            for command in self.commands:
                command.apply(self.adapter.world)

    def undo(self):
        with self.adapter.world.modify_world():
            for command in self.commands:
                command.undo(self.adapter.world)


class RemoveObjectDiff:
    def __init__(self, adapter: "SemanticDigitalTwinAdapter", old_object: TrackedObject):
        self.adapter = adapter
        self.old_object = old_object

        self.commands = list()

        self.commands.append(RemoveBodyModification(body_name=old_object.body.name))

        for connection in self.old_object.conns:
            self.commands.append(
                RemoveConnectionModification(connection_name=connection.name)
            )
            for dof in connection.dofs.values():
                self.commands.append(
                    RemoveDegreeOfFreedomModification(dof_name=dof.name)
                )

        for semantic_annotation in self.old_object.semantic_annotations:
            self.commands.append(RemoveSemanticAnnotationModification(semantic_annotation=semantic_annotation))

    def apply(self):
        with self.adapter.world.modify_world():
            for command in self.commands:
                command.apply(self.adapter.world)
        self.adapter.tracked_objects.remove(self.old_object)

    def undo(self):
        with self.adapter.world.modify_world():
            for command in self.commands:
                command.undo(self.adapter.world)
        self.adapter.tracked_objects.append(self.old_object)


def load_semantic_annotation_extensions():
    for package in get_packages_with_prefixes().keys():
        try:
            module = f"{package}.descriptors.object_knowledge"
            importlib.import_module(module)
        except ModuleNotFoundError:
            pass
        except Exception as e:
            print(f"Error loading semantic world extensions from package {package}: {e}")


class SemanticDigitalTwinAdapter:
    """Class to convert RoboKudo concepts to the SemanticWorld."""

    def __init__(self, cas, urdf_path: str = None, semantic_annotation_sources: list = None):
        load_semantic_annotation_extensions()

        if urdf_path is not None:
            parser = URDFParser.from_file(file_path=urdf_path)
            self.world = parser.parse()
            self.root = self.world.root
            self.world.validate()
        else:
            self.world = World()  # Adapter holds the main world instance
            self.root = Body(
                name=PrefixedName(name="root", prefix="world")
            )  # Use Prefix "RoboKudo"?

        with self.world.modify_world():
            self.world.add_kinematic_structure_entity(self.root)

        self.cas = cas

        self.tracked_objects: list[TrackedObject] = list()

        self.comparators = {
            "translation_vector": TranslationComparator(weight=0.4, max_distance=0.5),
            "class": ClassnameComparator(weight=0.4),
            "bbox": BboxComparator(weight=0.2),
            "color_histogram": HistogramComparator(weight=0.3),
            "semantic_color": SemanticColorComparator(weight=0.2),
            "roi": RoiComparator(weight=0),
        }

        self.semantic_color_to_rgb = {
            "red": Color(R=1.0, G=0.0, B=0.0, A=1.0),
            "yellow": Color(R=1.0, G=1.0, B=0.0, A=1.0),
            "green": Color(R=0.0, G=1.0, B=0.0, A=1.0),
            "cyan": Color(R=0.0, G=1.0, B=1.0, A=1.0),
            "blue": Color(R=0.0, G=0.0, B=1.0, A=1.0),
            "magenta": Color(R=1.0, G=0.0, B=1.0, A=1.0),
            "white": Color(R=1.0, G=1.0, B=1.0, A=1.0),
            "black": Color(R=0.2, G=0.2, B=0.2, A=1.0),
            "grey": Color(R=0.5, G=0.5, B=0.5, A=1.0),
        }

    def compute_diffs(self, new_objects: list[Object]) -> list:
        diffs = []
        matched_objects: set[uuid.UUID] = (
            set()
        )  # old objects that were already matched to a new object
        for new_object in new_objects:
            # Skip objects without YOLO classification
            if "class" not in new_object.data:
                continue
            best_matching_object = None
            best_matching_confidence = -float("inf")

            for old_object in self.tracked_objects:
                if old_object.uid in matched_objects:
                    continue

                conf = self.compute_obj_diff(old_object.obj, new_object)
                if conf > best_matching_confidence:
                    best_matching_object = old_object
                    best_matching_confidence = conf

            if best_matching_object is not None and best_matching_confidence > 0.0:
                diffs.append(UpdateObjectDiff(self, best_matching_object, new_object))

                matched_objects.add(best_matching_object.uid)
            else:
                diffs.append(AddObjectDiff(self, new_object))

        for obj in self.tracked_objects:
            if obj.uid not in matched_objects:
                diffs.append(RemoveObjectDiff(self, obj))
        return diffs

    def compute_obj_diff(self, old_object: Object, new_object: Object) -> float:
        old_data, new_data = old_object.data, new_object.data
        old_keys, new_keys = set(old_data.keys()), set(new_data.keys())
        comparable_data = new_keys & old_keys
        if len(comparable_data) == 0:
            return 0.0

        total_similarity = 0
        total_weight = 0

        for key in comparable_data:
            comparator = self.comparators.get(key, AdditionalDataComparator(1.0))
            similarity = comparator.compute_similarity(old_data[key], new_data[key])

            total_similarity += comparator.weight * similarity
            total_weight += comparator.weight

        confidence = total_similarity / total_weight
        return confidence

    def object_to_tracked_object(self, obj: Object) -> TrackedObject:
        """Creates a TrackedObject from a RoboKudo object."""
        body = Body()

        semantic_annotations = list()

        if "semantic_color" in obj.data:
            color = self.semantic_color_to_rgb[obj.data["semantic_color"].color]
        else:
            color = Color(R=0.5, G=0.5, B=0.5, A=0.5)

        if "bbox" in obj.data:
            bb: BoundingBox3DAnnotation = obj.data["bbox"]

            map_pose = transform_pose_from_cam_to_world(self.cas(), bb.pose)

            translation = map_pose.translation
            rotation = map_pose.rotation

            origin = TransformationMatrix.from_xyz_quaternion(
                pos_x=translation[0],
                pos_y=translation[1],
                pos_z=translation[2],
                quat_x=rotation[0],
                quat_y=rotation[1],
                quat_z=rotation[2],
                quat_w=rotation[3],
            )

            scale = Scale(x=bb.x_length, y=bb.y_length, z=bb.z_length)

            body.collision.append(Box(color=color, origin=origin, scale=scale))

        if "class" in obj.data:
            semantic_annotation = self.class_to_semantic_annotation(obj.data["class"].classname, body=body)
            semantic_annotations.append(semantic_annotation)

        return TrackedObject(
            obj=obj,
            body=body,
            semantic_annotations=semantic_annotations,
            conns=[],
        )

    @staticmethod
    def class_to_semantic_annotation(class_name: str, **kwargs) -> SemanticAnnotation:
        """Convert a classification name to a SemanticWorld SemanticAnnotation."""
        available_semantic_annotations = recursive_subclasses(SemanticAnnotation)
        if len(available_semantic_annotations) == 0:
            raise ValueError("no semantic_annotations available for conversion from class name")

        class_candidates = [semantic_annotation_cls for semantic_annotation_cls in available_semantic_annotations if
                            semantic_annotation_cls.__name__ == class_name]

        for class_candidate in class_candidates:
            required_fields = set()
            optional_fields = set()
            for field in dataclasses.fields(class_candidate):
                if field.default == dataclasses.MISSING and field.default_factory == dataclasses.MISSING:
                    required_fields.add(field.name)
                else:
                    optional_fields.add(field.name)

            provided_fields = set(kwargs.keys())
            provided_optional_fields = provided_fields - required_fields

            # All required fields must be provided, all provided optional fields must be valid
            if all(req in provided_fields for req in required_fields) and all(
                    opt in optional_fields for opt in provided_optional_fields):
                return class_candidate(**kwargs)
        raise ValueError(
            f"could not convert class name {class_name} to semantic_annotation, candidates checked: {class_candidates}")

    @staticmethod
    def apply_diffs(diffs: list):
        for diff in diffs:
            diff.apply()
