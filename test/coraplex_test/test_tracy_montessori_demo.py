"""
Tests for the scene of the Tracy Montessori demo.
"""

from __future__ import annotations

from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    ShapeSortingPiece,
)

# %% gripping a piece


def test_the_closed_pads_stay_clear_of_the_table_at_every_pieces_grasp(
    tracy_montessori_demo,
):
    """
    The grasp is aimed at where the closed fingers meet, so how far the pads reach past
    that point decides whether closing them presses them into the table.
    """
    scene = tracy_montessori_demo.demo.MontessoriScene()
    world = scene.build_world()
    [tracy] = world.get_semantic_annotations_by_type(Tracy)
    gripper = scene.sorting_arm(tracy).end_effector
    approach_axis = gripper.approach_axis.to_np()[:3]
    with world.reset_state_context():
        gripper.get_joint_state_by_type(GripperState.CLOSE).apply_to(world)
        pad_reach = max(
            float(corner.to_np()[:3] @ approach_axis)
            for finger in (gripper.thumb, gripper.finger)
            for body in finger.bodies
            if body.has_collision()
            for box in body.collision.as_bounding_box_collection_in_frame(
                gripper.tool_frame
            ).bounding_boxes
            for corner in box.get_points()
        )

    pad_ends = []
    for piece in world.get_semantic_annotations_by_type(ShapeSortingPiece):
        [grasp] = piece.grasp_candidates()
        world_T_tool = world.transform(
            gripper.tool_frame_goal_for_grasp(grasp.grasp_pose), world.root
        ).to_np()
        pad_ends.append(
            float(
                (
                    world_T_tool[:3, 3]
                    + pad_reach * world_T_tool[:3, :3] @ approach_axis
                )[2]
            )
        )

    assert min(pad_ends) > scene.table_top_z
