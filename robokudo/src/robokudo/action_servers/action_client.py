from __future__ import annotations

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from typing_extensions import TYPE_CHECKING, Optional

from robokudo_msgs.action import Query

if TYPE_CHECKING:
    from rclpy.action.client import ClientGoalHandle
    from rclpy.task import Future
    from rclpy.timer import Timer
    from robokudo_msgs.action._query import Query_FeedbackMessage


class DebugActionClient(Node):
    """
    Debug Action Client for the Query Action Server.
    Allows sending dynamic goals and handles feedback, result, and cancellation.
    """

    def __init__(self):
        super().__init__("debug_action_client")
        self._action_client: ActionClient = ActionClient(
            self, Query, "/robokudo/query"
        )  # Connect to the action server
        self._goal_handle: Optional[ClientGoalHandle] = None
        self._cancel_timer: Optional[Timer] = None  # Re-enabled cancellation timer
        self.done: bool = False  # Flag to indicate completion

    def send_goal(self, goal_type: str) -> None:
        """Waits for the action server and sends a dynamic goal.

        :param goal_type: Type of the goal to send (e.g., 'human', 'robot', etc.)
        """
        self.get_logger().info("Waiting for action server...")
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Action server not available!")
            return

        # Create and send a goal
        goal_msg = Query.Goal()
        goal_msg.obj.type = goal_type

        self.get_logger().info(f"Sending goal request with type: '{goal_type}'")
        send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future: Future) -> None:
        """Handles the response from the action server regarding goal acceptance.

        :param future: The future of the async goal task returned by the action client upon sending a goal.
        """
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            self.get_logger().error("Goal rejected by the action server.")
            self.done = True
            return

        self.get_logger().info("Goal accepted by the action server.")

        # Scheduling of cancellation after 5 seconds
        self._cancel_timer = self.create_timer(5.0, self.cancel_goal)

        # Wait for the result
        result_future = self._goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg: Query_FeedbackMessage) -> None:
        """Processes feedback messages from the action server.

        :param feedback_msg: The feedback message returned by the action client.
        """
        feedback = feedback_msg.feedback
        self.get_logger().info(f"Received feedback: {feedback.feedback}")

    def cancel_goal(self) -> None:
        """Sends a cancel request for the active goal."""
        if not self._goal_handle:
            self.get_logger().error("No active goal to cancel.")
            return

        self.get_logger().info("Sending cancel request...")
        cancel_future = self._goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(self.cancel_done_callback)

        # Stop the cancel timer
        if self._cancel_timer:
            self._cancel_timer.cancel()

    def cancel_done_callback(self, future: Future) -> None:
        """Handles the response from the action server regarding goal cancellation.

        :param future: The future of the async goal task returned by the action client upon cancelling a goal.
        """
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info("Goal cancellation accepted by the server.")
        else:
            self.get_logger().warning("Goal cancellation was not successful.")
        self.done = True
        self.get_logger().info("Shutting down after cancellation is accepted.")
        rclpy.shutdown()

    def result_callback(self, future: Future) -> None:
        """Processes the result from the action server.

        :param future: The future of the async goal task returned by the action client upon getting the goal result.
        """
        try:
            result = future.result().result
            self.get_logger().info(f"Result received: {result}")
        except Exception as e:
            self.get_logger().error(f"Error receiving result: {e}")
        finally:
            self.get_logger().info("Shutting down after receiving the result.")
            self.done = True


def main(args=None):
    rclpy.init(args=args)
    action_client = DebugActionClient()

    try:
        # Accept user input for the goal dynamically
        goal_type = input("Enter goal type (e.g., 'human', 'robot', etc.): ")

        action_client.send_goal(goal_type)

        # Keep the node alive until the action is done
        while rclpy.ok() and not action_client.done:
            rclpy.spin_once(action_client, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        action_client.destroy_node()


if __name__ == "__main__":
    main()
