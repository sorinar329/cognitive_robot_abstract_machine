from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.reasoning.predicates import is_supported_by
from semantic_digital_twin.world import World

world = URDFParser.parse("")
world : World
bodies = world.bodies
milk = world.get_body_by_name("milk")
table = world.get_body_by_name("table")

is_supported_by(milk, table)
