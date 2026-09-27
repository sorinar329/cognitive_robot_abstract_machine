#!/usr/bin/env python3
"""Build the wind turbine knowledge representation (OWL 2) from the model.

  build_ontology.py            -> ontology/windturbine.ttl, windturbine.owl (RDF/XML, for Protégé),
                                  windturbine-aicor-alignment.ttl

TBox: components, faults, maintenance actions, robots, tasks, zones, with the
axioms a reasoner can use (partOf is transitive, defined classes such as
DrivetrainFault or G1InspectableFault). ABox: this turbine, generated from the
model's sources (turbine/parts: URDF links, faults, inspection points, signals;
turbine/bolts: flange bolt sets; the G1 tasks T0-T6), so it stays in step with
the model. The alignment module maps the main classes onto the AICOR L2 ontology
(which builds on DUL).

Needs rdflib (the cramera-port or cram venv has it).
"""
import os
import re
import sys

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.collection import Collection
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, XSD

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from turbine import bolts as tb  # noqa: E402
from turbine import dims as d  # noqa: E402
from turbine import parts  # noqa: E402
from turbine.dims import site  # noqa: E402

BASE = "http://www.semanticweb.org/windturbine-twin/ontology"
WT = Namespace(BASE + "#")
AICOR = Namespace("http://aicor.knowledge/l2-ontology.owl#")
OUT = os.path.join(ROOT, "ontology")

g = Graph()
g.bind("wt", WT)
g.bind("owl", OWL)
g.bind("dcterms", DCTERMS)


# %% helpers
def label(name):
    """CamelCase or snake_case -> words."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name).replace("_", " ")
    return text[0].upper() + text[1:] if text else text


def cls(name, parents=("Thing",), comment=None, lab=None, source=None):
    c = WT[name]
    g.add((c, RDF.type, OWL.Class))
    g.add((c, RDFS.label, Literal(lab or label(name), lang="en")))
    for p in parents:
        g.add((c, RDFS.subClassOf, OWL.Thing if p == "Thing" else WT[p]))
    if comment:
        g.add((c, RDFS.comment, Literal(comment, lang="en")))
    if source:
        g.add((c, DCTERMS.source, Literal(source)))
    return c


def seq(items):
    node = BNode()
    Collection(g, node, list(items))
    return node


def restriction(prop, kind, value, card=None):
    r = BNode()
    g.add((r, RDF.type, OWL.Restriction))
    g.add((r, OWL.onProperty, WT[prop]))
    if kind == "some":
        g.add((r, OWL.someValuesFrom, value))
    elif kind == "only":
        g.add((r, OWL.allValuesFrom, value))
    elif kind == "value":
        g.add((r, OWL.hasValue, value))
    elif kind in ("exactly", "min", "max"):
        pred = {"exactly": OWL.qualifiedCardinality, "min": OWL.minQualifiedCardinality,
                "max": OWL.maxQualifiedCardinality}[kind]
        g.add((r, pred, Literal(card, datatype=XSD.nonNegativeInteger)))
        g.add((r, OWL.onClass, value))
    return r


def subclass_of(name, expr):
    g.add((WT[name], RDFS.subClassOf, expr))


def intersection(*parts_):
    n = BNode()
    g.add((n, RDF.type, OWL.Class))
    g.add((n, OWL.intersectionOf, seq(parts_)))
    return n


def union(*parts_):
    n = BNode()
    g.add((n, RDF.type, OWL.Class))
    g.add((n, OWL.unionOf, seq(parts_)))
    return n


def equivalent(name, expr):
    g.add((WT[name], OWL.equivalentClass, expr))


def oprop(name, domain=None, range_=None, comment=None, inverse=None, chars=(), parent=None):
    p = WT[name]
    g.add((p, RDF.type, OWL.ObjectProperty))
    g.add((p, RDFS.label, Literal(label(name), lang="en")))
    if domain:
        g.add((p, RDFS.domain, WT[domain]))
    if range_:
        g.add((p, RDFS.range, WT[range_]))
    if comment:
        g.add((p, RDFS.comment, Literal(comment, lang="en")))
    if inverse:
        g.add((WT[inverse], RDF.type, OWL.ObjectProperty))
        g.add((WT[inverse], RDFS.label, Literal(label(inverse), lang="en")))
        g.add((p, OWL.inverseOf, WT[inverse]))
    for c in chars:
        g.add((p, RDF.type, {"transitive": OWL.TransitiveProperty, "functional": OWL.FunctionalProperty,
                             "asymmetric": OWL.AsymmetricProperty, "irreflexive": OWL.IrreflexiveProperty}[c]))
    if parent:
        g.add((p, RDFS.subPropertyOf, WT[parent]))
    return p


def dprop(name, domain=None, dtype=XSD.decimal, comment=None, functional=True):
    p = WT[name]
    g.add((p, RDF.type, OWL.DatatypeProperty))
    g.add((p, RDFS.label, Literal(label(name), lang="en")))
    if domain:
        g.add((p, RDFS.domain, WT[domain]))
    g.add((p, RDFS.range, dtype))
    if comment:
        g.add((p, RDFS.comment, Literal(comment, lang="en")))
    if functional:
        g.add((p, RDF.type, OWL.FunctionalProperty))
    return p


def ind(name, types, lab=None, comment=None):
    i = WT[name]
    g.add((i, RDF.type, OWL.NamedIndividual))
    for t in ([types] if isinstance(types, str) else types):
        g.add((i, RDF.type, WT[t]))
    g.add((i, RDFS.label, Literal(lab or label(name), lang="en")))
    if comment:
        g.add((i, RDFS.comment, Literal(comment, lang="en")))
    return i


def rel(s, p, o):
    g.add((WT[s], WT[p], WT[o]))


def val(s, p, v, dtype=None):
    if dtype is None:
        dtype = XSD.boolean if isinstance(v, bool) else XSD.integer if isinstance(v, int) else \
            XSD.decimal if isinstance(v, float) else XSD.string
    g.add((WT[s], WT[p], Literal(round(v, 4) if isinstance(v, float) else v, datatype=dtype)))


def all_disjoint(*names):
    n = BNode()
    g.add((n, RDF.type, OWL.AllDisjointClasses))
    g.add((n, OWL.members, seq(WT[x] for x in names)))


# %% ontology header
onto = URIRef(BASE)
g.add((onto, RDF.type, OWL.Ontology))
g.add((onto, OWL.versionInfo, Literal("1.0")))
g.add((onto, DCTERMS.title, Literal("Wind Turbine Twin ontology", lang="en")))
g.add((onto, DCTERMS.description, Literal(
    "Knowledge representation of an onshore wind turbine (IEA-3.4-130-RWT) for robot inspection and "
    "maintenance: components and their structure, faults and how they are observed, maintenance actions, "
    "access, and the Unitree G1 tasks. The individuals are generated from the Wind Turbine Twin model.", lang="en")))
g.add((onto, DCTERMS.source, Literal("IEA Wind TCP Task 37, IEA-3.4-130-RWT (NREL/TP-5000-73492, 2019)")))

# %% TBox: physical objects
cls("PhysicalObject", comment="Anything with a location in space: the turbine and its parts, robots, tools, people.")
cls("Artifact", ["PhysicalObject"], "A physical object made for a purpose.")
cls("WindTurbine", ["Artifact"], "A horizontal-axis wind turbine as a whole.")
cls("Assembly", ["Artifact"], "A major structural or functional group of a wind turbine.")
for a, c in (("Foundation", "Concrete foundation with plinth, grout joint and anchor bolts."),
             ("Tower", "Tubular steel tower, bolted from sections, with its interior."),
             ("Nacelle", "Machine housing on the tower top, turning about the yaw axis."),
             ("Rotor", "Hub with three blades."),
             ("Drivetrain", "Main shaft, main bearing, gearbox, high-speed shaft, brake, coupling and generator."),
             ("YawSystem", "Yaw bearing and drives turning the nacelle into the wind.")):
    cls(a, ["Assembly"], c)
    subclass_of(a, restriction("partOf", "some", WT.WindTurbine))
subclass_of("Drivetrain", restriction("partOf", "some", WT.Nacelle))
cls("TurbineComponent", ["Artifact"], "A part of a wind turbine that can be inspected, maintained or replaced.")
subclass_of("TurbineComponent", restriction("partOf", "some", WT.Assembly))
groups = {
    "StructuralComponent": ("Load-carrying structure.", {
        "FoundationPlinth": "Concrete pedestal of the foundation.", "GroutJoint": "Grout layer under the tower base flange.",
        "TowerSection": "One bolted steel tower section.", "YawBearing": "Slewing ring between tower and nacelle.",
        "Bedplate": "Nacelle main frame carrying the drivetrain.", "NacelleHousing": "Glass-fibre cover of the nacelle.",
        "Hub": "Cast hub carrying the blades.", "Blade": "Rotor blade, lofted from the IEA airfoils."}),
    "DrivetrainComponent": ("Part of the power train from hub to generator.", {
        "MainShaft": "Low-speed shaft from hub to gearbox.", "MainBearing": "Main shaft bearing with seals.",
        "GreaseCollector": "Container collecting spent bearing grease.", "RotorLock": "Mechanical lock of the rotor for work.",
        "RotorLockPin": "Pin of the rotor lock.", "Gearbox": "Three-stage gearbox, ratio 1:97.",
        "SightGlass": "Oil level sight glass.", "OilFilterIndicator": "Clogging indicator of the offline oil filter.",
        "TorqueArmBushing": "Elastomer bushing supporting the gearbox torque arm.", "HighSpeedShaft": "Shaft from gearbox to generator.",
        "BrakeDisc": "Disc of the mechanical rotor brake.", "BrakeCaliper": "Hydraulic brake caliper.",
        "FlexibleCoupling": "Disc-pack coupling between gearbox and generator.", "Generator": "Doubly-fed induction generator."}),
    "ElectricalComponent": ("Power conversion, control and grounding.", {
        "Transformer": "Cast-resin transformer.", "ConverterCabinet": "Power converter cabinet.",
        "ControllerCabinet": "Turbine controller cabinet.", "StatusLight": "Indicator light on a cabinet door.",
        "CableLoop": "Power cable loop under the yaw bearing, allowing the nacelle to yaw.",
        "GroundController": "Control cabinet on the tower ground floor.", "EarthingStrap": "Earthing connection from tower flange to foundation."}),
    "AuxiliarySystem": ("Supporting systems.", {
        "HydraulicUnit": "Hydraulic power unit (brake, yaw brakes).", "CoolingUnit": "Liquid cooling of generator and converter.",
        "CoolantLevelGauge": "Level tube of the coolant expansion tank.", "WeatherMast": "Mast on the nacelle roof.",
        "Anemometer": "Cup anemometer.", "AviationLight": "Aviation obstruction light."}),
    "HandlingEquipment": ("Equipment that lifts or carries loads and people.", {
        "ServiceCrane": "On-board crane under the nacelle roof.", "CraneRail": "Rail of the service crane.",
        "CraneTrolley": "Trolley of the service crane.", "CraneHook": "Hook block of the service crane, reaching the ground.",
        "LiftingPlatform": "Platform hung from the crane hook; hoists a robot or loads through the rear floor hatch.",
        "ServiceLift": "Service lift in the tower, ground floor to yaw deck.", "LiftGuide": "Guide and hoist wires of the service lift."}),
    "AccessStructure": ("Structure for moving around the turbine.", {
        "Platform": "Floor inside the tower.", "Ladder": "Ladder with fall-arrest rail.", "Door": "A door.",
        "Hatch": "A floor hatch."}),
    "SafetyEquipment": ("Equipment for the safety of people.", {
        "FireExtinguisher": "Portable fire extinguisher.", "SafetyKit": "First-aid kit and emergency equipment."}),
    "Storage": ("Where tools are kept.", {"ToolRack": "Rack for tools.", "ToolTray": "Tray to put tools on at the work place."}),
    "Fastener": ("Bolted connections.", {
        "BoltSet": "The bolts of one bolted joint."}),
}
for group, (comment, members) in groups.items():
    cls(group, ["TurbineComponent"], comment)
    for name, c in members.items():
        cls(name, [group], c)
cls("RestPlatform", ["Platform"], "Platform about 1.1 m below a flange joint, for bolt checks.")
cls("YawDeck", ["Platform"], "Top platform under the yaw bearing.")
cls("TowerDoor", ["Door"], "Entrance door at ground level.")
cls("CabinetDoor", ["Door"], "Door of a switch cabinet.")
cls("FloorHatch", ["Hatch"], "Hatch between tower and nacelle.")
cls("CraneHatch", ["Hatch"], "Rear floor hatch for the crane hoist.")
cls("FlangeBoltSet", ["BoltSet"], "Bolts of an internal tower flange joint, with torque markings.",
    source="ACP RP 401 (by analogy for the thresholds)")
cls("ShrinkDiscBoltSet", ["BoltSet"], "Bolts of the shrink disc between main shaft and gearbox.")
cls("CoverBoltSet", ["BoltSet"], "Bolts of the gearbox inspection cover.")
cls("AnchorBoltSet", ["BoltSet"], "Anchor bolts and nuts of the tower base flange.")
for name, whole in (("Gearbox", "Drivetrain"), ("Generator", "Drivetrain"), ("MainShaft", "Drivetrain"),
                    ("Blade", "Rotor"), ("Hub", "Rotor"), ("TowerSection", "Tower"), ("YawBearing", "YawSystem"),
                    ("FoundationPlinth", "Foundation")):
    subclass_of(name, restriction("partOf", "some", WT[whole]))
subclass_of("SightGlass", restriction("partOf", "some", WT.Gearbox))
subclass_of("TorqueArmBushing", restriction("partOf", "some", WT.Gearbox))
subclass_of("FlangeBoltSet", restriction("fastens", "min", WT.TowerSection, 2))
subclass_of("ServiceLift", restriction("providesAccessTo", "value", WT.zone_tower))
subclass_of("LiftingPlatform", restriction("providesAccessTo", "value", WT.zone_nacelle))

cls("Site", ["PhysicalObject"], "The turbine's surroundings.")
cls("Ground", ["Site"], "Terrain around the tower: grass, crane pad, road.")
cls("Tool", ["Artifact"], "A hand tool or tool set.")
cls("TorqueToolCase", ["Tool"], "Case with the torque tool for bolted joints; carried by its T-grip.")
cls("Agent", ["PhysicalObject"], "Something that acts: a robot or a person.")
cls("Robot", ["Agent", "Artifact"], "A robot.")
cls("HumanoidRobot", ["Robot"], "A robot with a human-like body.")
cls("UnitreeG1", ["HumanoidRobot"], "Unitree G1 humanoid (29 DoF, Dex3 hands, RealSense D435 head camera).")
cls("RobotPart", ["Artifact"], "A part of a robot.")
cls("RobotHand", ["RobotPart"], "An end effector with fingers.")
cls("Sensor", ["RobotPart"], "A device that measures something.")
cls("DepthCamera", ["Sensor"], "RGB-D camera.")
subclass_of("UnitreeG1", restriction("hasRobotPart", "exactly", WT.RobotHand, 2))   # simple property: cardinality is OWL 2 DL
subclass_of("UnitreeG1", restriction("hasRobotPart", "some", WT.DepthCamera))
cls("Person", ["Agent"], "A human.")
cls("Technician", ["Person"], "Service technician.")
all_disjoint("TurbineComponent", "Robot", "Person", "Tool", "Site")

# %% TBox: places, observation, signals
cls("Zone", comment="A region of the turbine where a robot works: outside, inside the tower, inside the nacelle.")
g.add((WT.Zone, OWL.equivalentClass, (lambda n: (g.add((n, RDF.type, OWL.Class)), g.add((n, OWL.oneOf, seq([WT.zone_outside, WT.zone_tower, WT.zone_nacelle]))), n)[-1])(BNode())))
cls("InspectionPoint", comment="A place on a component to look at, with the direction and distance a camera should look from.")
subclass_of("InspectionPoint", restriction("inspects", "exactly", WT.TurbineComponent, 1))
subclass_of("InspectionPoint", restriction("locatedInZone", "exactly", WT.Zone, 1))
cls("SensingModality", comment="How a condition can be observed.")
cls("ConditionSignal", comment="A measured value from the turbine's condition monitoring (SCADA / CMS).")
cls("Severity", comment="How urgent a fault is.")
g.add((WT.Severity, OWL.equivalentClass, (lambda n: (g.add((n, RDF.type, OWL.Class)), g.add((n, OWL.oneOf, seq([WT.severity_low, WT.severity_medium, WT.severity_high]))), n)[-1])(BNode())))

# %% TBox: faults
cls("Fault", comment="An abnormal condition of a turbine component that an inspection should find.")
subclass_of("Fault", restriction("affects", "some", WT.TurbineComponent))
subclass_of("Fault", restriction("hasSeverity", "exactly", WT.Severity, 1))
subclass_of("Fault", restriction("observableBy", "some", WT.SensingModality))
fault_tree = {
    "MaterialDegradation": ("The material itself is damaged or worn.", {
        "Crack": "A crack in concrete, metal, composite or elastomer.", "Corrosion": "Rust or fretting corrosion.",
        "Erosion": "Material worn away by impact (rain, dust), e.g. blade leading edges.",
        "Wear": "Wear of a consumable part, e.g. brake pads.", "Chafing": "Rubbing damage, e.g. cable insulation.",
        "CoatingDamage": "Damaged paint or coating.", "Breakout": "Material broken out, e.g. grout."}),
    "Leak": ("A fluid escapes where it should not.", {
        "OilLeak": "Gear oil leaking.", "GreaseLeak": "Bearing grease leaking.", "HydraulicLeak": "Hydraulic oil leaking."}),
    "FastenerFault": ("A bolted joint lost preload or a fastener is missing.", {
        "LooseFastener": "Nut or bolt turned back (torque marking offset): lost preload.",
        "MissingFastener": "A bolt is missing."}),
    "ThermalFault": ("Too hot.", {"Overheating": "Temperature above its limit."}),
    "FunctionalFault": ("Something does not work as it should.", {
        "FluidLevelLow": "Oil or coolant below the minimum mark.", "Clogging": "A filter is clogged.",
        "Contamination": "Dirt or dust where it harms, e.g. carbon dust at slip rings.",
        "FaultIndication": "A device reports a fault (red light, fault code).",
        "LightFailure": "A light does not work.", "Disconnection": "A connection is broken.",
        "MechanicalDamage": "A device is mechanically damaged.", "ServiceDue": "A consumable is full or used up."}),
    "ExternalDamage": ("Damage from outside.", {"LightningDamage": "Damage from a lightning strike."}),
    "UnsafeCondition": ("A configuration that is unsafe or not allowed in operation.", {
        "MissingSafetyEquipment": "Required safety equipment is not there.",
        "OpenEnclosure": "A cabinet or cover is left open.", "LockEngaged": "A lock is engaged that must be released."}),
}
for group, (comment, members) in fault_tree.items():
    cls(group, ["Fault"], comment)
    for name, c in members.items():
        cls(name, [group], c)
all_disjoint(*fault_tree)

# maintenance actions (repairs)
cls("MaintenanceAction", comment="What a technician does to remedy a fault.", source="ACP O&M Recommended Practices")
repairs = {
    "Retighten": "Bring single bolts back to the specified torque.",
    "Retension": "Re-tension all bolts of a joint (hydraulic tensioner or torque wrench), cross pattern, two passes.",
    "ReplaceFastener": "Replace bolt, nut and washers.", "Reseal": "Replace a seal.",
    "Refill": "Top up oil or coolant.", "ReplaceFilter": "Replace a filter element.",
    "Clean": "Clean a surface or component.", "EmptyCollector": "Empty a grease collector.",
    "ReplacePart": "Replace a worn or broken part.", "StructuralRepair": "Repair concrete, metal or composite structure.",
    "Recoat": "Remove corrosion and repaint.", "Reconnect": "Restore an electrical connection.",
    "RestoreConfiguration": "Close, release or put back what was left in the wrong state.",
    "Diagnose": "Read the fault code and diagnose the device.",
}
for name, c in repairs.items():
    cls(name, ["MaintenanceAction"], c)
remedies = {
    "LooseFastener": union(WT.Retighten, WT.Retension), "MissingFastener": WT.ReplaceFastener,
    "OilLeak": WT.Reseal, "GreaseLeak": WT.Reseal, "HydraulicLeak": WT.ReplacePart, "FluidLevelLow": WT.Refill,
    "Clogging": WT.ReplaceFilter, "Contamination": WT.Clean, "ServiceDue": WT.EmptyCollector, "Crack": WT.StructuralRepair,
    "Corrosion": WT.Recoat, "CoatingDamage": WT.Recoat, "Erosion": WT.StructuralRepair, "Wear": WT.ReplacePart,
    "Chafing": WT.ReplacePart, "Breakout": WT.StructuralRepair, "Overheating": WT.Diagnose, "FaultIndication": WT.Diagnose,
    "LightFailure": WT.ReplacePart, "Disconnection": WT.Reconnect, "MechanicalDamage": WT.ReplacePart,
    "LightningDamage": WT.StructuralRepair, "MissingSafetyEquipment": WT.RestoreConfiguration,
    "OpenEnclosure": WT.RestoreConfiguration, "LockEngaged": WT.RestoreConfiguration,
}
for fault_class, action in remedies.items():
    subclass_of(fault_class, restriction("remediedBy", "some", action))

# %% TBox: tasks
cls("RobotTask", comment="A task the robot executes as a CRAM plan.")
subclass_of("RobotTask", restriction("performedBy", "some", WT.Robot))
for name, c in (("InspectionRound", "Visit inspection points and report their state."),
                ("AssistanceTask", "Help a technician, e.g. by bringing tools."),
                ("AccessTask", "Get the robot to a zone it cannot walk to (lift, hoist).")):
    cls(name, ["RobotTask"], c)
cls("BoltCheck", ["InspectionRound"], "Read torque markings of a bolted joint and assess its preload.")
cls("CRAMAction", comment="An action designator of the CRAM stack (coraplex) used in the plans.")
cls("MaintenanceDocument", comment="A manual or recommended practice describing inspections or repairs.")

# %% TBox: defined classes (inferred by a reasoner)
cls("HighSeverityFault", ["Thing"], "A fault of high severity.")
equivalent("HighSeverityFault", intersection(WT.Fault, restriction("hasSeverity", "value", WT.severity_high)))
cls("DroneInspectableFault", ["Thing"], "A fault a drone camera can see (blades, outside of the nacelle).")
equivalent("DroneInspectableFault", intersection(WT.Fault, restriction("observableBy", "value", WT.modality_drone_rgb)))
cls("ThermallyDetectableFault", ["Thing"], "A fault a thermal camera can see.")
equivalent("ThermallyDetectableFault", intersection(WT.Fault, restriction("observableBy", "value", WT.modality_thermal)))
cls("NacelleFault", ["Thing"], "A fault observed inside the nacelle.")
equivalent("NacelleFault", intersection(WT.Fault, restriction("observedAt", "some",
                                                               restriction("locatedInZone", "value", WT.zone_nacelle))))
cls("DrivetrainFault", ["Thing"], "A fault on any part of the drivetrain (partOf is transitive).")
equivalent("DrivetrainFault", intersection(WT.Fault, restriction("affects", "some", restriction("partOf", "value", WT.drivetrain))))
cls("MonitoredFault", ["Thing"], "A fault that also shows in a condition monitoring signal.")
equivalent("MonitoredFault", intersection(WT.Fault, restriction("indicatedBySignal", "some", WT.ConditionSignal)))
cls("G1InspectableFault", ["Thing"], "A fault at an inspection point that a G1 task visits.")
equivalent("G1InspectableFault", intersection(WT.Fault, restriction("observedAt", "some",
                                                                     restriction("visitedBy", "some", WT.RobotTask))))

# %% properties
oprop("partOf", comment="Structural part-whole relation (transitive).", inverse="hasPart", chars=("transitive",))
g.add((WT.hasPart, RDF.type, OWL.TransitiveProperty))
oprop("mountedOn", "PhysicalObject", "PhysicalObject", "Kinematic parent in the model (URDF): what a part is attached to.")
oprop("hasRobotPart", "Robot", "RobotPart", "A hand, camera or other part of a robot (not transitive, so it can be counted).")
oprop("fastens", "BoltSet", "TurbineComponent", "The components a bolt set joins.")
oprop("locatedInZone", None, "Zone", "The zone something is in or reached from.", chars=("functional",))
oprop("providesAccessTo", "PhysicalObject", "Zone", "Equipment through which a zone is reached.")
oprop("inspects", "InspectionPoint", "TurbineComponent", "The component an inspection point looks at.",
      inverse="hasInspectionPoint", chars=("functional",))
oprop("affects", "Fault", "TurbineComponent", "The component a fault is on.", inverse="hasFault")
oprop("observedAt", "Fault", "InspectionPoint", "Where the fault is seen.")
oprop("observableBy", "Fault", "SensingModality", "How the fault can be observed.")
oprop("hasSeverity", "Fault", "Severity", "Severity of a fault.", chars=("functional",))
oprop("indicatedBySignal", "Fault", "ConditionSignal", "A condition monitoring signal that shows the fault.")
oprop("remediedBy", "Fault", "MaintenanceAction", "What remedies the fault.")
oprop("describedIn", None, "MaintenanceDocument", "Where a procedure or threshold is described.")
oprop("performedBy", "RobotTask", "Agent", "Who executes a task.", inverse="performs")
oprop("visits", "RobotTask", "InspectionPoint", "Inspection points a task looks at.", inverse="visitedBy")
oprop("takesPlaceIn", "RobotTask", "Zone", "Zones a task works in.")
oprop("usesAction", "RobotTask", "CRAMAction", "CRAM actions in the task's plan.")
oprop("rides", "RobotTask", "HandlingEquipment", "Equipment the robot rides in a task.")
oprop("carries", "RobotTask", "Tool", "What the robot carries.")
oprop("assists", "RobotTask", "Person", "Who the task helps.")

dprop("hubHeight", "WindTurbine", comment="Hub height above ground (m).")
dprop("rotorDiameter", "WindTurbine", comment="Rotor diameter (m).")
dprop("ratedPower", "WindTurbine", comment="Rated power (MW).")
dprop("towerHeight", "WindTurbine", comment="Tower height (m).")
dprop("heightAboveGround", "PhysicalObject", comment="Height above ground (m).")
dprop("mass", "PhysicalObject", comment="Mass (kg).")
dprop("urdfLink", "PhysicalObject", XSD.string, "Name of the link in the model's URDF.")
dprop("boltSize", "BoltSet", XSD.string, "Thread size, e.g. M42 (property class 10.9).")
dprop("boltCount", "BoltSet", XSD.integer, "Number of bolts.")
dprop("preload", "BoltSet", comment="Specified preload per bolt (kN), F_p,C = 0.7 f_ub A_s.")
dprop("tighteningTorque", "BoltSet", comment="Tightening torque (Nm), torque coefficient 0.13.")
dprop("turnToPreload", "BoltSet", comment="Nut turn from snug to full preload (deg); turning back by it releases all preload.")
dprop("faultId", "Fault", XSD.string, "Identifier of the fault in the model's scenarios.")
dprop("severityRank", "Severity", XSD.integer, "1 low, 2 medium, 3 high.")
dprop("nominalValue", "ConditionSignal", comment="Value of the healthy turbine.")
dprop("unit", "ConditionSignal", XSD.string, "Unit of the signal.")
dprop("viewDistance", "InspectionPoint", comment="Camera distance to look from (m).")
dprop("travel", "HandlingEquipment", comment="Travel of a lift or hoist (m).")
dprop("speed", "HandlingEquipment", comment="Speed in the simulation (m/s).")
dprop("cameraHeight", "Robot", comment="Camera height above the floor (m).")
dprop("scriptPath", "RobotTask", XSD.string, "The CRAM plan script in the model repository.")
dprop("crameraScene", "RobotTask", XSD.string, "Name of the CRAMERA recording.")

# %% ABox: fixed individuals
for z, c in (("zone_outside", "Around the tower base and outside the nacelle (ground robot or drone)."),
             ("zone_tower", "Inside the tower: ground floor, rest platforms, yaw deck."),
             ("zone_nacelle", "Inside the nacelle: walkways along the drivetrain.")):
    ind(z, "Zone", label(z.split("_")[1]), c)
for rank, s in enumerate(("low", "medium", "high"), start=1):
    ind(f"severity_{s}", "Severity", s)
    val(f"severity_{s}", "severityRank", rank)
diff = BNode()
g.add((diff, RDF.type, OWL.AllDifferent))
g.add((diff, OWL.distinctMembers, seq([WT.zone_outside, WT.zone_tower, WT.zone_nacelle])))
diff2 = BNode()
g.add((diff2, RDF.type, OWL.AllDifferent))
g.add((diff2, OWL.distinctMembers, seq([WT.severity_low, WT.severity_medium, WT.severity_high])))
MODALITIES = {"rgb": "RGB camera on the robot", "drone_rgb": "RGB camera on a drone", "thermal": "Thermal camera",
              "vibration": "Vibration measurement (CMS)"}
for m, c in MODALITIES.items():
    ind(f"modality_{m}", "SensingModality", c)

ind("iea_3_4_130_rwt", "WindTurbine", "IEA-3.4-130-RWT", "The modelled land-based reference turbine.")
val("iea_3_4_130_rwt", "hubHeight", 110.0)
val("iea_3_4_130_rwt", "rotorDiameter", 130.0)
val("iea_3_4_130_rwt", "ratedPower", 3.37)
val("iea_3_4_130_rwt", "towerHeight", float(site.TOWER_HEIGHT))
ASSEMBLIES = {"foundation_system": "Foundation", "tower": "Tower", "nacelle_assembly": "Nacelle", "rotor": "Rotor",
              "drivetrain": "Drivetrain", "yaw_system": "YawSystem"}
for name, c in ASSEMBLIES.items():
    ind(name, c, label(c))
    rel(name, "partOf", "iea_3_4_130_rwt")
rel("drivetrain", "partOf", "nacelle_assembly")

# %% ABox: components from the URDF links
LINK_CLASS = {
    "ground": "Ground", "foundation": "FoundationPlinth", "foundation_grout": "GroutJoint", "foundation_earthing": "EarthingStrap",
    "tower_door": "TowerDoor", "yaw_bearing": "YawBearing", "cable_loop": "CableLoop", "ground_controller": "GroundController",
    "lift_rails": "LiftGuide", "service_lift": "ServiceLift", "nacelle": "NacelleHousing", "nacelle_floor_hatch": "FloorHatch",
    "crane_hatch": "CraneHatch", "weather_mast": "WeatherMast", "anemometer": "Anemometer", "aviation_light": "AviationLight",
    "bedplate": "Bedplate", "main_bearing": "MainBearing", "main_bearing_grease_collector": "GreaseCollector",
    "main_shaft": "MainShaft", "main_shaft_shrink_disc_bolts": "ShrinkDiscBoltSet", "rotor_lock": "RotorLock",
    "rotor_lock_pin": "RotorLockPin", "hub": "Hub", "gearbox": "Gearbox", "gearbox_sight_glass": "SightGlass",
    "gearbox_filter_indicator": "OilFilterIndicator", "gearbox_cover_bolts": "CoverBoltSet", "gearbox_bushing_left": "TorqueArmBushing",
    "fast_shaft": "HighSpeedShaft", "fast_shaft_brake_disc": "BrakeDisc", "fast_shaft_coupling": "FlexibleCoupling",
    "brake_caliper": "BrakeCaliper", "generator": "Generator", "transformer": "Transformer",
    "cabinet_converter": "ConverterCabinet", "cabinet_converter_door": "CabinetDoor", "converter_status_light": "StatusLight",
    "cabinet_controller": "ControllerCabinet", "cabinet_controller_door": "CabinetDoor", "hydraulic_unit": "HydraulicUnit",
    "cooling_unit": "CoolingUnit", "cooling_level": "CoolantLevelGauge", "crane_rail": "CraneRail", "crane_trolley": "CraneTrolley",
    "crane_hook": "CraneHook", "lifting_platform": "LiftingPlatform", "fire_extinguisher": "FireExtinguisher",
    "safety_equipment": "SafetyKit", "tool_rack": "ToolRack", "tool_tray": "ToolTray",
}
for k in range(1, 5):
    LINK_CLASS[f"tower_section_{k}"] = "TowerSection"
    LINK_CLASS[f"ladder_{k}"] = "Ladder"
    LINK_CLASS[f"platform_{k}"] = "YawDeck" if k == 4 else "RestPlatform"
for k in range(1, 4):
    LINK_CLASS[f"blade_{k}"] = "Blade"
    LINK_CLASS[f"flange_{k}_bolts"] = "FlangeBoltSet"
MODULE_ASSEMBLY = {"site": "foundation_system", "tower_interior": "tower", "nacelle": "nacelle_assembly",
                   "bedplate": "nacelle_assembly", "main_shaft": "drivetrain", "rotor": "rotor", "gearbox": "drivetrain",
                   "generator": "drivetrain", "systems": "nacelle_assembly"}
PART_OVERRIDE = {   # finer part-of than the assembly of the module
    "tower_section_1": "tower", "tower_section_2": "tower", "tower_section_3": "tower", "tower_section_4": "tower",
    "tower_door": "tower_section_1", "yaw_bearing": "yaw_system", "ground": None, "drivetrain": None,
    "gearbox_sight_glass": "gearbox", "gearbox_filter_indicator": "gearbox", "gearbox_cover_bolts": "gearbox",
    "gearbox_bushing_left": "gearbox", "main_bearing_grease_collector": "main_bearing", "main_shaft_shrink_disc_bolts": "main_shaft",
    "rotor_lock_pin": "rotor_lock", "fast_shaft_brake_disc": "fast_shaft", "fast_shaft_coupling": "fast_shaft",
    "cabinet_converter_door": "cabinet_converter", "converter_status_light": "cabinet_converter_door",
    "cabinet_controller_door": "cabinet_controller", "cooling_level": "cooling_unit", "anemometer": "weather_mast",
    "aviation_light": "weather_mast", "crane_rail": "service_crane", "crane_trolley": "service_crane", "crane_hook": "service_crane",
    "lifting_platform": "service_crane", "lift_rails": "service_lift", "nacelle": "nacelle_assembly",
}
ind("service_crane", "ServiceCrane", "Service crane")
rel("service_crane", "partOf", "nacelle_assembly")
ind("anchor_bolts", "AnchorBoltSet", "Anchor bolts", f"{site.ANCHOR_NUTS} anchor bolts on the base flange.")
rel("anchor_bolts", "partOf", "foundation_system")
val("anchor_bolts", "boltCount", site.ANCHOR_NUTS)
rel("anchor_bolts", "fastens", "tower_section_1")
rel("anchor_bolts", "fastens", "foundation")
for module in parts.ALL:
    mod = module.__name__.split(".")[-1]
    for link in module.LINKS:
        name = link["name"]
        if name == "drivetrain":
            val("drivetrain", "urdfLink", name)
            continue
        ind(name, LINK_CLASS[name], label(name))
        val(name, "urdfLink", name)
        whole = PART_OVERRIDE.get(name, MODULE_ASSEMBLY[mod])
        if whole:
            rel(name, "partOf", whole)
        if link["parent"] not in ("world",):
            rel(name, "mountedOn", link["parent"])
for k in range(1, 4):
    rel(f"blade_{k}", "partOf", "rotor")
    rel(f"flange_{k}_bolts", "fastens", f"tower_section_{k}")
    rel(f"flange_{k}_bolts", "fastens", f"tower_section_{k + 1}")
rel("hub", "partOf", "rotor")
for k, z in enumerate(site.platform_heights(), start=1):
    val(f"platform_{k}", "heightAboveGround", round(site.TOWER_BASE_Z + z, 2))
    rel(f"platform_{k}", "locatedInZone", "zone_tower")
val("service_lift", "travel", round(site.LIFT_TRAVEL, 2))
val("service_lift", "speed", site.LIFT_SPEED)
g.add((WT.service_lift, WT.providesAccessTo, WT.zone_tower))
g.add((WT.lifting_platform, WT.providesAccessTo, WT.zone_nacelle))
g.add((WT.tower_door, WT.providesAccessTo, WT.zone_tower))
val("crane_hook", "travel", 115.0)
val("crane_hook", "speed", d.HOOK_SPEED)
for k in range(1, 4):
    b = tb.spec(site.FLANGE_BOLTS[k - 1][0])
    name = f"flange_{k}_bolts"
    val(name, "boltSize", b["size"] + " 10.9")
    val(name, "boltCount", site.FLANGE_BOLTS[k - 1][1])
    val(name, "preload", round(b["preload"] / 1e3, 1))
    val(name, "tighteningTorque", round(b["torque_nm"]))
    val(name, "turnToPreload", round(b["turn_to_preload_deg"], 1))
    val(name, "heightAboveGround", round(site.TOWER_BASE_Z + site.TOWER_FLANGES[k - 1], 2))

# %% ABox: inspection points, signals, faults
point_zone = {}


def component_of(point):
    """The component a point looks at: its URDF parent, or for points on an assembly frame
    (the drivetrain link) the component its name starts with (main_bearing_inspect_... -> main_bearing)."""
    if point["parent"] in LINK_CLASS:
        return point["parent"]
    stem = point["name"].split("_inspect")[0]
    return next((l for l in sorted(LINK_CLASS, key=len, reverse=True) if stem.startswith(l)), "nacelle")


for module in parts.ALL:
    for p in module.INSPECTION_POINTS:
        name = p["name"]
        ind(name, "InspectionPoint", label(name.replace("_inspect_", ": ")), p["what"])
        rel(name, "inspects", component_of(p))
        rel(name, "locatedInZone", f"zone_{p['zone']}")
        val(name, "viewDistance", float(p["distance"]))
        point_zone[name] = p["zone"]

UNITS = [("_ohm", "Ω"), ("_c", "°C"), ("_mm_s", "mm/s"), ("_percent", "%"), ("_bar", "bar"), ("_counter", "count"),
         ("_count", "count"), ("_code", "code"), ("_level", "fraction of nominal"), ("_plausible", "boolean"), ("_on", "boolean")]
signals = {}
for module in parts.ALL:
    signals.update(getattr(module, "SIGNALS", {}))
for s, v in signals.items():
    ind(f"signal_{s}", "ConditionSignal", label(s), "Condition monitoring signal.")
    if isinstance(v, bool):
        g.add((WT[f"signal_{s}"], WT.nominalValue, Literal(1.0 if v else 0.0, datatype=XSD.decimal)))
    else:
        val(f"signal_{s}", "nominalValue", float(v))
    val(f"signal_{s}", "unit", next((u for suffix, u in UNITS if s.endswith(suffix)), "-"))

FAULT_CLASS = [   # (substring of the fault id, class)
    ("crack", "Crack"), ("breakout", "Breakout"), ("disconnected", "Disconnection"), ("corrosion", "Corrosion"),
    ("coating", "CoatingDamage"), ("chafed", "Chafing"), ("bolts_loose", "LooseFastener"), ("bolt_loose", "LooseFastener"),
    ("anemometer_damaged", "MechanicalDamage"), ("light_failed", "LightFailure"), ("grease_leak", "GreaseLeak"),
    ("collector_full", "ServiceDue"), ("left_engaged", "LockEngaged"), ("erosion", "Erosion"),
    ("lightning", "LightningDamage"), ("oil_leak", "OilLeak"), ("level_low", "FluidLevelLow"), ("clogged", "Clogging"),
    ("bolt_missing", "MissingFastener"), ("pads_worn", "Wear"), ("overheat", "Overheating"), ("carbon_dust", "Contamination"),
    ("fault_light", "FaultIndication"), ("door_left_open", "OpenEnclosure"), ("hose_leak", "HydraulicLeak"),
    ("coolant_low", "FluidLevelLow"), ("extinguisher_missing", "MissingSafetyEquipment"),
]
FAULT_TARGET = {"tower.anchor_nut_corrosion": "anchor_bolts", "main_shaft.shrink_disc_bolt_loose": "main_shaft_shrink_disc_bolts",
                "gearbox.cover_bolt_missing": "gearbox_cover_bolts", "main_bearing.grease_collector_full": "main_bearing_grease_collector",
                "gearbox.oil_level_low": "gearbox_sight_glass", "brake.pads_worn": "brake_caliper",
                "converter.fault_light": "converter_status_light", "controller.door_left_open": "cabinet_controller_door",
                "cooling.coolant_low": "cooling_level", "tower.flange_2_bolts_loose": "flange_2_bolts",
                "tower.flange_3_bolts_loose": "flange_3_bolts", "tower.cable_loop_chafed": "cable_loop",
                "nacelle.anemometer_damaged": "anemometer", "nacelle.aviation_light_failed": "aviation_light"}
points_by_name = {p["name"]: p for m in parts.ALL for p in m.INSPECTION_POINTS}
for module in parts.ALL:
    for fid, f in module.FAULTS.items():
        name = "fault_" + fid.replace(".", "_")
        klass = next(c for key, c in FAULT_CLASS if key in fid)
        ind(name, klass, f["component"] + ": " + label(fid.split(".")[1]).lower(), f["description"])
        val(name, "faultId", fid)
        point = points_by_name[f["inspection_point"]]
        target = FAULT_TARGET.get(fid, component_of(point))
        rel(name, "affects", target)
        rel(name, "observedAt", f["inspection_point"])
        rel(name, "hasSeverity", f"severity_{f['severity']}")
        for m in f["observable_by"]:
            rel(name, "observableBy", f"modality_{m}")
        for s in f["signals"]:
            if s in signals:
                rel(name, "indicatedBySignal", f"signal_{s}")

# %% ABox: robot, people, tools, CRAM actions, documents, tasks
ind("g1", "UnitreeG1", "Unitree G1", "The humanoid of the CRAM stack (OFFIS description).")
val("g1", "mass", 34.8)
val("g1", "cameraHeight", 1.27)
for side in ("left", "right"):
    ind(f"g1_{side}_hand", "RobotHand", f"G1 {side} Dex3 hand", "Three-finger Dex3 hand; fingers 1.4 Nm, thumb rotation 2.45 Nm.")
    rel("g1", "hasRobotPart", f"g1_{side}_hand")
g.add((WT.g1_left_hand, OWL.differentFrom, WT.g1_right_hand))
ind("g1_d435", "DepthCamera", "G1 RealSense D435", "Head camera, optical axis 48° down.")
rel("g1", "hasRobotPart", "g1_d435")
ind("technician", "Technician", "Technician")
ind("torque_tool_case", "TorqueToolCase", "Torque tool case", "Case with T-grip; 3 kg.")
val("torque_tool_case", "mass", d.TOOL_CASE_MASS)
for a, c in (("NavigateAction", "Drive the base to a pose (route planning)."), ("LookAtAction", "Point the camera at a target."),
             ("PickUpAction", "Grasp and lift an object."), ("PlaceAction", "Put an object down."),
             ("ParkArmsAction", "Move the arms to the park pose."), ("StraightMove", "Straight base move without route planning."),
             ("RideCarrier", "Stand on a lift car or platform while its joint moves.")):
    ind("action_" + a, "CRAMAction", a, c)
for doc, c in (("acp_rp_401", "ACP RP 401: foundation inspections and base bolt tensioning."),
               ("acp_om_recommended_practices", "AWEA/ACP Operations and Maintenance Recommended Practices, 2nd ed. 2017."),
               ("ge_3mw_module_3d", "GE 3 MW maintenance manual, module 3D (electrical)."),
               ("vestas_v90_safety", "Vestas V90-3 MW safety regulations for operators and technicians.")):
    ind(doc, "MaintenanceDocument", c.split(":")[0].split(",")[0], c)
subclass_of("FastenerFault", restriction("describedIn", "value", WT.acp_rp_401))
subclass_of("Leak", restriction("describedIn", "value", WT.acp_om_recommended_practices))
subclass_of("FaultIndication", restriction("describedIn", "value", WT.ge_3mw_module_3d))

TASKS = {
    "task_t0": ("InspectionRound", "T0: look at the tower door", ["tower_inspect_door"], ["zone_outside"],
                ["NavigateAction", "LookAtAction"], "scripts/g1_inspection_round.py", "windturbine_g1_t1"),
    "task_t1": ("InspectionRound", "T1: tower base round", ["tower_inspect_base_coating", "foundation_inspect_plinth",
                "foundation_inspect_grout", "foundation_inspect_earthing", "tower_inspect_anchor_nuts", "tower_inspect_door"],
                ["zone_outside"], ["NavigateAction", "LookAtAction", "ParkArmsAction"], "scripts/g1_inspection_round.py", "windturbine_g1_t1"),
    "task_t2": ("InspectionRound", "T2: nacelle walkway round", ["main_shaft_inspect_shrink_disc", "main_bearing_inspect_grease_collector",
                "gearbox_inspect_input_seal", "gearbox_inspect_bushing_left", "gearbox_inspect_sight_glass", "generator_inspect_slip_ring"],
                ["zone_nacelle"], ["StraightMove", "LookAtAction", "ParkArmsAction"], "scripts/g1_inspection_round.py", "windturbine_g1_t2"),
    "task_t3": ("AssistanceTask", "T3: find the loose bolt, bring the torque tool", ["main_shaft_inspect_shrink_disc"], ["zone_nacelle"],
                ["StraightMove", "LookAtAction", "PickUpAction", "PlaceAction", "ParkArmsAction"], "scripts/g1_bring_the_tool.py", "windturbine_g1_t3"),
    "task_t4": ("AccessTask", "T4: crane hoist into the nacelle", ["controller_inspect_door"], ["zone_outside", "zone_nacelle"],
                ["StraightMove", "RideCarrier", "LookAtAction"], "scripts/g1_climb.py", "windturbine_g1_t4"),
    "task_t5": ("AccessTask", "T5: tower lift to the yaw deck", ["tower_inspect_cable_loop"], ["zone_outside", "zone_tower"],
                ["StraightMove", "RideCarrier", "LookAtAction"], "scripts/g1_climb.py", "windturbine_g1_t5"),
    "task_t6": ("BoltCheck", "T6: flange bolt check", ["tower_inspect_flange_1", "tower_inspect_flange_2", "tower_inspect_flange_3"],
                ["zone_outside", "zone_tower"], ["StraightMove", "RideCarrier", "LookAtAction"], "scripts/g1_tower_bolts.py", "windturbine_g1_t6"),
}
for name, (klass, lab, points, zones, actions, script, scene) in TASKS.items():
    ind(name, klass, lab)
    rel(name, "performedBy", "g1")
    for p in points:
        rel(name, "visits", p)
    for z in zones:
        rel(name, "takesPlaceIn", z)
    for a in actions:
        rel(name, "usesAction", "action_" + a)
    val(name, "scriptPath", script)
    val(name, "crameraScene", scene)
rel("task_t3", "carries", "torque_tool_case")
rel("task_t3", "assists", "technician")
rel("task_t4", "rides", "lifting_platform")
rel("task_t5", "rides", "service_lift")
rel("task_t6", "rides", "service_lift")

# %% write
os.makedirs(OUT, exist_ok=True)
g.serialize(os.path.join(OUT, "windturbine.ttl"), format="turtle")
g.serialize(os.path.join(OUT, "windturbine.owl"), format="xml")          # plain RDF/XML keeps every list
check = Graph().parse(os.path.join(OUT, "windturbine.owl"), format="xml")
from rdflib.compare import isomorphic  # noqa: E402
assert isomorphic(check, g), "RDF/XML does not round-trip"

al = Graph()
al.bind("wt", WT)
al.bind("aicor", AICOR)
al.bind("owl", OWL)
alignment = URIRef(BASE + "/aicor-alignment")
al.add((alignment, RDF.type, OWL.Ontology))
al.add((alignment, OWL.imports, onto))
al.add((alignment, OWL.imports, URIRef("http://aicor.knowledge/l2-ontology.owl")))
al.add((alignment, RDFS.comment, Literal("Maps the Wind Turbine Twin ontology onto the AICOR L2 ontology (DUL-based).", lang="en")))
for ours, theirs in (("PhysicalObject", "PhysicalEntity"), ("Robot", "Robot"), ("Sensor", "Sensor"),
                     ("RobotTask", "Task"), ("MaintenanceAction", "PhysicalRepair"), ("ConditionSignal", "SensorOutput")):
    al.add((WT[ours], RDFS.subClassOf, AICOR[theirs]))
al.serialize(os.path.join(OUT, "windturbine-aicor-alignment.ttl"), format="turtle")

n_cls = len(set(g.subjects(RDF.type, OWL.Class)) - {s for s in g.subjects(RDF.type, OWL.Class) if isinstance(s, BNode)})
print(f"wrote {OUT}: {len(g)} triples, {n_cls} classes, {len(set(g.subjects(RDF.type, OWL.ObjectProperty)))} object properties, "
      f"{len(set(g.subjects(RDF.type, OWL.DatatypeProperty)))} data properties, {len(set(g.subjects(RDF.type, OWL.NamedIndividual)))} individuals")
