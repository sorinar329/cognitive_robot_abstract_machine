# Maintenance manuals and procedures

Openly reachable documents on how turbines like ours are maintained and inspected, and
where they touch the model. Manufacturer service manuals are normally confidential;
the two manufacturer documents below are copies that ended up in public planning
filings (they still carry the manufacturers' confidentiality notes, so we link them
rather than copy them).

## Documents

| Document | What it is | Useful for |
|---|---|---|
| [AWEA / ACP *Operations and Maintenance Recommended Practices*, 2nd ed. 2017](https://cleanpower.org/wp-content/uploads/2024/06/AWEA-Operations-and-Maintenance-Recommended-Practices-Second-Edition-2017.pdf) (454 pages, [overview](https://cleanpower.org/resources/awea-operations-maintenance-recommended-practices-second-edition/)) | Industry consensus procedures, organised by subsystem (RP 101-902) | The main reference for inspection and maintenance procedures and their intervals |
| [ACP RP 108 *Wear Debris Collection and Analysis for Wind Turbine Gearboxes*](https://cleanpower.org/wp-content/uploads/2024/06/AWEA_OM_RP_108-Wear-Debris-Collection-and-Analysis-for-Wind-Turbine-Gearboxes.pdf) | Gearbox oil debris procedure | Gearbox filter/oil faults, particle count signal |
| [ACP RP Chapter 7 *End of Warranty*](https://cleanpower.org/wp-content/uploads/2024/06/AWEA-OM-RP-Chapter-7_End-of-Warranty.pdf) and [Chapter 8 *Condition Based Maintenance*](https://cleanpower.org/wp-content/uploads/2024/06/AWEA-OM-RP-Chapter-8_Condition-Based-Maintenance.pdf) | Whole-turbine inspection lists; condition monitoring (vibration, grease/oil sampling, temperatures) | Inspection checklists; non-visual signals (vibration, temperature) |
| [ACP *Recommended Practices for Onshore Wind Turbine Foundation Maintenance*](https://cleanpower.org/resources/recommended-practices-for-onshore-wind-turbine-foundation-maintenance/) | Foundation inspection, grout, anchor bolts | Foundation zone (plinth crack, grout breakout, anchor nuts) |
| [GE Renewable Energy, *Wartungshandbuch 3MW-Plattform Onshore, Modul 3D – 3MW DFIG*, 2017 rev. 2](https://www.uvp-verbund.de/documents-ige-ng/igc_mv/EA910A3C-0A64-4308-B16B-4E63C4AE07E1/16_01_5b_Wartungspflichtenheft_3MW%20WEA.pdf) (German translation, 28 pages, from a German environmental-impact filing) | Manufacturer maintenance manual for the **electrical module** of a 3 MW DFIG turbine: converter cabinets, main cabinet (MCC), low-voltage distribution, safety chain, maintenance frequencies (6/12/48-month tasks) | Same size and generator type as our IEA 3.4 MW turbine. Contains a real maintenance schedule table ("Häufigkeit der Wartung": visual checks and system tests per cabinet and interval; e.g. coolant exchange every 5 years, pump motor bearings greased every 48 months, some 6-month tasks extendable to 12 months) |
| [Vestas *Mechanical Operating and Maintenance Manual V90-3.0 MW, VCRS 60 Hz (Mk 7)*, 2007](https://puc.sd.gov/commission/dockets/electric/2018/EL18-026/prefiledexhibits/fuerniss/9.pdf) (South Dakota PUC docket EL18-026) | Cover and chapter index of the mechanical manual (the chapters themselves are not in the filing), plus the complete 32-page *Safety Regulations for Operators and Technicians V90-3MW/V100-2.75MW* (2006): turbine inspection procedure, emergency stop locations, rotor lock and internal crane operation, rescue equipment | Chapter list of a real 3 MW geared turbine: rotor lock, blades, blade bearing, pitch, gearbox, brake, composite coupling, gear oil lubrication, generator and transformer, yaw gear, yaw bearing, hydraulics, wind sensors, air conditioning |
| [NREL *Wind Turbine Drivetrain Condition Monitoring – An Overview*](https://docs.nrel.gov/docs/fy12osti/50698.pdf) | Drivetrain failure modes and monitoring techniques | Gearbox and bearing faults and their signals |
| [NREL *Gearbox Reliability Collaborative: Gearbox 1 Failure Analysis*](https://docs.nrel.gov/docs/fy12osti/53062.pdf) | Documented gearbox damage from a test turbine | Realistic gearbox damage appearance |
| [Sandia / EPRI *Blade Visual Inspection and Maintenance Quantification* (2022)](https://www.sandia.gov/app/uploads/sites/273/2022/11/EPRI-Blade-Maintenance-Quantification-October19_2022-21.pdf) | How blade damage is found and classified in visual inspections | Blade faults (erosion, lightning, trailing edge) and their severity |
| [PNNL *O&M Best Practices for On-site Wind Turbines*](https://www.pnnl.gov/projects/om-best-practices/onsite-wind-turbines) | Short checklist-style O&M guide | Plain-language maintenance checklist |
| [Festo *Nacelle Operation and Maintenance* training manual](https://amtekcompany.com/doc/Festo%20Curriculum/festo-nacelle-operation-maintenance.pdf) | Technician training curriculum (drivetrain, gearbox basics, shaft alignment) | Background on how technicians work in the nacelle |
| [Wind Empowerment *Maintenance Manual for "Piggott" Small Wind Turbines* v3.2](https://windempowerment.org/wp-content/uploads/2020/05/Maintenance-manual-v3.2.1-web.pdf) | Complete open maintenance manual for a small turbine | Example of a full manual structure (different scale) |

## Where they meet the model

| Procedure (source) | Our parts / faults |
|---|---|
| RP 102 gear oil sampling, RP 106 filtration, RP 108 wear debris (ACP) | `gearbox.oil_level_low`, `gearbox.filter_clogged`, signal `gearbox_oil_particle_count` |
| RP 201 generator collector ring (slip ring) maintenance (ACP) | `generator.slip_ring_carbon_dust`, signal `generator_brush_wear_percent` |
| RP 202 grease-lubricated bearings, RP 812 main bearing grease sampling (ACP) | `main_bearing.grease_leak_front_seal`, `main_bearing.grease_collector_full` |
| RP 204 converter maintenance (ACP); converter cabinets (GE module 3D) | `converter.fault_light`, `controller.door_left_open` |
| RP 301 blades, RP 304 rotor lightning protection (ACP); Sandia/EPRI blade inspection | `blade_*.leading_edge_erosion`, `blade_*.lightning_damage`, `blade_*.trailing_edge_crack` |
| RP 401 foundation inspections and base bolt tensioning (ACP); foundation RP | `foundation.plinth_crack`, `foundation.grout_breakout`, `tower.anchor_nut_corrosion` |
| RP 402 fall protection and rescue, RP 404 elevators (ACP) | tower interior: ladders with fall-arrest rail, service lift (`service_lift_joint`); cable loop fault `tower.cable_loop_chafed` |
| RP 811 vibration analysis, RP 816 temperature measurement (ACP) | signals `gearbox_vibration_mm_s`, `generator_terminal_temperature_c`, `brake_disc_temperature_c` |
| Vestas chapters: manual rotor lock, brake system, composite coupling, hydraulic system | `rotor_lock.left_engaged`, `brake.*`, `coupling.disc_pack_cracked`, `hydraulic.hose_leak` |
| RP 302 rotor hubs, RP 814 pitch bearing grease (ACP); Vestas blade bearing, pitch system | hub zone (next): pitch bearing grease leak, pitch drive faults |
| Vestas yaw gear and yaw bearing system | yaw deck zone (next): yaw gear teeth, yaw brakes |

The ACP chapters also give intervals (e.g. gear oil sampling every 6 months) that can
become part of the scenarios later, for example "overdue" findings.

## Is there a maintenance plan?

Not for the whole turbine. Complete maintenance plans (task x interval for every
component) are manufacturer documents and are not public; the IEA reference turbine is a
design study and has none. What exists openly:

- a real interval table for the **electrical module** of a 3 MW DFIG turbine (GE, above);
- **procedures with intervals** per subsystem in the ACP recommended practices;
- the **inspection and safety procedures** of the Vestas V90 (above);
- the legal frame in Germany: a recurring inspection by an expert every 2 years (BWE
  principles), maintenance typically once or twice a year.

A maintenance plan for our model turbine can be compiled from these sources.

## Bolted joints (checking and re-tightening)

- **Anchor bolts** (ACP RP 401): tension check on a random 10% of the anchor bolts
  (every 10th bolt, counted clockwise from the bolt under the tower door) once a year in
  years 1-5, then on 20% of the turbines every 5 years. A single bolt below 85% of the
  specified tension, or an average below 90%, means re-tensioning all bolts of that
  tower. Corroded nuts count as a warning sign: they may be seized and no longer hold
  tension.
- **Construction and end of warranty** (ACP RP 901, RP 701): 10% checks for the tower
  base, blade-to-hub bolts and the turbine after erection; bolt torque tests in the
  end-of-warranty plan.
- **Electrical module** (GE 3 MW manual): check fasteners for movement "by the torque
  marking and/or" re-torquing; tightening torques for power cable terminals; bolted
  platform plates, railings and ladders; discoloured zinc coating at bolted joints.
- **Vestas V90 safety regulations**: "look very closely for oil spills and loose bolts
  ... Loose bolts in the structure mean danger. They must be tightened immediately." The
  manual rotor lock uses 16 M42 bolts, with the M16 bolts tightened to 70 Nm and then
  140 Nm in a circular sequence.

In the model: torque markings on the shrink disc bolts (`main_shaft.shrink_disc_bolt_loose`),
a missing gearbox cover bolt, corroded anchor nuts. The large joints need hydraulic
tensioners or torque wrenches of several hundred to thousands of Nm, well beyond what a
G1 arm can apply. So the robot's part is finding loose bolts by their markings and
bringing the tools.
