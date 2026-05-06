# knowwind

**Semantic annotations for wind-turbine SCADA sensor data** — inspired by
[`semantic_digital_twin`](https://cram2.github.io/semantic_digital_twin/intro.html)
from the CRAM / AICOR ecosystem, re-targeted at time-series data from
wind-farm SCADA systems.

---

## Core idea

`semantic_digital_twin` attaches semantic meaning to 3-D bodies in a robotic
scene (e.g. *"this mesh is a Handle"*). `knowwind` does the same for sensor
columns in a SCADA CSV:

> *"column `wind_speed_3_avg` is a `WindSpeedSensor` measuring `m/s`"*
> *"column `sensor_11_avg` is a `GearboxTemperatureSensor` measuring `°C`"*

| semantic_digital_twin | knowwind |
|---|---|
| `World` | `WindTurbine` |
| `SemanticAnnotation` | `SensorAnnotation` |
| `world.add_semantic_annotation(…)` | `turbine.add_annotation(…)` |
| `world.get_annotations_by_type(Handle)` | `turbine.get_annotations_by_type(WindSpeedSensor)` |

---

## Quick start

```python
from knowwind.pipeline import load_feature_description, annotate_scada_dataframe
from knowwind.semantic_annotations import WindSpeedSensor

turbine = load_feature_description("feature_description.csv", asset_id="T01")
print(turbine.get_annotations_by_type(WindSpeedSensor))

import pandas as pd
df = pd.read_csv("event_001.csv", sep=";")
df = annotate_scada_dataframe(df, turbine)
```

## CLI demo

```bash
python -m knowwind.demo.annotate_wind_farm \
    --feature_desc WindFarmA/feature_description.csv \
    --event_csv    WindFarmA/datasets/event_001.csv \
    --asset_id     WF_A_T01
```

## Tests

```bash
PYTHONPATH=. pytest knowwind/tests/ -v
```
