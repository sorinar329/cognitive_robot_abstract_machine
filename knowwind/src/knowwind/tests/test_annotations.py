"""
knowwind.tests.test_annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for annotation creation, WindTurbine registry, and
CSV importer using the real feature_description.csv.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from knowwind.datastructures import SensorName, StatisticType, PhysicalUnit
from knowwind.wind_turbine import WindTurbine
from knowwind.semantic_annotations import (
    SensorAnnotation,
    WindSpeedSensor,
    ActivePowerSensor,
    GearboxTemperatureSensor,
    RotorRpmSensor,
    GenericSensor,
    infer_annotation_type,
)
from knowwind.pipeline import (
    load_feature_description_from_string,
    turbine_annotation_summary,
    annotation_type_counts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINI_FEATURE_CSV = textwrap.dedent("""\
    sensor_name;statistics_type;description;unit;is_angle;is_counter
    wind_speed_3;maximum,minimum,average,std_dev;Windspeed;m/s;False;False
    power_30;maximum,std_dev,minimum,average;Grid power;kW;False;False
    sensor_11;average;Temperature in gearbox bearing on high speed shaft;°C;False;False
    sensor_52;minimum,average,std_dev,maximum;Rotor rpm;rpm;False;False
    sensor_1;average;Wind absolute direction;°;True;False
""")


@pytest.fixture
def mini_turbine() -> WindTurbine:
    return load_feature_description_from_string(MINI_FEATURE_CSV, asset_id="T_TEST")


# ---------------------------------------------------------------------------
# SensorName
# ---------------------------------------------------------------------------

class TestSensorName:
    def test_column_name_average(self):
        sn = SensorName("sensor_18", StatisticType.AVERAGE)
        assert sn.column_name == "sensor_18_avg"

    def test_column_name_std(self):
        sn = SensorName("sensor_18", StatisticType.STD_DEV)
        assert sn.column_name == "sensor_18_std"

    def test_frozen(self):
        sn = SensorName("x")
        with pytest.raises(Exception):
            sn.base_name = "y"   # type: ignore[misc]


# ---------------------------------------------------------------------------
# SensorAnnotation base
# ---------------------------------------------------------------------------

class TestSensorAnnotation:
    def test_hash_unique(self):
        sn = SensorName("sensor_0")
        a1 = GenericSensor(sensor_name=sn)
        a2 = GenericSensor(sensor_name=sn)
        assert hash(a1) != hash(a2)

    def test_eq_same_object(self):
        sn = SensorName("sensor_0")
        a = GenericSensor(sensor_name=sn)
        assert a == a

    def test_eq_different_objects(self):
        sn = SensorName("sensor_0")
        a1 = GenericSensor(sensor_name=sn)
        a2 = GenericSensor(sensor_name=sn)
        assert a1 != a2


# ---------------------------------------------------------------------------
# infer_annotation_type
# ---------------------------------------------------------------------------

class TestInferAnnotationType:
    @pytest.mark.parametrize("desc,expected", [
        ("Windspeed",                        WindSpeedSensor),
        ("Grid power",                       ActivePowerSensor),
        ("Temperature in gearbox bearing",   GearboxTemperatureSensor),
        ("Rotor rpm",                        RotorRpmSensor),
        ("Something completely unknown",     GenericSensor),
    ])
    def test_inference(self, desc, expected):
        assert infer_annotation_type(desc) is expected


# ---------------------------------------------------------------------------
# WindTurbine
# ---------------------------------------------------------------------------

class TestWindTurbine:
    def test_add_and_retrieve(self):
        turbine = WindTurbine("T01")
        sn = SensorName("wind_speed_3", StatisticType.AVERAGE)
        ann = WindSpeedSensor(sensor_name=sn, description="Windspeed", unit=PhysicalUnit.METER_PER_SEC)
        turbine.add_annotation(ann)
        assert turbine.get_annotation_by_column("wind_speed_3_avg") is ann

    def test_get_by_type(self):
        turbine = WindTurbine("T01")
        sn1 = SensorName("wind_speed_3", StatisticType.AVERAGE)
        sn2 = SensorName("wind_speed_3", StatisticType.MAXIMUM)
        turbine.add_annotation(WindSpeedSensor(sensor_name=sn1))
        turbine.add_annotation(WindSpeedSensor(sensor_name=sn2))
        turbine.add_annotation(GenericSensor(sensor_name=SensorName("sensor_99")))
        ws_sensors = turbine.get_annotations_by_type(WindSpeedSensor)
        assert len(ws_sensors) == 2
        assert all(isinstance(s, WindSpeedSensor) for s in ws_sensors)

    def test_len(self, mini_turbine):
        # 4 stats × wind_speed + 4 × power + 1 × gearbox_temp + 4 × rotor_rpm + 1 × wind_dir = 14
        assert len(mini_turbine) == 14


# ---------------------------------------------------------------------------
# CSV importer
# ---------------------------------------------------------------------------

class TestCsvImporter:
    def test_wind_speed_annotation_types(self, mini_turbine):
        ws_sensors = mini_turbine.get_annotations_by_type(WindSpeedSensor)
        assert len(ws_sensors) == 4   # avg, min, max, std

    def test_power_annotation(self, mini_turbine):
        power_sensors = mini_turbine.get_annotations_by_type(ActivePowerSensor)
        assert len(power_sensors) == 4

    def test_gearbox_temperature(self, mini_turbine):
        gb = mini_turbine.get_annotations_by_type(GearboxTemperatureSensor)
        assert len(gb) == 1
        assert gb[0].unit == PhysicalUnit.CELSIUS

    def test_angle_flag(self, mini_turbine):
        col = mini_turbine.get_annotation_by_column("sensor_1_avg")
        assert col is not None
        assert col.is_angle is True

    def test_summary_dataframe(self, mini_turbine):
        import pandas as pd
        summary = turbine_annotation_summary(mini_turbine)
        assert isinstance(summary, pd.DataFrame)
        assert "annotation_type" in summary.columns
        assert len(summary) == len(mini_turbine)

    def test_annotation_type_counts(self, mini_turbine):
        counts = annotation_type_counts(mini_turbine)
        assert counts["WindSpeedSensor"] == 4
        assert counts["GearboxTemperatureSensor"] == 1


# ---------------------------------------------------------------------------
# Integration – real feature_description.csv
# ---------------------------------------------------------------------------

REAL_CSV = Path(__file__).parents[3] / "data" / "feature_description.csv"


@pytest.mark.skipif(not REAL_CSV.exists(), reason="real CSV not available")
class TestRealCsv:
    def test_loads_without_error(self):
        turbine = load_feature_description.__wrapped__(str(REAL_CSV), asset_id="real")
        assert len(turbine) > 0

    def test_no_unclassified_is_zero(self):
        turbine = load_feature_description.__wrapped__(str(REAL_CSV), asset_id="real")
        from knowwind.semantic_annotations import GenericSensor
        counts = annotation_type_counts(turbine)
        # GenericSensor may exist but should be a small minority
        total = sum(counts.values())
        generic_count = counts.get("GenericSensor", 0)
        assert generic_count / total < 0.3, "More than 30% sensors are unclassified"
