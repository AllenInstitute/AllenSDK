"""Regression tests for pandas 2.x datetime compatibility.

pandas 2.x changed ``pd.to_datetime`` to require an explicit ``format``
parameter when the input has mixed-precision timestamps (some with
microseconds, some without).  These tests exercise the SDK code paths
that parse ``date_of_acquisition`` from metadata tables to ensure they
handle real-world format variation.
"""

import pandas as pd
import pytest

from allensdk.brain_observatory.behavior.behavior_project_cache.tables.metadata_table_schemas import (  # noqa: E501
    BehaviorSessionMetadataSchema,
)


# Minimal valid records for BehaviorSessionMetadataSchema.
# Only date_of_acquisition and required fields are populated.
def _make_record(date_str):
    return {
        "age_in_days": 100,
        "date_of_acquisition": date_str,
    }


class TestDateOfAcquisitionParsing:
    """Regression: the SDK must parse date_of_acquisition values with
    inconsistent microsecond precision, as found in real Allen Institute
    metadata CSVs.
    """

    def test_schema_parses_date_without_microseconds(self):
        record = _make_record("2019-09-25 13:31:46")
        schema = BehaviorSessionMetadataSchema()
        result = schema.load(record)
        assert isinstance(result["date_of_acquisition"], pd.Timestamp)
        assert result["date_of_acquisition"].tzinfo is not None

    def test_schema_parses_date_with_microseconds(self):
        record = _make_record("2019-10-01 14:22:33.123456")
        schema = BehaviorSessionMetadataSchema()
        result = schema.load(record)
        assert isinstance(result["date_of_acquisition"], pd.Timestamp)
        assert result["date_of_acquisition"].microsecond == 123456

    def test_schema_parses_timezone_aware_date(self):
        record = _make_record("2019-09-25T13:31:46+00:00")
        schema = BehaviorSessionMetadataSchema()
        result = schema.load(record)
        assert isinstance(result["date_of_acquisition"], pd.Timestamp)
        assert result["date_of_acquisition"].tzinfo is not None

    def test_cloud_api_mixed_precision_dates(self):
        """Simulate the cloud API's to_datetime call on a column with
        mixed microsecond precision, as found in real session CSVs."""
        df = pd.DataFrame({
            "date_of_acquisition": [
                "2019-09-25 13:31:46",
                "2019-10-01 14:22:33.123456",
                "2020-01-15 09:00:00",
                "2020-06-30 16:45:12.500000",
            ]
        })
        # This is the exact call pattern used in behavior_project_cloud_api.py
        df["date_of_acquisition"] = pd.to_datetime(
            df["date_of_acquisition"], format="ISO8601", utc=True
        )
        assert df["date_of_acquisition"].dt.tz is not None
        assert len(df) == 4

    def test_mixed_precision_fails_without_explicit_format(self):
        """Verify that the pandas 2.x breakage we're guarding against is
        real: without format="ISO8601", a strict format string rejects
        rows that lack microseconds."""
        major = int(pd.__version__.split(".")[0])
        if major < 2:
            pytest.skip("pandas <2.x infers formats silently")

        with pytest.raises(ValueError, match="doesn't match format"):
            pd.to_datetime(
                pd.Series([
                    "2019-09-25 13:31:46",
                    "2019-10-01 14:22:33.123456",
                ]),
                format="%Y-%m-%d %H:%M:%S.%f",
            )
