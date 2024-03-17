import polars as pl
import pytest
import polars_deltalake as pldl
from polars.testing import assert_frame_equal
from datetime import datetime, timezone


@pytest.fixture()
def data_batch_1():
    return pl.DataFrame(
        {
            "foo": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "bar": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "datetime": [datetime(2010, 1, 1)] * 9,
            "datetime_tz": [datetime(2010, 1, 1, tzinfo=timezone.utc)] * 9,
            "date_month": [
                201001,
                201002,
                201003,
                201004,
                201005,
                201006,
                201007,
                201008,
                201009,
            ],
            "static_part": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            # "list": [["5", "B"]] * 9,
        }
    )


def test_roundtrip_read(tmp_path, data_batch_1: pl.DataFrame):
    data_batch_1.write_delta(tmp_path, mode="append")

    result = pldl.scan_delta(str(tmp_path)).collect()

    assert_frame_equal(result, data_batch_1)

    data_batch_1.write_delta(tmp_path, mode="append")
    result = pldl.scan_delta(str(tmp_path)).collect()
    assert_frame_equal(result, pl.concat([data_batch_1] * 2))


def test_roundtrip_read_filter(tmp_path, data_batch_1: pl.DataFrame):
    data_batch_1.write_delta(tmp_path, mode="append")

    result = pldl.scan_delta(str(tmp_path)).filter(pl.col("foo") > 5).collect()

    assert_frame_equal(result, data_batch_1.filter(pl.col("foo") > 5))

    data_batch_1.write_delta(tmp_path, mode="append")
    result = pldl.scan_delta(str(tmp_path)).filter(pl.col("foo") > 5).collect()
    assert_frame_equal(result, pl.concat([data_batch_1] * 2).filter(pl.col("foo") > 5))


def test_roundtrip_read_partitioned_dtypes(tmp_path, data_batch_1: pl.DataFrame):
    """Checks if partition dtypes are using the delta types, Polars inference of hive
    parition dtypes is not always correct."""
    data_batch_1.write_delta(
        tmp_path,
        mode="append",
        delta_write_options={"partition_by": ["bar"]},
    )

    result = pldl.scan_delta(str(tmp_path)).collect()

    assert_frame_equal(
        result, data_batch_1, check_row_order=False, check_column_order=False
    )

    data_batch_1.write_delta(tmp_path, mode="append")
    result = pldl.scan_delta(str(tmp_path)).collect()
    assert_frame_equal(
        result,
        pl.concat([data_batch_1] * 2),
        check_row_order=False,
        check_column_order=False,
    )


def test_roundtrip_read_partitioned(tmp_path, data_batch_1: pl.DataFrame):
    data_batch_1.write_delta(
        tmp_path,
        mode="append",
        delta_write_options={"partition_by": ["date_month", "static_part"]},
    )

    result = pldl.scan_delta(str(tmp_path)).collect()

    assert_frame_equal(result, data_batch_1, check_row_order=False)

    data_batch_1.write_delta(tmp_path, mode="append")
    result = pldl.scan_delta(str(tmp_path)).collect()
    assert_frame_equal(result, pl.concat([data_batch_1] * 2), check_row_order=False)


def test_roundtrip_read_partitioned_filtered(tmp_path, data_batch_1: pl.DataFrame):
    data_batch_1.write_delta(
        tmp_path,
        mode="append",
        delta_write_options={"partition_by": ["date_month", "static_part"]},
    )

    result = (
        pldl.scan_delta(str(tmp_path))
        .filter(
            (pl.col("static_part") == "A")
            & (pl.col("date_month").is_in([201001, 201002]))
        )
        .collect()
    )

    assert_frame_equal(
        result,
        data_batch_1.filter(
            (pl.col("static_part") == "A")
            & (pl.col("date_month").is_in([201001, 201002]))
        ),
        check_row_order=False,
    )

    data_batch_1.write_delta(tmp_path, mode="append")
    result = (
        pldl.scan_delta(str(tmp_path))
        .filter(
            (pl.col("static_part") == "A")
            & (pl.col("date_month").is_in([201001, 201002]))
        )
        .collect()
    )
    assert_frame_equal(
        result,
        pl.concat([data_batch_1] * 2).filter(
            (pl.col("static_part") == "A")
            & (pl.col("date_month").is_in([201001, 201002]))
        ),
        check_row_order=False,
    )


def test_roundtrip_read_partitioned_filtered_select(
    tmp_path, data_batch_1: pl.DataFrame
):
    data_batch_1.write_delta(
        tmp_path,
        mode="append",
        delta_write_options={"partition_by": ["date_month", "static_part"]},
    )

    result = (
        pldl.scan_delta(str(tmp_path))
        .filter(
            (pl.col("static_part") == "A")
            & (pl.col("date_month").is_in([201001, 201002]))
        )
        .select("foo")
        .collect()
    )

    assert_frame_equal(
        result,
        data_batch_1.filter(
            (pl.col("static_part") == "A")
            & (pl.col("date_month").is_in([201001, 201002]))
        ).select("foo"),
        check_row_order=False,
    )

    data_batch_1.write_delta(tmp_path, mode="append")
    result = (
        pldl.scan_delta(str(tmp_path))
        .filter(
            (pl.col("static_part") == "A")
            & (pl.col("date_month").is_in([201001, 201002]))
        )
        .select("foo")
        .collect()
    )
    assert_frame_equal(
        result,
        pl.concat([data_batch_1] * 2)
        .filter(
            (pl.col("static_part") == "A")
            & (pl.col("date_month").is_in([201001, 201002]))
        )
        .select("foo"),
        check_row_order=False,
    )


def test_roundtrip_read_schema_evolved(tmp_path, data_batch_1: pl.DataFrame):
    data_batch_1.write_delta(tmp_path, mode="append")

    result = pldl.scan_delta(str(tmp_path)).collect()

    assert_frame_equal(result, data_batch_1)

    data_batch_2 = data_batch_1.with_columns(pl.lit("new_value").alias("new_col"))

    data_batch_2.write_delta(
        tmp_path,
        mode="append",
        delta_write_options={"schema_mode": "merge", "engine": "rust"},
    )
    result = pldl.scan_delta(str(tmp_path)).collect()

    assert result.schema == data_batch_2.schema

    assert_frame_equal(
        result,
        pl.concat([data_batch_1, data_batch_2], how="diagonal"),
        check_row_order=False,
        check_column_order=False,
    )
