"""Unit tests for run_all_metrics.py CLI runner and helper functions."""

import argparse
import numpy as np
import pytest
import xarray as xr

from physmetrics_weather import run_all_metrics as runner

from physmetrics_weather.run_all_metrics import (
    _align_ref_to_model,
    _grids_match,
    _parse_lead_times,
    _resolve_dates,
)


def test_parse_lead_times():
    """Test parsing of lead-time specification strings."""
    parsed = _parse_lead_times("12h,5d,10d")
    assert len(parsed) == 3
    assert parsed[0] == ("12h", np.timedelta64(12, "h"))
    assert parsed[1] == ("5d", np.timedelta64(120, "h"))
    assert parsed[2] == ("10d", np.timedelta64(240, "h"))

    with pytest.raises(ValueError):
        _parse_lead_times("invalid_time")


def test_resolve_dates():
    """Test resolution of date arguments."""
    args = argparse.Namespace(dates=["2022-01-01", "2022-01-02"], month=None, year=2022)
    resolved = _resolve_dates(args)
    assert resolved == ["2022-01-01T00:00:00", "2022-01-02T00:00:00"]

    args_month = argparse.Namespace(dates=None, month="2022-02", year=2022)
    resolved_month = _resolve_dates(args_month)
    assert len(resolved_month) == 28


def test_grids_match():
    """Test dataset grid matching comparison."""
    lats = np.linspace(90, -90, 32)
    lons = np.linspace(0, 350, 64)

    ds_a = xr.Dataset(coords={"latitude": lats, "longitude": lons})
    ds_b = xr.Dataset(coords={"latitude": lats, "longitude": lons})
    ds_c = xr.Dataset(coords={"latitude": np.linspace(90, -90, 16), "longitude": lons})

    assert _grids_match(ds_a, ds_b) is True
    assert _grids_match(ds_a, ds_c) is False


def test_align_ref_to_model():
    """Test grid alignment between reference and model datasets."""
    lats_ref = np.linspace(90, -90, 33)
    lats_mod = np.linspace(90, -90, 32)
    lons = np.linspace(0, 350, 64)

    ds_ref = xr.Dataset(
        data_vars={"temp": (["latitude", "longitude"], np.zeros((33, 64)))},
        coords={"latitude": lats_ref, "longitude": lons},
    )
    ds_mod = xr.Dataset(
        data_vars={"temp": (["latitude", "longitude"], np.zeros((32, 64)))},
        coords={"latitude": lats_mod, "longitude": lons},
    )

    aligned = _align_ref_to_model(ds_ref, ds_mod)
    assert aligned.sizes["latitude"] == 32


def _two_lead_datasets():
    init = np.datetime64("2020-01-01", "ns")
    leads = np.array([12, 24], dtype="timedelta64[h]")
    ref = xr.Dataset(coords={
        "time": init + leads, "level": [500, 850],
        "latitude": [-60., 0., 60.], "longitude": [0., 90., 180., 270.],
    })
    dims = ("time", "level", "latitude", "longitude")
    state = {"t": 280., "q": 0.01, "z": 0., "u": 0., "v": 0.}
    for name, value in state.items():
        ref[name] = (dims, np.full((2, 2, 3, 4), value))
    ref["surface_pressure"] = xr.ones_like(ref.t.isel(level=0, drop=True)) * 1e5
    ref["surface_pressure"][1] = 101000.
    surface = ref.z.isel(time=0, level=0, drop=True)
    ref["geopotential_at_surface"] = xr.zeros_like(surface)
    pred = ref.drop_vars("geopotential_at_surface")
    pred = pred.rename(time="prediction_timedelta")
    pred = pred.assign_coords(prediction_timedelta=leads).expand_dims(time=[init])
    members = xr.IndexVariable("ens", [0, 1])
    pred = xr.concat([pred, pred], dim=members, data_vars="all")
    pred["surface_pressure"][1] = 100000.
    return pred, ref, leads


@pytest.mark.parametrize("missing_reference", [False, True])
def test_cli_matches_reference_budget_trends(monkeypatch, missing_reference):
    pred, ref, leads = _two_lead_datasets()
    if missing_reference:
        ref = ref.isel(time=[1])
    stores = {"pred": pred, "ref": ref}
    monkeypatch.setattr(runner, "open_zarr_anonymous", stores.__getitem__)
    monkeypatch.setattr(runner, "_align_ref_to_model", lambda *args: None)
    rows, *_ = runner._evaluate_one(
        "pred", "ref", "2020-01-01", "24h", leads[-1], 1, 1, "joint", False
    )
    for name in ("water_mass_drift_pct_per_day", "total_energy_drift_pct_per_day"):
        values = [r["model_value"] for r in rows if r["metric_name"] == name]
        assert len(values) == 2
        if missing_reference:
            assert np.isnan(values).all()
        else:
            np.testing.assert_allclose(values, [0., -2.], atol=1e-9)


def test_balance_summary_pairs_12h_forecast_with_12h_reference(monkeypatch):
    pred, ref, leads = _two_lead_datasets()
    pred["surface_pressure"] = pred.surface_pressure + 200 * pred.ens
    stores = {"pred": pred, "ref": ref}
    monkeypatch.setattr(runner, "open_zarr_anonymous", stores.__getitem__)
    monkeypatch.setattr(runner, "_align_ref_to_model", lambda *args: None)

    # Distinct lead/member markers isolate time pairing from numerical kernels.
    def marker(ds, area, level_dim):
        return ds.surface_pressure.isel(latitude=0, longitude=0, drop=True)

    monkeypatch.setattr(runner, "compute_hydrostatic_imbalance", marker)
    monkeypatch.setattr(runner, "compute_geostrophic_imbalance", marker)
    rows, series, *_ = runner._evaluate_one(
        "pred", "ref", "2020-01-01", "12h", leads[0], 1, 1, "joint", False
    )
    for metric in ("hydrostatic_rmse", "geostrophic_rmse"):
        actual = {
            (row["lead_time_hours"], row["ensemble_member"]):
                (row["model_value"], row["ref_value"])
            for row in rows if row["metric_name"] == metric
        }
        assert actual == {(12, 0): (100000., 100000.), (12, 1): (100200., 100000.)}
        assert any(row["forecast_hour"] == 24 and row[metric] == 101000.
                   for row in series)
