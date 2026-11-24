"""Unit tests for physics_metrics.py using pytest and mock xarray datasets."""

import numpy as np
import pytest
import xarray as xr

from physmetrics_weather.physics_metrics import (
    _detect_ensemble_dim,
    _detect_level_dim,
    _integrate_column,
    compute_conservation_scalars,
    compute_drift_percentages,
    compute_drift_slope,
    compute_dry_air_mass,
    compute_geostrophic_imbalance,
    compute_hydrostatic_imbalance,
    compute_ke_spectrum,
    compute_lapse_rate_wasserstein,
    compute_pure_tcwv,
    compute_q_spectrum,
    compute_scalar_spectrum,
    compute_spectral_scores,
    compute_total_energy,
    compute_water_mass,
    derive_surface_pressure,
    get_grid_cell_area,
)


@pytest.fixture
def mock_grid_coords():
    """Create standard Driscoll-Healy latitude/longitude grid coordinates (32x64)."""
    lats = np.linspace(90, -90, 32)
    lons = np.linspace(0, 354.375, 64)
    levels = np.array([1000.0, 850.0, 500.0, 250.0, 100.0])
    return lats, lons, levels


@pytest.fixture
def mock_deterministic_ds(mock_grid_coords):
    """Create a deterministic mock xarray dataset without ensemble dimension."""
    lats, lons, levels = mock_grid_coords
    nlat, nlon, nlev = len(lats), len(lons), len(levels)

    np.random.seed(42)
    shape_3d = (nlev, nlat, nlon)
    shape_2d = (nlat, nlon)

    temp = 280.0 + np.random.randn(*shape_3d) * 5.0
    q = np.abs(np.random.randn(*shape_3d) * 0.005)
    u = np.random.randn(*shape_3d) * 10.0
    v = np.random.randn(*shape_3d) * 10.0
    phi = (
        np.broadcast_to(levels[:, None, None] * 100.0, shape_3d)
        + np.random.randn(*shape_3d) * 10.0
    )
    msl = 101325.0 + np.random.randn(*shape_2d) * 500.0
    sp = 100000.0 + np.random.randn(*shape_2d) * 1000.0

    ds = xr.Dataset(
        data_vars={
            "temperature": (["level", "latitude", "longitude"], temp),
            "specific_humidity": (["level", "latitude", "longitude"], q),
            "u_component_of_wind": (["level", "latitude", "longitude"], u),
            "v_component_of_wind": (["level", "latitude", "longitude"], v),
            "geopotential": (["level", "latitude", "longitude"], phi),
            "mean_sea_level_pressure": (["latitude", "longitude"], msl),
            "surface_pressure": (["latitude", "longitude"], sp),
        },
        coords={
            "latitude": lats,
            "longitude": lons,
            "level": levels,
        },
    )
    return ds


@pytest.fixture
def mock_ensemble_ds(mock_deterministic_ds):
    """Create an ensemble mock xarray dataset with an 'ens' dimension (3 members)."""
    ds = mock_deterministic_ds
    members = [0, 1, 2]

    data_vars = {}
    for var_name, da in ds.data_vars.items():
        shape = (len(members),) + da.shape
        var_data = np.stack([da.values + i * 0.1 for i in members], axis=0)
        dims = ["ens"] + list(da.dims)
        data_vars[var_name] = (dims, var_data)

    coords = dict(ds.coords)
    coords["ens"] = members

    return xr.Dataset(data_vars=data_vars, coords=coords)


@pytest.fixture
def mock_static_ds(mock_grid_coords):
    """Create a static dataset containing surface geopotential."""
    lats, lons, _ = mock_grid_coords
    shape_2d = (len(lats), len(lons))
    z_sfc = np.clip(np.random.randn(*shape_2d) * 500.0, 0, None)

    return xr.Dataset(
        data_vars={"geopotential_at_surface": (["latitude", "longitude"], z_sfc)},
        coords={"latitude": lats, "longitude": lons},
    )


# ============================================================================
# Unit Tests
# ============================================================================

def test_detect_ensemble_dim(mock_deterministic_ds, mock_ensemble_ds):
    """Test auto-detection of ensemble dimension."""
    assert _detect_ensemble_dim(mock_deterministic_ds) is None
    assert _detect_ensemble_dim(mock_ensemble_ds) == "ens"


def test_detect_level_dim(mock_deterministic_ds):
    """Test auto-detection of pressure level dimension."""
    assert _detect_level_dim(mock_deterministic_ds) == "level"


def test_get_grid_cell_area(mock_deterministic_ds):
    """Test computation of grid cell areas."""
    area = get_grid_cell_area(mock_deterministic_ds)
    assert isinstance(area, xr.DataArray)
    assert area.shape == (32, 64)
    assert (area.values > 0).all()
    # Total surface area of Earth ≈ 5.1e14 m²
    assert np.isclose(float(area.sum()), 4 * np.pi * (6.371e6) ** 2, rtol=1e-2)


def test_derive_surface_pressure(mock_deterministic_ds, mock_static_ds):
    """Test surface pressure derivation from MSL and surface geopotential."""
    sp = derive_surface_pressure(mock_deterministic_ds, mock_static_ds)
    assert isinstance(sp, xr.DataArray)
    assert sp.shape == (32, 64)
    assert (sp.values > 50000.0).all()


def test_integrate_column(mock_grid_coords):
    """Test 3D column integration helper."""
    lats, lons, levels = mock_grid_coords
    nlat, nlon, nlev = len(lats), len(lons), len(levels)
    field_3d = xr.DataArray(
        np.ones((nlev, nlat, nlon)), dims=("level", "latitude", "longitude"),
        coords={"level": levels, "latitude": lats, "longitude": lons},
    )
    ps_2d = xr.full_like(field_3d.isel(level=0, drop=True), 100000.0)

    integrated = _integrate_column(field_3d, levels, ps_2d)
    assert integrated.dims == ("latitude", "longitude")
    np.testing.assert_allclose(integrated, 100000.0 / 9.80665)


def test_compute_dry_air_mass_deterministic(mock_deterministic_ds):
    """Test dry air mass calculation for deterministic dataset."""
    area = get_grid_cell_area(mock_deterministic_ds)
    ps = mock_deterministic_ds["surface_pressure"]

    dry_mass = compute_dry_air_mass(mock_deterministic_ds, ps, area)
    assert dry_mass.dims == ()
    assert dry_mass.attrs["units"] == "Eg"
    assert 4000 < dry_mass.item() < 6000


def test_compute_dry_air_mass_ensemble(mock_ensemble_ds):
    """Test dry air mass calculation for ensemble dataset."""
    area = get_grid_cell_area(mock_ensemble_ds)
    ps = mock_ensemble_ds["surface_pressure"]

    dry_mass = compute_dry_air_mass(mock_ensemble_ds, ps, area)
    assert dry_mass.dims == ("ens",)
    np.testing.assert_array_equal(dry_mass.ens, [0, 1, 2])
    assert np.all((dry_mass > 4000) & (dry_mass < 6000))


def test_one_exagram_of_dry_air():
    # 10000 kg/m^2 on one cell of 10^11 m^2 gives exactly one exagram
    ps = xr.DataArray([[98066.5]], dims=("latitude", "longitude"))
    area = xr.full_like(ps, 1e11)
    mass = compute_dry_air_mass(xr.Dataset(), ps, area, tcwv=xr.zeros_like(ps))
    assert mass.item() == pytest.approx(1.0)
    assert mass.attrs["units"] == "Eg"


def _check_member_metric(single, ensemble):
    assert isinstance(single, xr.DataArray)
    assert single.dims == ()
    assert isinstance(ensemble, xr.DataArray)
    assert ensemble.dims == ("ens",)
    np.testing.assert_array_equal(ensemble.ens, [0, 1, 2])
    assert np.isfinite(ensemble).all()
    # member zero is identical to the deterministic fixture
    np.testing.assert_allclose(ensemble.sel(ens=0), single)


def test_compute_water_mass(mock_deterministic_ds, mock_ensemble_ds):
    """Test global water mass calculation for deterministic and ensemble datasets."""
    area = get_grid_cell_area(mock_deterministic_ds)
    ps = mock_deterministic_ds["surface_pressure"]

    water_det = compute_water_mass(mock_deterministic_ds, ps, area)
    water_ens = compute_water_mass(mock_ensemble_ds, mock_ensemble_ds["surface_pressure"], area)
    _check_member_metric(water_det, water_ens)
    assert (water_ens > 0).all()


def test_compute_total_energy(mock_deterministic_ds, mock_static_ds, mock_ensemble_ds):
    """Test total energy calculation for deterministic and ensemble datasets."""
    area = get_grid_cell_area(mock_deterministic_ds)
    ps = mock_deterministic_ds["surface_pressure"]
    z_sfc = mock_static_ds["geopotential_at_surface"]

    te_det = compute_total_energy(mock_deterministic_ds, ps, area, z_sfc=z_sfc)
    te_ens = compute_total_energy(
        mock_ensemble_ds, mock_ensemble_ds["surface_pressure"], area, z_sfc=z_sfc
    )
    _check_member_metric(te_det, te_ens)
    assert (te_ens > 0).all()


def test_compute_ke_spectrum(mock_deterministic_ds, mock_ensemble_ds):
    """Test KE spectrum calculation."""
    res_det = compute_ke_spectrum(mock_deterministic_ds, level=500.0)
    assert isinstance(res_det, tuple)
    k, e = res_det
    assert len(k) == len(e)

    res_ens = compute_ke_spectrum(mock_ensemble_ds, level=500.0)
    assert isinstance(res_ens, dict)
    assert len(res_ens) == 3


def test_compute_q_spectrum(mock_deterministic_ds, mock_ensemble_ds):
    """Test Q spectrum calculation."""
    res_det = compute_q_spectrum(mock_deterministic_ds, level=500.0)
    assert isinstance(res_det, tuple)

    res_ens = compute_q_spectrum(mock_ensemble_ds, level=500.0)
    assert isinstance(res_ens, dict)
    assert len(res_ens) == 3


def test_compute_spectral_scores():
    """Test spectral divergence and residual calculations."""
    e_true = np.exp(-np.linspace(0, 5, 20))
    e_pred = e_true * 1.1

    s_div, s_res = compute_spectral_scores(e_pred, e_true)
    assert isinstance(s_div, float)
    assert isinstance(s_res, float)
    assert s_div >= 0
    assert s_res >= 0


def test_compute_hydrostatic_imbalance(mock_deterministic_ds, mock_ensemble_ds):
    """Test hydrostatic balance RMSE calculation."""
    area = get_grid_cell_area(mock_deterministic_ds)

    hydro_det = compute_hydrostatic_imbalance(mock_deterministic_ds, area)
    hydro_ens = compute_hydrostatic_imbalance(mock_ensemble_ds, area)
    _check_member_metric(hydro_det, hydro_ens)
    assert (hydro_ens >= 0).all()


def test_compute_geostrophic_imbalance(mock_deterministic_ds, mock_ensemble_ds):
    """Test geostrophic balance RMSE calculation."""
    area = get_grid_cell_area(mock_deterministic_ds)

    geo_det = compute_geostrophic_imbalance(mock_deterministic_ds, area)
    geo_ens = compute_geostrophic_imbalance(mock_ensemble_ds, area)
    _check_member_metric(geo_det, geo_ens)
    assert (geo_ens >= 0).all()


def test_compute_lapse_rate_wasserstein(mock_deterministic_ds):
    """Test lapse rate Wasserstein distance calculation."""
    area = get_grid_cell_area(mock_deterministic_ds)
    res = compute_lapse_rate_wasserstein(mock_deterministic_ds, mock_deterministic_ds, area)
    assert isinstance(res, dict)
    assert "lapse_rate_w1_tropics" in res


def test_compute_drift_slope_and_percentages():
    """Test drift slope and drift percentage calculations."""
    hours = np.array([12, 24, 36, 48])
    values = np.array([100.0, 101.0, 102.0, 103.0])

    slope = compute_drift_slope(hours, values)
    assert np.isclose(slope, 2.0)  # 1 unit per 12h = 2 units per 24h (day)

    drift = compute_drift_percentages(
        hours, values, values, values,
        hours, values, values
    )
    assert isinstance(drift, dict)
    assert "dry_mass_drift_pct_per_day" in drift


def test_compute_conservation_scalars(mock_deterministic_ds, mock_static_ds, mock_ensemble_ds):
    """Test conservation scalars wrapper function."""
    area = get_grid_cell_area(mock_deterministic_ds)
    ps = mock_deterministic_ds["surface_pressure"]
    z_sfc = mock_static_ds["geopotential_at_surface"]

    scalars_det = compute_conservation_scalars(mock_deterministic_ds, ps, area, z_sfc=z_sfc)
    assert isinstance(scalars_det, tuple)
    assert len(scalars_det) == 3

    scalars_ens = compute_conservation_scalars(
        mock_ensemble_ds, mock_ensemble_ds["surface_pressure"], area, z_sfc=z_sfc
    )
    assert isinstance(scalars_ens, tuple)
    assert len(scalars_ens) == 3
    for single, ensemble in zip(scalars_det, scalars_ens):
        _check_member_metric(single, ensemble)
    for member in mock_ensemble_ds.ens.values:
        ds = mock_ensemble_ds.sel(ens=member, drop=True)
        expected = compute_conservation_scalars(ds, ds.surface_pressure, area, z_sfc)
        for ensemble, single in zip(scalars_ens, expected):
            np.testing.assert_allclose(ensemble.sel(ens=member), single)



def test_compute_pure_tcwv(mock_deterministic_ds):
    """Test fixed-level pure TCWV integration."""
    tcwv_pure = compute_pure_tcwv(mock_deterministic_ds)
    assert isinstance(tcwv_pure, xr.DataArray)
    assert tcwv_pure.shape == (32, 64)


def test_compute_scalar_spectrum(mock_deterministic_ds):
    """Test spherical harmonic spectrum calculation for arbitrary scalar fields."""
    k, spec = compute_scalar_spectrum(mock_deterministic_ds, var_name="temperature", level=850.0)
    assert isinstance(k, np.ndarray)
    assert isinstance(spec, np.ndarray)
    assert len(k) == len(spec)
    assert len(k) > 0


def test_gaussian_area_mass_and_balance():
    """Integrate a constant column and a balanced layer on shuffled rings."""
    from physmetrics_weather import physics_metrics as pm
    a = np.degrees(np.arcsin(1 / np.sqrt(3)))
    order = np.arange(10)[::-1]
    lat = np.repeat([-a, a], [4, 6])[order]
    lon = np.r_[np.arange(4) * 90, np.arange(6) * 60][order]
    ds = xr.Dataset(coords={"latitude": ("point", lat),
                            "longitude": ("point", lon), "level": [500, 850]})
    area = pm.get_grid_cell_area(ds)
    globe = 4 * np.pi * pm.EARTH_RADIUS**2
    np.testing.assert_allclose(area, np.where(lat < 0, globe / 8, globe / 12))
    ds["t"] = (("level", "point"), np.full((2, 10), 280.))
    ds["q"] = xr.zeros_like(ds.t)
    ds["u"] = ds["v"] = xr.zeros_like(ds.t)
    thickness = pm.R_DRY * 280 * np.log(850 / 500)
    ds["z"] = xr.where(ds.level == 500, thickness, 0.) * xr.ones_like(ds.t)
    ps = xr.full_like(area, 100000.)
    dry, water, energy = pm.compute_conservation_scalars(ds, ps, area, ps * 0)
    mass = globe * 100000 / pm.GRAVITY
    np.testing.assert_allclose(dry, mass / pm.EXAGRAM)
    np.testing.assert_allclose(water, 0)
    np.testing.assert_allclose(energy, mass * pm.C_PD * 280)
    np.testing.assert_allclose(pm.compute_hydrostatic_imbalance(ds, area), 0,
                               atol=1e-9)
    ds["z"] = ds.z + 1000 * np.deg2rad(ds.latitude)
    ds["u"] = xr.ones_like(ds.t) * (
        -1000 / (2 * pm.OMEGA * np.sin(np.deg2rad(lat)) * pm.EARTH_RADIUS)
    )
    np.testing.assert_allclose(pm.compute_geostrophic_imbalance(ds, area),
                               0, atol=1e-9)
    with pytest.raises(ValueError, match="longitude ring"):
        pm.get_grid_cell_area(ds.isel(point=slice(1, None)))


def test_gaussian_derivatives_are_periodic_and_use_matching_longitudes():
    from physmetrics_weather._gaussian import GaussianRings
    nodes = np.polynomial.legendre.leggauss(4)[0]
    rows = np.arcsin(nodes)
    errors = []
    for scale in (1, 2):
        counts = np.array([8, 12, 12, 8]) * scale
        lat = np.repeat(rows, counts)
        lon = np.concatenate([np.arange(n) * 360. / n for n in counts])
        grid = GaussianRings.from_coords(np.rad2deg(lat), lon)
        lam = np.deg2rad(lon)
        north, east = grid.derivatives(lat + np.sin(lam))
        errors.append(np.max(abs(north - 1)) + np.max(abs(east - np.cos(lam))))
    assert errors[1] < .4 * errors[0]
