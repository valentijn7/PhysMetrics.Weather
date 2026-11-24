""" geometry of a complete Gaussian grid stored along one point dimension """

from dataclasses import dataclass

import numpy as np
from scipy.special import roots_legendre


@dataclass
class GaussianRings:
    latitude: np.ndarray
    longitude: np.ndarray
    rings: list
    weights: np.ndarray

    @classmethod
    def from_coords(cls, lat, lon):
        lat = np.asarray(lat, dtype=float)
        lon = np.mod(np.asarray(lon, dtype=float), 360)

        if lat.ndim != 1 or lat.shape != lon.shape or not lat.size:
            raise ValueError("Expected latitude and longitude per point")
        if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
            raise ValueError("Non-finite point coordinates")

        rows, counts = np.unique(lat, return_counts=True)
        if rows.size < 2 or np.any(counts < 3):
            raise ValueError("Incomplete latitude rings")

        nodes, weights = roots_legendre(rows.size)
        if not np.allclose(rows, np.rad2deg(np.arcsin(nodes)),
                           atol=5e-4, rtol=0):
            raise ValueError("Latitudes are not global Gaussian nodes")

        rings = np.split(np.lexsort((lon, lat)), np.cumsum(counts)[:-1])
        for ring in rings:
            x = lon[ring]
            gaps = np.diff(x, append=x[0] + 360)

            if not np.allclose(gaps, 360 / len(x), atol=1e-3, rtol=0):
                raise ValueError("Incomplete or irregular Gaussian longitude ring")

        return cls(np.deg2rad(rows), lon, rings, weights)

    def cell_area(self, radius):
        area = np.empty(self.longitude.size)
        for ring, weight in zip(self.rings, self.weights):
            area[ring] = 2 * np.pi * radius**2 * weight / len(ring)
        return area

    def derivatives(self, phi):
        """Return dPhi/d(latitude, longitude), both per radian, in input order."""
        north = np.empty(phi.shape, dtype=float)
        east = np.empty_like(north)

        for i, ring in enumerate(self.rings):
            vals = phi[ring]
            east[ring] = (np.roll(vals, -1) - np.roll(vals, 1)) / (
                4 * np.pi / len(ring)
            )
            lo = max(0, i - 1)
            hi = min(len(self.rings) - 1, i + 1)

            def on_row(j):
                idx = self.rings[j]
                return np.interp(self.longitude[ring], self.longitude[idx],
                                 phi[idx], period=360)

            north[ring] = (on_row(hi) - on_row(lo)) / (
                self.latitude[hi] - self.latitude[lo]
            )
        
        return north, east
