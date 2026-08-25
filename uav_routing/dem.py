"""Load a real digital elevation model (DEM) into a :class:`GridTerrain`.

Supports any raster that ``rasterio`` can open (GeoTIFF ``.tif`` as shipped by
ASTER GDEM V3, SRTM ``.hgt``, ESRI ``.asc``, ...).  A geographic (lon/lat) DEM
is reprojected to a metric UTM frame so that distances and the energy model are
in metres, matching the rest of the package.

This module has an optional dependency on ``rasterio`` (and, for reprojection,
``pyproj`` which rasterio pulls in).  Install with ``pip install rasterio``.
"""

from __future__ import annotations

import numpy as np

from .terrain import GridTerrain


def _utm_epsg_for(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing (lon, lat)."""
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


def load_dem(
    path: str,
    bbox_lonlat: tuple | None = None,
    dst_epsg: int | None = None,
    downsample: int = 1,
    smooth_sigma: float = 0.0,
):
    """Read a DEM file and return a metric :class:`GridTerrain`.

    Parameters
    ----------
    path : str
        Path to the raster file.
    bbox_lonlat : (min_lon, min_lat, max_lon, max_lat), optional
        If given and the DEM is geographic, crop to this window before
        reprojection (keeps the problem small and fast).
    dst_epsg : int, optional
        Target projected CRS EPSG code (e.g. 32650 for UTM 50N). If omitted,
        the UTM zone of the DEM centre is used. Ignored if the DEM is already
        projected in metres.
    downsample : int
        Take every ``downsample``-th pixel to coarsen the grid (default 1).
    smooth_sigma : float
        Standard deviation, in pixels, of an optional Gaussian smoothing of the
        elevation grid (default 0, no smoothing). A small value (1-2) removes
        pixel-scale roughness that would otherwise make the interpolated terrain
        gradient noisy and slow the optimiser, without moving ridges or valleys
        appreciably. Physically reasonable: a cargo drone does not track 30 m
        pixel detail.

    Returns
    -------
    (terrain, meta) : (GridTerrain, dict)
        ``meta`` carries the local-frame origin easting/northing and the CRS so
        that geographic points can be mapped into the terrain frame with
        :func:`lonlat_to_local`.
    """
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.windows import from_bounds

    with rasterio.open(path) as src:
        src_crs = src.crs
        is_geographic = src_crs is not None and src_crs.is_geographic

        window = None
        if bbox_lonlat is not None and is_geographic:
            window = from_bounds(*bbox_lonlat, transform=src.transform)

        data = src.read(1, window=window, masked=True)
        transform = src.window_transform(window) if window is not None else src.transform
        src_bounds = rasterio.windows.bounds(window, src.transform) if window is not None else src.bounds

        if is_geographic:
            lon_c = 0.5 * (src_bounds[0] + src_bounds[2])
            lat_c = 0.5 * (src_bounds[1] + src_bounds[3])
            dst_crs = rasterio.crs.CRS.from_epsg(dst_epsg or _utm_epsg_for(lon_c, lat_c))
            dst_transform, w, h = calculate_default_transform(
                src_crs, dst_crs, data.shape[1], data.shape[0], *src_bounds
            )
            dst = np.zeros((h, w), dtype="float32")
            src_float = np.asarray(data.astype("float32").filled(np.nan))
            reproject(
                source=src_float,
                destination=dst,
                src_transform=transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )
            heights = dst
            transform = dst_transform
            crs = dst_crs
        else:
            heights = np.asarray(data.filled(np.nan), dtype="float32")
            crs = src_crs

    if downsample > 1:
        heights = heights[::downsample, ::downsample]
        transform = transform * transform.identity().scale(downsample, downsample)

    # Fill any remaining NaNs with the minimum valid elevation.
    if np.isnan(heights).any():
        heights = np.where(np.isnan(heights), np.nanmin(heights), heights)

    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter

        heights = gaussian_filter(heights.astype("float64"), sigma=smooth_sigma)

    nrows, ncols = heights.shape
    # Pixel-centre coordinates in the projected CRS.
    xs = transform.c + transform.a * (np.arange(ncols) + 0.5)
    ys = transform.f + transform.e * (np.arange(nrows) + 0.5)

    # rasterio rows go top->bottom (transform.e < 0); flip so y increases upward.
    if transform.e < 0:
        ys = ys[::-1]
        heights = heights[::-1, :]

    # Shift to a local frame with origin at the lower-left corner.
    x0, y0 = float(xs[0]), float(ys[0])
    x_local = xs - x0
    y_local = ys - y0

    terrain = GridTerrain(x_coords=x_local, y_coords=y_local, heights=heights)
    meta = {"origin_easting": x0, "origin_northing": y0, "crs": crs}
    return terrain, meta


def lonlat_to_local(lon, lat, meta):
    """Map a geographic point to the terrain's local (x, y) metric frame."""
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", meta["crs"], always_xy=True)
    e, n = tr.transform(lon, lat)
    return e - meta["origin_easting"], n - meta["origin_northing"]
