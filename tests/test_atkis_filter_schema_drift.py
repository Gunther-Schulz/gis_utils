"""A classification filter with no matching columns must not pass through.

When ALL classification columns are absent from the WFS result (schema drift),
the old code returned the frame unfiltered AND stamped it with the
classification labels — false provenance (e.g. widmung=BAB on never-matched
features).  It must raise instead.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from pbs_gis.atkis import _attribute_filter_gdf

CRS = "EPSG:25833"


def _lines_gdf(**cols):
    n = len(next(iter(cols.values()))) if cols else 1
    geoms = [LineString([(i, 0), (i + 1, 1)]) for i in range(n)]
    return gpd.GeoDataFrame({**cols, "geometry": geoms}, crs=CRS)


def test_all_columns_missing_raises():
    gdf = _lines_gdf(some_other_col=["x", "y"])
    with pytest.raises(RuntimeError, match="keine der Klassifikationsspalten"):
        _attribute_filter_gdf(gdf, {"widmung": "1301", "bezeichnung": "A24"})


def test_present_columns_filter_as_before():
    gdf = _lines_gdf(widmung=["1301", "9999", "1301"])
    out = _attribute_filter_gdf(gdf, {"widmung": "1301"})
    assert len(out) == 2
    assert set(out["widmung"]) == {"1301"}
