"""Step 10: exports - GeoTIFF, CSV, Shapefile, interactive HTML maps and district risk statistics."""

from __future__ import annotations

import io
import math
import os
import zipfile
from pathlib import Path

from .config import BASEMAP_ATTRIBUTION, BASEMAP_URL, CLASS_COLORS, CLASS_LABELS
from .model import assign_classes, rank_descending

# Short shapefile field (<= 10 characters) -> descriptive CSV column.
FIELD_DESCRIPTIONS = {
    "ADM0_NAME": "country", "ADM1_NAME": "region", "ADM2_NAME": "district", "ADM2_CODE": "district_code",
    "area_km2": "area_km2", "pop_total": "population", "pop_dens": "population_per_km2",
    "crop_pct": "cropland_cover_pct", "urban_pct": "settlement_cover_pct", "infra": "infrastructure_index_mean",
    "fl_haz": "flood_hazard_mean", "fl_exp": "flood_exposure_mean", "fl_vul": "flood_vulnerability_mean",
    "fl_risk": "flood_risk_index", "fl_class": "flood_risk_class", "fl_rank": "flood_risk_rank",
    "fl_hi_pct": "flood_high_veryhigh_area_pct", "pop_fl_hi": "flood_high_veryhigh_population",
    "dr_haz": "drought_hazard_mean", "dr_exp": "drought_exposure_mean", "dr_vul": "drought_vulnerability_mean",
    "dr_risk": "drought_risk_index", "dr_class": "drought_risk_class", "dr_rank": "drought_risk_rank",
    "dr_hi_pct": "drought_high_veryhigh_area_pct", "pop_dr_hi": "drought_high_veryhigh_population",
    "mh_risk": "multi_hazard_risk_index", "mh_class": "multi_hazard_risk_class", "mh_rank": "multi_hazard_risk_rank",
    # vulnerability components (0-1); v_poverty is empty unless POVERTY_CSV is set
    "v_socio": "vuln_socioeconomic_mean", "v_health": "vuln_health_access_mean",
    "v_remote": "vuln_remoteness_mean", "v_depriv": "vuln_deprivation_mean", "v_poverty": "vuln_poverty_mean",
    "v_fl_land": "vuln_flood_land_use_mean", "v_dr_land": "vuln_drought_land_use_mean",
    "v_agri": "vuln_agricultural_dependence_mean",
}
VULNERABILITY_FIELDS = ["v_socio", "v_health", "v_remote", "v_depriv", "v_poverty", "v_fl_land", "v_dr_land", "v_agri"]


def polygon_parts(geometry):
    """Return only the polygonal parts of a geometry as a MultiPolygon (None if there are none).

    Some GAUL 2015 districts come back from Earth Engine as GeometryCollections
    or with line slivers after simplification; a polygon shapefile rejects them.
    """
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.validation import make_valid

    if geometry is None or geometry.is_empty:
        return None
    geometry = make_valid(geometry)
    if isinstance(geometry, Polygon):
        return MultiPolygon([geometry])
    if isinstance(geometry, MultiPolygon):
        return geometry
    parts = []
    for part in getattr(geometry, "geoms", []):
        cleaned = polygon_parts(part)
        if cleaned is not None:
            parts.extend(cleaned.geoms)
    return MultiPolygon(parts) if parts else None


def finalise_district_table(gdf, method: str = "equal"):
    """Add High/Very High %, district classes and ranks; return (gdf, descriptive table)."""
    import pandas as pd

    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(polygon_parts)
    gdf = gdf.set_geometry("geometry")
    missing = gdf.geometry.isna()
    if missing.any():
        print(f"{missing.sum()} district(s) without polygon geometry (kept in the CSV, left out of maps): "
              f"{gdf.loc[missing, 'ADM2_NAME'].tolist()}")
    for column in ["pop_total", "pop_fl_hi", "pop_dr_hi", "fl_hi_frac", "dr_hi_frac"]:
        gdf[column] = gdf[column].fillna(0) if column in gdf else 0.0
    area = gdf["area_km2"].where(gdf["area_km2"] > 0) if "area_km2" in gdf else float("nan")
    gdf["pop_dens"] = gdf["pop_total"] / area
    for fraction, pct in (("crop_frac", "crop_pct"), ("urban_frac", "urban_pct")):
        gdf[pct] = 100 * gdf[fraction] if fraction in gdf else float("nan")
    for column in ["infra"] + VULNERABILITY_FIELDS:  # optional layers may be missing
        if column not in gdf:
            gdf[column] = float("nan")
    gdf["fl_hi_pct"] = 100 * gdf["fl_hi_frac"]
    gdf["dr_hi_pct"] = 100 * gdf["dr_hi_frac"]
    for prefix in ("fl", "dr", "mh"):
        values = gdf[f"{prefix}_risk"].tolist()
        gdf[f"{prefix}_class"] = assign_classes(values, method)
        gdf[f"{prefix}_rank"] = pd.array(rank_descending(values), dtype="Int64")

    table = pd.DataFrame(gdf.drop(columns="geometry"))[list(FIELD_DESCRIPTIONS)].rename(columns=FIELD_DESCRIPTIONS)
    table = table.sort_values("multi_hazard_risk_rank")
    for column in table.columns:
        if column.endswith(("_mean", "_index")):
            table[column] = table[column].round(3)
        elif column.endswith("_pct") or column in ("area_km2", "population_per_km2"):
            table[column] = table[column].round(1)
        elif column == "population" or column.endswith("_population"):
            table[column] = table[column].round(0).astype(int)
    return gdf, table


def save_shapefile(gdf, out_dir: Path, prefix: str) -> Path:
    """Write the district polygons with statistics as a zipped shapefile plus a field dictionary."""
    import pandas as pd

    shp_dir = Path(out_dir) / f"{prefix}_district_risk_shp"
    shp_dir.mkdir(parents=True, exist_ok=True)
    fields = [f for f in FIELD_DESCRIPTIONS if f != "ADM0_NAME"]
    shp = gdf.loc[gdf.geometry.notna(), fields + ["geometry"]].copy()
    for column in ("fl_rank", "dr_rank", "mh_rank"):
        shp[column] = shp[column].astype(float)
    shp.to_file(shp_dir / f"{prefix}_district_risk.shp")
    pd.DataFrame({"field": fields, "description": [FIELD_DESCRIPTIONS[f] for f in fields]}).to_csv(
        shp_dir / "field_dictionary.csv", index=False)
    zip_path = Path(out_dir) / f"{prefix}_district_risk_shapefile.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in shp_dir.iterdir():
            archive.write(file, arcname=file.name)
    return zip_path


NO_DATA_LABEL = "No data"
NO_DATA_COLOR = "bdbdbd"

# Leaflet snippet that turns a set of overlay layers into "pick one" layers: ticking
# one in the layer control hides the others, so class maps are never stacked.
_EXCLUSIVE_LAYERS_JS = """
(function () {
    var map = %(map)s;
    var group = [%(layers)s];
    map.on('overlayadd', function (event) {
        if (group.indexOf(event.layer) === -1) { return; }
        // Deferred: while the layer control handles a click it re-adds every ticked
        // layer; removing afterwards also refreshes the checkboxes.
        setTimeout(function () {
            group.forEach(function (layer) {
                if (layer !== event.layer && map.hasLayer(layer)) { map.removeLayer(layer); }
            });
        }, 0);
    });
})();
"""


def _add_script(fmap, code: str) -> None:
    """Attach JavaScript that runs after the map and its layers are created.

    Scripts added to the page root run before folium defines the map variable,
    so they are added as a child of the map instead (rendered in order).
    """
    from branca.element import MacroElement
    from jinja2 import Template

    element = MacroElement()
    element._template = Template("{% macro script(this, kwargs) %}" + code + "{% endmacro %}")
    fmap.add_child(element)


def _make_exclusive(fmap, layers) -> None:
    """Show at most one of ``layers`` at a time (radio-like overlays in the layer control)."""
    script = _EXCLUSIVE_LAYERS_JS % {"map": fmap.get_name(), "layers": ", ".join(l.get_name() for l in layers)}
    _add_script(fmap, script)


def _legend_html(title: str, labels, colours, note: str = "") -> str:
    rows = "".join(
        f'<div style="display:flex;align-items:center;margin:2px 0">'
        f'<span style="display:inline-block;width:18px;height:12px;margin-right:6px;'
        f'border:1px solid #777;background:#{colour}"></span>{label}</div>'
        for label, colour in zip(labels, colours))
    note_html = f'<div style="margin-top:6px;color:#555;max-width:220px">{note}</div>' if note else ""
    return (f'<div style="font-weight:bold;margin-bottom:4px">{title}</div>{rows}{note_html}')


def _add_box(fmap, html: str, position: str = "bottomright") -> None:
    """Add a fixed HTML box (legend / note) as a Leaflet control."""
    import json

    script = f"""
    (function () {{
        var box = L.control({{position: '{position}'}});
        box.onAdd = function () {{
            var div = L.DomUtil.create('div', 'safe4all-box');
            div.style.cssText = 'background:rgba(255,255,255,0.92);padding:8px 10px;border-radius:4px;'
                + 'box-shadow:0 1px 4px rgba(0,0,0,0.3);font:12px/1.4 Arial,sans-serif;color:#222';
            div.innerHTML = {json.dumps(html)};
            L.DomEvent.disableClickPropagation(div);
            L.DomEvent.disableScrollPropagation(div);
            return div;
        }};
        box.addTo({fmap.get_name()});
    }})();
    """
    _add_script(fmap, script)


def save_district_html(gdf, path: Path) -> Path:
    """Self-contained interactive district map (flood / drought / multi-hazard layers, pop-ups).

    Works offline apart from the basemap and the Leaflet scripts: all district
    polygons and statistics are embedded in the file. Districts without a risk
    value (e.g. fully masked as water/urban) are drawn grey as "No data".
    """
    import folium

    fields = ["ADM1_NAME", "ADM2_NAME", "pop_total", "fl_risk", "fl_class", "pop_fl_hi",
              "dr_risk", "dr_class", "pop_dr_hi", "mh_risk", "mh_class"]
    aliases = ["Region", "District", "Population", "Flood risk index", "Flood risk class",
               "Pop. in high flood risk", "Drought risk index", "Drought risk class",
               "Pop. in high drought risk", "Multi-hazard index", "Multi-hazard class"]
    data = gdf[gdf.geometry.notna()].copy()
    for column in ("fl_risk", "dr_risk", "mh_risk"):
        # Strings, so a missing value reads "No data" in the tooltip instead of "null".
        data[column] = data[column].apply(lambda v: NO_DATA_LABEL if v is None or v != v else f"{v:.3f}")
    for column in ("pop_total", "pop_fl_hi", "pop_dr_hi"):
        data[column] = data[column].fillna(0).round(0).astype(int)
    for column in ("fl_class", "dr_class", "mh_class"):
        data[column] = data[column].fillna(NO_DATA_LABEL)

    categories = CLASS_LABELS + [NO_DATA_LABEL]
    colours = CLASS_COLORS + [NO_DATA_COLOR]
    fmap = folium.Map(tiles=None, control_scale=True)
    folium.TileLayer(BASEMAP_URL, attr=BASEMAP_ATTRIBUTION, name="Esri World Street Map").add_to(fmap)
    colour_of = {label: "#" + colour for label, colour in zip(categories, colours)}
    data = data[fields + ["geometry"]]
    layers = []
    for i, (column, name) in enumerate([("fl_class", "Flood risk"), ("dr_class", "Drought risk"),
                                        ("mh_class", "Multi-hazard risk")]):
        def style(feature, column=column):
            fill = colour_of.get(feature["properties"][column], colour_of[NO_DATA_LABEL])
            return {"fillColor": fill, "color": "#444444", "weight": 0.6, "fillOpacity": 0.75}

        layer = folium.GeoJson(
            data, name=name, show=(i == 0), style_function=style,
            highlight_function=lambda _: {"weight": 2.5, "color": "#000000", "fillOpacity": 0.85},
            tooltip=folium.GeoJsonTooltip(fields, aliases=aliases, sticky=True),
            popup=folium.GeoJsonPopup(fields, aliases=aliases),
        )
        layer.add_to(fmap)
        layers.append(layer)
    minx, miny, maxx, maxy = data.total_bounds
    fmap.fit_bounds([[miny, minx], [maxy, maxx]])
    _add_box(fmap, _legend_html("Risk class (district mean)", categories, colours,
                                "Tick one layer at a time in the top-right box: flood, drought or multi-hazard."))
    folium.LayerControl(collapsed=False).add_to(fmap)
    _make_exclusive(fmap, layers)
    fmap.save(str(path))
    return Path(path)


def gradient_legend(title: str, colours, low: str, high: str) -> str:
    """HTML for a continuous colour-bar legend (``extra_legend`` of :func:`save_raster_html`)."""
    gradient = ",".join("#" + c.lstrip("#") for c in colours)
    return (f'<div style="font-weight:bold;margin:8px 0 4px">{title}</div>'
            f'<div style="width:200px;height:12px;border:1px solid #777;background:linear-gradient(to right,{gradient})"></div>'
            f'<div style="display:flex;justify-content:space-between;width:202px"><span>{low}</span><span>{high}</span></div>')


def ee_tile_url(image, vis: dict) -> str:
    """XYZ tile URL for an Earth Engine image (valid only for a limited time, typically hours)."""
    return image.getMapId(vis)["tile_fetcher"].url_format


def save_raster_html(ee, layers, gdf, path: Path, title: str, study_gdf=None,
                     class_legend: bool = True, index_legend: bool = True, extra_legend: str = "") -> Path:
    """Standalone interactive raster map: Earth Engine tile layers plus vector district outlines.

    ``layers`` is a list of ``(image, vis, name, shown)``; the last shown layer is on
    top. Unlike ``geemap.Map.to_html`` this does not embed Jupyter widget state, so
    it opens in any browser without Jupyter, holds only this map, and its legend
    renders. Earth Engine tile URLs expire (typically after a few hours to a day);
    district and study-area outlines are embedded as vectors and never expire.
    Re-run the notebook to refresh the raster layers.
    """
    import datetime
    import html

    import folium
    from folium.plugins import Fullscreen

    fmap = folium.Map(tiles=None, control_scale=True)
    folium.TileLayer(BASEMAP_URL, attr=BASEMAP_ATTRIBUTION, name="Esri World Street Map").add_to(fmap)
    raster_layers = []
    for image, vis, name, shown in layers:
        tile = folium.TileLayer(ee_tile_url(image, vis), attr="Google Earth Engine", name=name,
                                overlay=True, control=True, show=bool(shown), opacity=0.85, max_zoom=18)
        tile.add_to(fmap)
        raster_layers.append(tile)

    outlines = gdf[gdf.geometry.notna()].copy()
    tooltip_fields = [f for f in ("ADM1_NAME", "ADM2_NAME", "fl_class", "dr_class", "mh_class") if f in outlines]
    tooltip_alias = {"ADM1_NAME": "Region", "ADM2_NAME": "District", "fl_class": "Flood risk (district)",
                     "dr_class": "Drought risk (district)", "mh_class": "Multi-hazard risk (district)"}
    keep = tooltip_fields + ["geometry"]
    outlines = outlines[keep]
    for field in tooltip_fields:
        outlines[field] = outlines[field].fillna(NO_DATA_LABEL).astype(str)
    folium.GeoJson(
        outlines, name="Districts",
        style_function=lambda _: {"color": "#555555", "weight": 0.7, "fillOpacity": 0, "fill": True},
        highlight_function=lambda _: {"color": "#000000", "weight": 2},
        tooltip=folium.GeoJsonTooltip(tooltip_fields, aliases=[tooltip_alias[f] for f in tooltip_fields]),
    ).add_to(fmap)
    bounds_source = outlines
    if study_gdf is not None and len(study_gdf):
        study = study_gdf[study_gdf.geometry.notna()][["geometry"]]
        folium.GeoJson(study, name="Study area", interactive=False,
                       style_function=lambda _: {"color": "#000000", "weight": 2, "fill": False}).add_to(fmap)
        bounds_source = study
    minx, miny, maxx, maxy = bounds_source.total_bounds
    fmap.fit_bounds([[miny, minx], [maxy, maxx]])

    legend = ""
    if class_legend:
        legend += _legend_html("Risk class (class layers)", CLASS_LABELS, CLASS_COLORS)
    if index_legend:
        gradient = ",".join("#" + c for c in CLASS_COLORS)
        legend += ('<div style="font-weight:bold;margin:8px 0 4px">Index layers (0-1)</div>'
                   f'<div style="width:200px;height:12px;border:1px solid #777;'
                   f'background:linear-gradient(to right,{gradient})"></div>'
                   '<div style="display:flex;justify-content:space-between;width:202px">'
                   '<span>0 low</span><span>0.5</span><span>1 high</span></div>')
    legend += extra_legend
    if legend:
        _add_box(fmap, legend, "bottomright")
    created = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    title = html.escape(title)
    _add_box(fmap, f'<b>{title}</b><br>Raster tiles generated {created}. Earth Engine tiles expire '
                   'after some hours; if layers stay blank, re-run the notebook export. '
                   'District outlines always work.', "bottomleft")
    Fullscreen().add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    _make_exclusive(fmap, raster_layers)
    fmap.get_root().header.add_child(folium.Element(f"<title>{title}</title>"))
    fmap.save(str(path))
    return Path(path)


def save_ranking_chart(table, path: Path, top_n: int = 15) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colours = {label: "#" + c for label, c in zip(CLASS_LABELS, CLASS_COLORS)}
    top_n = min(top_n, len(table))
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * top_n + 1.8))
    for ax, (index, cls, title) in zip(axes, [("flood_risk_index", "flood_risk_class", "Flood risk"),
                                             ("drought_risk_index", "drought_risk_class", "Drought risk")]):
        top = table.nlargest(top_n, index).iloc[::-1]
        ax.barh(top["district"], top[index], color=[colours.get(c, "#999999") for c in top[cls]],
                edgecolor="#555555", lw=0.4)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Mean risk index (0-1)")
        ax.set_title(f"Top {top_n} districts - {title}", loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=[Patch(color=colours[l], label=l) for l in CLASS_LABELS],
               loc="lower center", ncol=5, frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    return Path(path)


def save_static_maps(ee, panels: list, districts, study_fc, study_area, gdf, out_dir: Path,
                     prefix: str, title: str) -> list[Path]:
    """PNG map panel of pixel classes (hazard and risk) and PNG district class maps for reports."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import requests
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from PIL import Image

    handles = [Patch(facecolor="#" + c, edgecolor="#555555", label=l) for l, c in zip(CLASS_LABELS, CLASS_COLORS)]
    bounds = study_area.bounds(1000)
    corners = bounds.coordinates().getInfo()[0]
    xs, ys = [p[0] for p in corners], [p[1] for p in corners]
    extent = [min(xs), max(xs), min(ys), max(ys)]
    outline = (ee.Image().byte().paint(districts, 1, 1).visualize(palette=["555555"])
               .blend(ee.Image().byte().paint(study_fc, 1, 2).visualize(palette=["000000"])))

    def thumbnail(class_image):
        # A country-wide class map is computed in one synchronous request, which can
        # exceed Earth Engine's time/memory limits; retry smaller and report EE's message.
        vis = class_image.visualize(min=1, max=5, palette=CLASS_COLORS).blend(outline)
        error = ""
        for dimensions in (1200, 800, 512):
            try:
                url = vis.getThumbURL({"region": bounds, "dimensions": dimensions, "format": "png", "crs": "EPSG:4326"})
                response = requests.get(url, timeout=600)
            except Exception as exc:
                error = str(exc)
                continue
            if response.ok and response.headers.get("Content-Type", "").startswith("image"):
                return Image.open(io.BytesIO(response.content))
            error = response.text[:300]
        raise RuntimeError(error or "no image returned")

    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    rows = math.ceil(len(panels) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(12, 6 * rows * aspect + 1.5), squeeze=False)
    for ax, (image, label) in zip(axes.ravel(), panels):
        try:
            ax.imshow(thumbnail(image), extent=extent)
            ax.tick_params(labelsize=8)
        except Exception as exc:
            print(f"{label}: Earth Engine could not render the static map ({str(exc)[:200]}). "
                  "Use the interactive map or the GeoTIFF export for this layer.")
            ax.text(0.5, 0.5, "Not rendered\n(see GeoTIFF export)", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        ax.set_title(label, fontweight="bold")
    for ax in axes.ravel()[len(panels):]:
        ax.set_axis_off()
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    panel_path = Path(out_dir) / f"{prefix}_risk_map_panel.png"
    plt.savefig(panel_path, dpi=200, bbox_inches="tight")
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, (column, label) in zip(axes, [("fl_class", "Flood risk"), ("dr_class", "Drought risk"),
                                          ("mh_class", "Multi-hazard risk")]):
        gdf.assign(_c=pd.Categorical(gdf[column], categories=CLASS_LABELS)).plot(
            column="_c", categorical=True, cmap=ListedColormap(["#" + c for c in CLASS_COLORS]),
            edgecolor="white", linewidth=0.3, ax=ax, missing_kwds={"color": "lightgrey"})
        ax.set_title(f"{label} by district", fontweight="bold")
        ax.set_axis_off()
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    district_path = Path(out_dir) / f"{prefix}_district_risk_maps.png"
    plt.savefig(district_path, dpi=200, bbox_inches="tight")
    plt.show()
    return [panel_path, district_path]


def export_geotiffs(ee, rasters: dict, prefix: str, folder: str, region, scale: int = 1000) -> list:
    """Start Earth Engine GeoTIFF exports to Google Drive (indices Float32, classes Byte)."""
    tasks = []
    for name, image in rasters.items():
        image = image.toByte() if name.endswith("_class") else image.toFloat()
        task = ee.batch.Export.image.toDrive(
            image=image, description=f"{prefix}_{name}"[:100], folder=folder,
            fileNamePrefix=f"{prefix}_{name}", region=region, scale=scale, crs="EPSG:4326",
            maxPixels=1e13, fileFormat="GeoTIFF")
        task.start()
        tasks.append(task)
        print("Started GeoTIFF export:", name)
    return tasks


GEOTIFF_NODATA = -9999  # index/value layers; class layers use 0 (classes are 1-5)


def _http_fetch(url, retries: int = 6, sleep=None):
    """GET ``url``; on an HTTP error, raise with Earth Engine's own error message.

    "429 Too Many Requests" (project busy, e.g. restricted mode) is retried with
    increasing waits (5, 10, 20 ... s) before giving up.
    """
    import time

    import requests

    sleep = sleep or time.sleep
    for attempt in range(retries + 1):
        response = requests.get(url, timeout=600)
        if response.status_code != 429 or attempt == retries:
            break
        sleep(5 * 2 ** attempt)
    if response.status_code >= 400:
        try:
            detail = response.json()["error"]["message"]
        except Exception:
            detail = response.text[:200]
        raise RuntimeError(f"HTTP {response.status_code}: {detail}")
    return response.content


def _prepare_geotiff(ee, name, image):
    is_class = name.endswith("_class")
    nodata = 0 if is_class else GEOTIFF_NODATA
    image = ee.Image(image).unmask(nodata)
    return (image.toByte() if is_class else image.toFloat()), nodata


def _download(image, path: Path, name: str, prefix: str, region, scale, fetch) -> None:
    url = image.getDownloadURL({"name": f"{prefix}_{name}", "region": region, "scale": scale,
                                "crs": "EPSG:4326", "format": "GEO_TIFF"})
    path.write_bytes(fetch(url))


def download_geotiffs(ee, rasters: dict, out_dir: Path, prefix: str, region, scale: int = 1000,
                      fetch=None, log=print, asset_folder: str | None = None, wait_minutes: float = 90,
                      poll_seconds: float = 20, sleep=None, breaks: dict | None = None) -> list:
    """Save each raster as a GeoTIFF in ``out_dir`` (no Drive export needed).

    Every layer is first requested as a direct download. Heavy layers (long time
    series, road density) can exceed the interactive time or memory limit (HTTP 503
    or 400). With ``asset_folder`` (e.g. ``projects/<EE_PROJECT>/assets/safe4all_exports``)
    those layers are then computed as Earth Engine batch tasks, which have much
    higher limits, into a temporary asset; the finished asset is downloaded and
    deleted. With ``breaks`` (from ``classify_layers``), a failed multi-hazard risk
    or ``<layer>_class`` raster is first derived locally from GeoTIFFs already saved
    (see :func:`derive_geotiffs_locally`), which needs no Earth Engine compute.
    Masked pixels are ``GEOTIFF_NODATA`` (0 for class layers).
    """
    fetch = fetch or _http_fetch
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved, failed = [], {}
    for name, image in rasters.items():
        image, nodata = _prepare_geotiff(ee, name, image)
        path = out_dir / f"{prefix}_{name}.tif"
        try:
            _download(image, path, name, prefix, region, scale, fetch)
        except Exception as error:
            failed[name] = (image, nodata)
            log(f"Direct download failed: {name} ({str(error)[:200]})")
            continue
        _tag_nodata(path, nodata)
        saved.append(path)
        log(f"Saved GeoTIFF: {path.name}")
    if failed and breaks is not None:
        derived = derive_geotiffs_locally(out_dir, prefix, list(failed), breaks, log)
        saved += derived
        for path in derived:
            failed.pop(path.stem[len(prefix) + 1:], None)
    if failed and asset_folder:
        saved += _download_via_assets(ee, failed, out_dir, prefix, region, scale, fetch, log,
                                      asset_folder, wait_minutes, poll_seconds, sleep)
    elif failed:
        log(f"Skipped: {sorted(failed)}")
    log(f"{len(saved)} of {len(rasters)} GeoTIFF(s) saved to {out_dir}")
    return saved


def _break_values(breaks) -> list | None:
    values = []
    for value in breaks:
        if isinstance(value, (int, float)):
            values.append(float(value))
            continue
        try:
            values.append(float(value.getInfo()))
        except Exception:
            return None
    return values if len(values) == 4 else None


def derive_geotiffs_locally(out_dir: Path, prefix: str, names, breaks: dict, log=print) -> list:
    """Build missing GeoTIFFs from ones already saved, with no Earth Engine request.

    - ``multi_hazard_risk`` = mean of ``flood_risk`` and ``drought_risk`` (nodata where
      either is nodata), exactly as the notebook defines it;
    - ``<layer>_class`` = ``<layer>`` cut at ``breaks[<layer>]`` into classes 1-5 (a value
      equal to a break falls in the lower class, as in ``classify``).

    The inputs share one grid because they were downloaded with the same region,
    scale and CRS. Returns the paths written; needs rasterio (skipped otherwise).
    """
    try:
        import numpy as np
        import rasterio
    except ImportError:
        return []
    out_dir = Path(out_dir)
    path = lambda name: out_dir / f"{prefix}_{name}.tif"  # noqa: E731
    wanted = [n for n in names if n == "multi_hazard_risk"] + [n for n in names if n != "multi_hazard_risk"]
    written = []
    for name in wanted:
        try:
            if name == "multi_hazard_risk":
                if not (path("flood_risk").exists() and path("drought_risk").exists()):
                    continue
                with rasterio.open(path("flood_risk")) as a, rasterio.open(path("drought_risk")) as b:
                    flood, drought, profile = a.read(1), b.read(1), a.profile
                    if (a.width, a.height, a.transform) != (b.width, b.height, b.transform):
                        continue
                invalid = ((flood == GEOTIFF_NODATA) | (drought == GEOTIFF_NODATA)
                           | ~np.isfinite(flood) | ~np.isfinite(drought))
                data = np.where(invalid, GEOTIFF_NODATA, (flood + drought) / 2).astype("float32")
                profile.update(dtype="float32", nodata=GEOTIFF_NODATA, count=1)
            elif name.endswith("_class"):
                base = name[: -len("_class")]
                values = _break_values(breaks.get(base, ())) if base in breaks else None
                if values is None or not path(base).exists():
                    continue
                with rasterio.open(path(base)) as source:
                    index, profile = source.read(1), source.profile
                classes = np.ones(index.shape, dtype="uint8")
                for value in values:
                    classes += (index > value).astype("uint8")
                data = np.where((index == GEOTIFF_NODATA) | ~np.isfinite(index), 0, classes).astype("uint8")
                profile.update(dtype="uint8", nodata=0, count=1)
            else:
                continue
            with rasterio.open(path(name), "w", **profile) as target:
                target.write(data, 1)
        except Exception as error:
            log(f"Could not derive {name} locally ({str(error)[:150]})")
            continue
        written.append(path(name))
        log(f"Saved GeoTIFF (derived from saved layers): {path(name).name}")
    return written


def _create_folder(ee, path: str) -> None:
    try:
        ee.data.createAsset({"type": "FOLDER"}, path)
    except Exception as error:
        if "exist" not in str(error).lower():
            raise


def _download_via_assets(ee, failed: dict, out_dir: Path, prefix: str, region, scale, fetch, log,
                         asset_folder: str, wait_minutes: float, poll_seconds: float, sleep) -> list:
    """Compute layers as batch tasks into temporary assets, then download and delete them."""
    import datetime
    import time
    import uuid

    sleep = sleep or time.sleep
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run = f"{asset_folder.rstrip('/')}/{prefix}_{stamp}_{uuid.uuid4().hex[:6]}"  # unique per participant run
    try:
        _create_folder(ee, asset_folder.rstrip("/"))
        _create_folder(ee, run)
    except Exception as error:
        log(f"Could not create the Earth Engine asset folder {run} ({str(error)[:200]}); skipped: {sorted(failed)}")
        return []

    tasks = {}
    for name, (image, _) in failed.items():
        try:
            task = ee.batch.Export.image.toAsset(
                image=image, description=f"{prefix}_{name}"[:100], assetId=f"{run}/{name}", region=region,
                scale=scale, crs="EPSG:4326", maxPixels=1e13, pyramidingPolicy={".default": "sample"})
            task.start()
            tasks[name] = task
        except Exception as error:
            log(f"GeoTIFF skipped: {name} (batch export could not start: {str(error)[:200]})")
    log(f"Computing {len(tasks)} layer(s) as Earth Engine batch tasks (up to {wait_minutes:g} min; "
        "slower while the project is in restricted mode)...")

    states, deadline = {}, time.time() + wait_minutes * 60
    while tasks:
        for name, task in tasks.items():
            if states.get(name) not in ("COMPLETED", "FAILED", "CANCELLED"):
                try:
                    status = task.status()
                except Exception as error:  # transient API error: try again next round
                    status = {"state": states.get(name, "UNKNOWN"), "error_message": str(error)[:100]}
                if status.get("state") != states.get(name):
                    log(f"  {name}: {status.get('state')} {status.get('error_message', '')}".rstrip())
                states[name] = status.get("state")
        if all(state in ("COMPLETED", "FAILED", "CANCELLED") for state in states.values()):
            break
        if time.time() > deadline:
            log(f"Stopped waiting after {wait_minutes:g} min. Still running: "
                f"{[n for n, s in states.items() if s not in ('COMPLETED', 'FAILED', 'CANCELLED')]}. "
                f"They will finish in {run}; re-run this cell later to retry.")
            break
        sleep(poll_seconds)

    saved = []
    for name, state in states.items():
        if state != "COMPLETED":
            continue
        asset_id = f"{run}/{name}"
        path = out_dir / f"{prefix}_{name}.tif"
        try:
            _download(ee.Image(asset_id), path, name, prefix, region, scale, fetch)
        except Exception as error:
            log(f"GeoTIFF skipped: {name} (download of the finished asset failed: {str(error)[:200]})")
            continue
        _tag_nodata(path, failed[name][1])
        saved.append(path)
        log(f"Saved GeoTIFF: {path.name}")
        try:
            ee.data.deleteAsset(asset_id)
        except Exception:
            pass
    if all(state in ("COMPLETED", "FAILED", "CANCELLED") for state in states.values()):
        try:
            ee.data.deleteAsset(run)  # only empty folders can be deleted; leftovers are kept
        except Exception:
            pass
    return saved


def _tag_nodata(path: Path, nodata) -> None:
    """Make masked pixels ``nodata`` and record it so GIS software hides them (needs rasterio).

    Earth Engine writes pixels outside the clipped study area as -inf in Float32
    downloads (``unmask`` only fills inside the footprint); they are set to ``nodata``.
    """
    try:
        import numpy as np
        import rasterio
    except ImportError:
        return
    import logging

    logger = logging.getLogger("rasterio._env")
    level = logger.level
    logger.setLevel(logging.ERROR)  # hides a harmless "legacy Deflate codec" warning
    try:
        with rasterio.open(path, "r+") as dataset:
            if np.dtype(dataset.dtypes[0]).kind == "f":
                data = dataset.read(1)
                invalid = ~np.isfinite(data)
                if invalid.any():
                    data[invalid] = nodata
                    dataset.write(data, 1)
            dataset.nodata = nodata
    except Exception:
        pass
    finally:
        logger.setLevel(level)


def export_tables(ee, collection, prefix: str, folder: str, formats=("CSV", "SHP")) -> list:
    tasks = []
    for fmt in formats:
        task = ee.batch.Export.table.toDrive(collection=collection, description=f"{prefix}_district_stats_{fmt}"[:100],
                                             folder=folder, fileNamePrefix=f"{prefix}_district_risk_stats",
                                             fileFormat=fmt)
        task.start()
        tasks.append(task)
        print("Started table export:", fmt)
    return tasks


def zip_outputs(out_dir: Path, zip_path: Path) -> Path:
    out_dir = Path(out_dir)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for root, _, files in os.walk(out_dir):
            for file in files:
                full = Path(root) / file
                if full.resolve() != Path(zip_path).resolve():
                    archive.write(full, arcname=full.relative_to(out_dir))
    return Path(zip_path)
