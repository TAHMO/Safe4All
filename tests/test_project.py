import ast
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from safe4all.aoi import aoi_label
from safe4all.config import (CLASS_LABELS, COUNTRIES, DEFAULT_WEIGHTS, class_label, country_settings,
                             normalise_weights)
from safe4all.export import FIELD_DESCRIPTIONS
from safe4all.gee import initialise_earth_engine
from safe4all.model import assign_classes, rank_descending
from safe4all.validation import rainfall_metrics

ROOT = Path(__file__).resolve().parents[1]


class ProjectTests(unittest.TestCase):
    def test_supported_countries_have_map_settings(self):
        self.assertEqual(set(COUNTRIES), {"Kenya", "Zimbabwe", "Ghana"})
        for country in COUNTRIES:
            settings = country_settings(country)
            self.assertIn("iso3", settings)
            self.assertIn("center", settings)

    def test_unsupported_country_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Unsupported country"):
            country_settings("Uganda")

    def test_rainfall_metrics(self):
        metrics = rainfall_metrics(observed=[1.0, 2.0, 3.0], estimated=[1.0, 3.0, 5.0])
        self.assertEqual(metrics['n'], 3.0)
        self.assertGreater(metrics['rmse_mm'], 0)

    def test_earth_engine_requires_a_project(self):
        # Earth Engine has required every request to name a Cloud project
        # since November 2024; fail fast with a clear message instead of a
        # cryptic error from the `ee` library.
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "EE_PROJECT"):
                initialise_earth_engine()

    def test_earth_engine_uses_a_service_account_key_when_configured(self):
        # Headless/workshop auth path: a service account key file (never
        # committed to the repo) should be used instead of interactive
        # ee.Authenticate() when EE_SERVICE_ACCOUNT_KEY is set.
        fake_ee = mock.MagicMock()
        with tempfile.TemporaryDirectory() as tmp_dir:
            key_path = Path(tmp_dir) / "key.json"
            key_path.write_text(json.dumps({"client_email": "bot@example.iam.gserviceaccount.com"}))
            env = {"EE_PROJECT": "some-project", "EE_SERVICE_ACCOUNT_KEY": str(key_path)}
            with mock.patch.dict(os.environ, env, clear=True), mock.patch.dict(sys.modules, {"ee": fake_ee}):
                initialise_earth_engine()

        fake_ee.ServiceAccountCredentials.assert_called_once_with(
            "bot@example.iam.gserviceaccount.com", str(key_path)
        )
        fake_ee.Authenticate.assert_not_called()
        fake_ee.Initialize.assert_called_once_with(
            fake_ee.ServiceAccountCredentials.return_value, project="some-project"
        )

    def test_earth_engine_uses_the_shared_drive_key_by_default(self):
        # Default workshop path: with no explicit key, service-key.json from the
        # shared SAFE4ALL-workshop Drive folder (a temp folder here) is used.
        fake_ee = mock.MagicMock()
        with tempfile.TemporaryDirectory() as tmp_dir:
            key_path = Path(tmp_dir) / "service-key.json"
            key_path.write_text(json.dumps({"client_email": "workshop@example.iam.gserviceaccount.com"}))
            env = {"EE_PROJECT": "safe4all-514410", "SAFE4ALL_SA_KEY_DIR": tmp_dir}
            with mock.patch.dict(os.environ, env, clear=True), mock.patch.dict(sys.modules, {"ee": fake_ee}):
                initialise_earth_engine()

        fake_ee.ServiceAccountCredentials.assert_called_once_with(
            "workshop@example.iam.gserviceaccount.com", str(key_path)
        )
        fake_ee.Authenticate.assert_not_called()
        fake_ee.Initialize.assert_called_once_with(
            fake_ee.ServiceAccountCredentials.return_value, project="safe4all-514410"
        )

    def test_earth_engine_falls_back_to_interactive_without_shared_key(self):
        fake_ee = mock.MagicMock()
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = {"EE_PROJECT": "safe4all-514410", "SAFE4ALL_SA_KEY_DIR": tmp_dir}
            with mock.patch.dict(os.environ, env, clear=True), mock.patch.dict(sys.modules, {"ee": fake_ee}):
                initialise_earth_engine()

        fake_ee.ServiceAccountCredentials.assert_not_called()
        fake_ee.Initialize.assert_called_once_with(project="safe4all-514410")

    def test_tahmo_credentials_are_read_from_the_shared_drive_file(self):
        from safe4all.tahmo_credentials import load_shared_credentials

        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "tahmo-credentials.txt").write_text(
                "# TAHMO workshop account\nTAHMO_API_KEY = abc123\n\nTAHMO_API_SECRET='s3cret=x'\n"
            )
            with mock.patch.dict(os.environ, {"SAFE4ALL_SA_KEY_DIR": tmp_dir}, clear=True):
                self.assertTrue(load_shared_credentials())
                self.assertEqual(os.environ["TAHMO_API_KEY"], "abc123")
                self.assertEqual(os.environ["TAHMO_API_SECRET"], "s3cret=x")

    def test_tahmo_credentials_missing_or_incomplete(self):
        from safe4all.tahmo_credentials import load_shared_credentials

        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.dict(os.environ, {"SAFE4ALL_SA_KEY_DIR": tmp_dir}, clear=True):
                self.assertFalse(load_shared_credentials())
                (Path(tmp_dir) / "tahmo-credentials.txt").write_text("TAHMO_API_KEY=abc123\n")
                with self.assertRaisesRegex(ValueError, "TAHMO_API_SECRET"):
                    load_shared_credentials()

    @unittest.skipUnless(importlib.util.find_spec("pandas") and importlib.util.find_spec("requests"),
                         "pandas/requests not installed")
    def test_tahmo_skips_inaccessible_stations_until_enough_have_data(self):
        # Kenya case: the station list includes stations this API account cannot
        # read; they must not use up MAX_STATIONS, and empty results must not be
        # cached (an empty file cached by an older version is refetched).
        import pandas as pd
        import requests
        from safe4all.tahmo import collect_station_rainfall

        def fetch(code, start, end):
            if code.startswith("NO"):
                error = requests.HTTPError("403 Client Error")
                error.response = mock.Mock(status_code=403)
                raise error
            if code == "EMPTY":
                return pd.DataFrame()
            return pd.DataFrame({"time": ["2024-03-01T00:00:00Z"], "value": [1.5], "station": [code]})

        codes = ["NO1", "NO2", "EMPTY", "OLD", "A", "B", "C"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "OLD_precip.csv").write_text("\n")  # stale empty cache file
            observations, used, skipped = collect_station_rainfall(
                codes, "2024-03-01", "2024-05-31", 3, cache_dir=tmp_dir, fetch=fetch, log=lambda *_: None)
            cached = sorted(p.name for p in Path(tmp_dir).iterdir())

        self.assertEqual(used, ["OLD", "A", "B"])
        self.assertEqual(skipped, {"NO1": "no access for this API account", "NO2": "no access for this API account",
                                   "EMPTY": "no rainfall data in the period"})
        self.assertEqual(sorted(observations["station"].unique()), ["A", "B", "OLD"])
        self.assertEqual(cached, ["A_precip.csv", "B_precip.csv", "OLD_precip.csv"])

    def test_earth_engine_service_account_key_must_exist(self):
        env = {"EE_PROJECT": "some-project", "EE_SERVICE_ACCOUNT_KEY": "/no/such/key.json"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.dict(sys.modules, {"ee": mock.MagicMock()}):
            with self.assertRaises(FileNotFoundError):
                initialise_earth_engine()


class RiskWorkflowTests(unittest.TestCase):
    def test_countries_define_flood_dry_and_rainy_seasons(self):
        for country in COUNTRIES:
            settings = country_settings(country)
            for key in ("flood_months", "dry_months", "rainy_months"):
                self.assertTrue(set(settings[key]) <= set(range(1, 13)), (country, key))
            self.assertFalse(set(settings["flood_months"]) & set(settings["dry_months"]), country)

    def test_country_settings_returns_independent_copies(self):
        country_settings("Ghana")["flood_months"].append(1)
        self.assertNotIn(1, country_settings("Ghana")["flood_months"])

    def test_weights_drop_missing_layers_and_rescale(self):
        weights = normalise_weights({"a": 0.5, "b": 0.25, "c": 0.25}, available=["a", "b"])
        self.assertEqual(set(weights), {"a", "b"})
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertAlmostEqual(weights["a"], 2 / 3)
        with self.assertRaises(ValueError):
            normalise_weights({"a": -0.1, "b": 1.1})
        with self.assertRaises(ValueError):
            normalise_weights({"a": 1.0}, available=[])

    def test_default_weight_groups_sum_to_one(self):
        for group, weights in DEFAULT_WEIGHTS.items():
            self.assertAlmostEqual(sum(weights.values()), 1.0, msg=group)
        for k in (1, 3, 6):
            self.assertIn(f"spi{k}", DEFAULT_WEIGHTS["drought_hazard"])

    def test_five_risk_classes_with_inclusive_upper_breaks(self):
        self.assertEqual(CLASS_LABELS, ["Very Low", "Low", "Moderate", "High", "Very High"])
        self.assertEqual([class_label(v) for v in (0.0, 0.2, 0.21, 0.5, 0.8, 0.95)],
                         ["Very Low", "Very Low", "Low", "Moderate", "High", "Very High"])

    def test_aoi_label_and_validation(self):
        self.assertEqual(aoi_label("Ghana"), "Ghana")
        self.assertEqual(aoi_label("Ghana", "admin1", admin1="Upper East"), "Ghana_Upper_East")
        with self.assertRaises(ValueError):
            aoi_label("Ghana", "district")
        with self.assertRaises(ValueError):
            aoi_label("Ghana", "watershed")

    def test_district_classes_and_ranks(self):
        values = [0.1, 0.9, float("nan"), 0.5, 0.5]
        self.assertEqual(assign_classes(values), ["Very Low", "Very High", "No data", "Moderate", "Moderate"])
        self.assertEqual(rank_descending(values), [4, 1, None, 2, 2])
        quantile = assign_classes([0.1, 0.2, 0.3, 0.4, 0.5], method="quantile")
        self.assertEqual(quantile, CLASS_LABELS)

    def test_shapefile_field_names_fit_the_dbf_limit(self):
        for field in FIELD_DESCRIPTIONS:
            self.assertLessEqual(len(field), 10, field)

    @unittest.skipUnless(importlib.util.find_spec("shapely"), "shapely not installed")
    def test_district_geometries_keep_only_polygons(self):
        from shapely.geometry import GeometryCollection, LineString, Polygon

        from safe4all.export import polygon_parts

        square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        mixed = polygon_parts(GeometryCollection([square, LineString([(2, 2), (3, 3)])]))
        self.assertEqual(mixed.geom_type, "MultiPolygon")
        self.assertAlmostEqual(mixed.area, 1.0)
        self.assertIsNone(polygon_parts(LineString([(0, 0), (1, 1)])))

    def test_notebook_follows_the_ten_step_workflow(self):
        notebook = json.loads((ROOT / "SAFE4ALL_Climate_Risk_Assessment_Workshop.ipynb").read_text(encoding="utf-8"))
        markdown = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
        for step in ("STEP 1–2", "STEP 3", "STEP 4", "STEP 5", "STEP 6", "STEP 7", "STEP 8", "STEP 9", "STEP 10"):
            self.assertIn(f"# {step} ·", markdown)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                self.assertEqual(cell["outputs"], [])

    def test_notebook_saves_the_raster_map_without_widget_embedding(self):
        # geemap's Map.to_html embeds the Jupyter widget state of every map in the
        # notebook; its anywidget legend/toolbar cannot render outside Jupyter.
        notebook = json.loads((ROOT / "SAFE4ALL_Climate_Risk_Assessment_Workshop.ipynb").read_text(encoding="utf-8"))
        code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
        self.assertNotIn(".to_html(", code)
        self.assertIn("save_raster_html(", code)


HAS_MAP_LIBS = all(importlib.util.find_spec(name) for name in ("folium", "geopandas", "shapely"))


class FakeImage:
    """Stand-in for an ee.Image: only getMapId is used by the HTML export."""

    def __init__(self, map_id):
        self.map_id = map_id

    def getMapId(self, vis):
        fetcher = mock.Mock(url_format=f"https://earthengine.googleapis.com/v1/projects/p/maps/{self.map_id}/tiles/{{z}}/{{x}}/{{y}}")
        return {"tile_fetcher": fetcher}


@unittest.skipUnless(HAS_MAP_LIBS, "folium/geopandas not installed")
class MapExportTests(unittest.TestCase):
    def district_gdf(self):
        import geopandas as gpd
        from shapely.geometry import Polygon

        def square(x, y):
            return Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)])

        return gpd.GeoDataFrame({
            "ADM1_NAME": ["North", "South"], "ADM2_NAME": ["Alpha", "Beta"],
            "pop_total": [1000.4, 2000.0], "pop_fl_hi": [10.0, 0.0], "pop_dr_hi": [5.0, 0.0],
            "fl_risk": [0.91, 0.30], "fl_class": ["Very High", "Low"],
            "dr_risk": [0.55, float("nan")], "dr_class": ["Moderate", "No data"],
            "mh_risk": [0.73, float("nan")], "mh_class": ["High", "No data"],
        }, geometry=[square(-2, 6), square(-1, 7)], crs="EPSG:4326")

    def test_district_map_shows_no_data_and_one_layer_at_a_time(self):
        from safe4all.export import NO_DATA_COLOR, save_district_html

        with tempfile.TemporaryDirectory() as tmp_dir:
            html = save_district_html(self.district_gdf(), Path(tmp_dir) / "d.html").read_text(encoding="utf-8")
        self.assertIn("No data", html)
        self.assertIn("#" + NO_DATA_COLOR, html)
        self.assertNotIn('"dr_risk": null', html)
        self.assertIn("overlayadd", html)
        # Custom scripts must run after folium creates the map, or the page breaks.
        self.assertLess(html.index("= L.map("), html.index("overlayadd"))
        self.assertLess(html.index("= L.map("), html.index("L.control({position"))
        for name in ("Flood risk", "Drought risk", "Multi-hazard risk"):
            self.assertIn(f'"{name}"', html)
        self.assertEqual(html.count("L.control.layers("), 1)

    def test_raster_map_is_plain_leaflet_with_ee_tiles_and_vector_outlines(self):
        from safe4all.export import save_raster_html

        layers = [(FakeImage("flood-index"), {"min": 0, "max": 1}, "Flood risk index", False),
                  (FakeImage("flood-class"), {"min": 1, "max": 5}, "Flood risk class", True)]
        gdf = self.district_gdf()
        with tempfile.TemporaryDirectory() as tmp_dir:
            html = save_raster_html(None, layers, gdf, Path(tmp_dir) / "r.html", title="Risk <Ghana>",
                                    study_gdf=gdf.dissolve()).read_text(encoding="utf-8")
        self.assertNotIn("widget-state", html)
        self.assertNotIn("anywidget", html)
        self.assertIn("maps/flood-index/tiles/{z}/{x}/{y}", html)
        self.assertIn("maps/flood-class/tiles/{z}/{x}/{y}", html)
        self.assertIn('"Districts"', html)
        self.assertIn('"Study area"', html)
        self.assertIn("Alpha", html)
        self.assertIn("Very High", html)
        self.assertIn("Risk &lt;Ghana&gt;", html)
        self.assertEqual(html.count("L.control.layers("), 1)


if __name__ == "__main__":
    unittest.main()
