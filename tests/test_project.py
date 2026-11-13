import ast
import contextlib
import importlib.util
import io
import json
import math
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

    @unittest.skipUnless(importlib.util.find_spec("pandas") and importlib.util.find_spec("requests"),
                         "pandas/requests not installed")
    def test_tahmo_station_list_is_never_cached_empty(self):
        # An empty cached station list (saved by an older version) made every
        # later run try 0 stations. It must be refetched, and an API answer
        # with no stations for the country must raise, not be cached.
        from safe4all import tahmo

        api = {"data": [
            {"code": "TA00001", "location": {"countrycode": "KE", "latitude": 0.1, "longitude": 37.0}},
            {"code": "TA00002", "location": {"countrycode": "GH", "latitude": 9.4, "longitude": -0.8}},
        ]}
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "tahmo" / "Kenya" / "station_metadata.csv"
            path.parent.mkdir(parents=True)
            path.write_text("code,location.latitude,location.longitude\n")  # stale empty list
            with mock.patch.object(tahmo, "_get", return_value=api):
                kenya = tahmo.load_station_metadata(path, "Kenya", log=lambda *_: None)
                self.assertEqual(kenya["code"].tolist(), ["TA00001"])
                self.assertIn("TA00001", path.read_text())

                zim_path = Path(tmp_dir) / "zim.csv"
                with self.assertRaisesRegex(LookupError, "none usable in Zimbabwe.*'KE': 1"):
                    tahmo.load_station_metadata(zim_path, "Zimbabwe", log=lambda *_: None)
                self.assertFalse(zim_path.exists())

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

    def test_region_statistics_are_fetched_once_or_left_lazy(self):
        from safe4all import ee_tools

        class Value:
            def __init__(self, value):
                self.value = value

            def get(self, _):
                return "band"

        class FakeEE:
            calls = 0
            fail = False

            def Dictionary(self, mapping):
                outer = self

                class Request:
                    def getInfo(self):
                        outer.calls += 1
                        if outer.fail:
                            raise RuntimeError("Computation timed out.")
                        return {"band": "band", "stats": mapping["stats"]}
                return Request()

        class Image:
            def bandNames(self):
                return Value(None)

        ee = FakeEE()
        stats = {"band_p2": 1, "band_p98": 9.5}
        self.assertEqual(ee_tools.resolve_stats(ee, Image(), stats, ["_p2", "_p98"]), [1.0, 9.5])
        self.assertEqual(ee.calls, 1)  # band name and both percentiles in one request
        # Empty region (null percentile) or a failed request: keep the lazy expression.
        self.assertIsNone(ee_tools.resolve_stats(ee, Image(), {"band_p2": None, "band_p98": 3}, ["_p2", "_p98"]))
        ee.fail = True
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(ee_tools.resolve_stats(ee, Image(), stats, ["_p2", "_p98"]))
        ee.fail = False
        original = ee_tools.PRECOMPUTE_STATS
        try:
            ee_tools.PRECOMPUTE_STATS = False
            calls = ee.calls
            self.assertIsNone(ee_tools.resolve_stats(ee, Image(), stats, ["_p2", "_p98"]))
            self.assertEqual(ee.calls, calls)
        finally:
            ee_tools.PRECOMPUTE_STATS = original

    def test_geotiffs_download_into_the_output_folder(self):
        from safe4all.export import GEOTIFF_NODATA, download_geotiffs

        class Image:
            def __init__(self, name, too_big=False):
                self.name, self.too_big, self.nodata, self.dtype = name, too_big, None, None

            def unmask(self, value):
                self.nodata = value
                return self

            def toByte(self):
                self.dtype = "byte"
                return self

            def toFloat(self):
                self.dtype = "float"
                return self

            def getDownloadURL(self, params):
                if self.too_big:
                    raise RuntimeError("Total request size must be less than or equal to 33554432 bytes.")
                assert params["format"] == "GEO_TIFF" and params["scale"] == 1000
                return f"https://ee/{self.name}"

        class FakeEE:
            def Image(self, image):
                return image

        images = {"flood_risk": Image("flood_risk"), "flood_risk_class": Image("flood_risk_class"),
                  "huge": Image("huge", too_big=True)}
        messages = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            saved = download_geotiffs(FakeEE(), images, Path(tmp_dir) / "geotiff", "Ghana", region=None,
                                      fetch=lambda url: url.encode(), log=messages.append)
            self.assertEqual([p.name for p in saved], ["Ghana_flood_risk.tif", "Ghana_flood_risk_class.tif"])
            self.assertEqual(saved[0].read_bytes(), b"https://ee/flood_risk")
        self.assertEqual((images["flood_risk"].nodata, images["flood_risk"].dtype), (GEOTIFF_NODATA, "float"))
        self.assertEqual((images["flood_risk_class"].nodata, images["flood_risk_class"].dtype), (0, "byte"))
        self.assertTrue(any(m.startswith("Direct download failed: huge") for m in messages))
        self.assertIn("2 of 3 GeoTIFF(s) saved", messages[-1])

    def test_heavy_geotiffs_fall_back_to_batch_tasks_and_temporary_assets(self):
        from safe4all.export import download_geotiffs

        class Image:
            def __init__(self, name, heavy=False):
                self.name, self.heavy = name, heavy

            def unmask(self, value):
                return self

            def toByte(self):
                return self

            def toFloat(self):
                return self

            def getDownloadURL(self, params):
                if self.heavy:
                    raise RuntimeError("HTTP 503: Computation timed out.")
                return f"https://ee/{self.name}"

        class Task:
            def __init__(self, asset_id, states):
                self.asset_id, self.states = asset_id, list(states)

            def start(self):
                pass

            def status(self):
                return {"state": self.states.pop(0) if len(self.states) > 1 else self.states[0]}

        created, deleted, started = [], [], {}

        class FakeEE:
            class data:
                @staticmethod
                def createAsset(value, path):
                    created.append(path)

                @staticmethod
                def deleteAsset(path):
                    deleted.append(path)

            class batch:
                class Export:
                    class image:
                        @staticmethod
                        def toAsset(image, assetId, **kwargs):
                            states = ["FAILED"] if image.name == "broken" else ["READY", "RUNNING", "COMPLETED"]
                            started[image.name] = Task(assetId, states)
                            return started[image.name]

            def Image(self, image):
                return Image(image.split("/")[-1]) if isinstance(image, str) else image

        images = {"population": Image("population"), "drought_hazard": Image("drought_hazard", heavy=True),
                  "broken": Image("broken", heavy=True)}
        messages = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            saved = download_geotiffs(FakeEE(), images, Path(tmp_dir), "Ghana", region=None,
                                      fetch=lambda url: url.encode(), log=messages.append,
                                      asset_folder="projects/p/assets/safe4all_exports", sleep=lambda _: None)
            self.assertEqual(sorted(p.name for p in saved), ["Ghana_drought_hazard.tif", "Ghana_population.tif"])
            self.assertEqual((Path(tmp_dir) / "Ghana_drought_hazard.tif").read_bytes(), b"https://ee/drought_hazard")
        self.assertEqual(sorted(started), ["broken", "drought_hazard"])  # only failed layers become tasks
        run = created[1]
        self.assertTrue(run.startswith("projects/p/assets/safe4all_exports/Ghana_"))
        self.assertIn(f"{run}/drought_hazard", deleted)  # temporary asset removed after download
        self.assertIn("2 of 3 GeoTIFF(s) saved", messages[-1])

    def test_district_table_includes_vulnerability_components(self):
        from safe4all.export import VULNERABILITY_FIELDS

        for field in VULNERABILITY_FIELDS:
            self.assertIn(field, FIELD_DESCRIPTIONS)
            self.assertLessEqual(len(field), 10, field)

    def test_roads_are_dropped_only_when_too_large_for_the_study_area(self):
        from safe4all.exposure import roads_usable

        class Roads:
            def toFloat(self):
                return self

            def getDownloadURL(self, params):
                return "https://ee/roads"

        class FakeEE:
            def Image(self, image):
                return image

        def failing(message):
            def fetch(url):
                raise RuntimeError(message)
            return fetch

        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertTrue(roads_usable(FakeEE(), Roads(), None, fetch=lambda url: b"tif"))
            self.assertFalse(roads_usable(FakeEE(), Roads(), None,
                                          fetch=failing("HTTP 400: Object too large (142340800 bytes).")))
            self.assertFalse(roads_usable(FakeEE(), Roads(), None,
                                          fetch=failing("HTTP 400: User memory limit exceeded.")))
            # A busy project (429) is not a reason to change the method.
            self.assertTrue(roads_usable(FakeEE(), Roads(), None,
                                         fetch=failing("HTTP 429: Too Many Requests")))
        self.assertIn("night-time lights only", out.getvalue())

    @unittest.skipUnless(importlib.util.find_spec("requests"), "requests not installed")
    def test_busy_project_downloads_are_retried(self):
        from safe4all.export import _http_fetch

        class Response:
            def __init__(self, status, content=b""):
                self.status_code, self.content, self.text = status, content, ""

            def json(self):
                return {"error": {"message": "Too Many Requests"}}

        replies = [Response(429), Response(429), Response(200, b"tif")]
        waits = []
        with mock.patch("requests.get", side_effect=lambda *a, **k: replies.pop(0)):
            self.assertEqual(_http_fetch("https://ee/x", sleep=waits.append), b"tif")
        self.assertEqual(waits, [5, 10])
        with mock.patch("requests.get", return_value=Response(429)):
            with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
                _http_fetch("https://ee/x", retries=2, sleep=lambda _: None)

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

    def test_district_table_reports_separate_exposure_layers(self):
        from safe4all.export import finalise_district_table

        gdf = self.district_gdf()
        for field, values in {"ADM0_NAME": ["Ghana", "Ghana"], "ADM2_CODE": [1, 2], "area_km2": [10.0, 0.0],
                              "fl_haz": [0.5, 0.2], "fl_exp": [0.4, 0.3], "fl_vul": [0.3, 0.2],
                              "dr_haz": [0.5, 0.2], "dr_exp": [0.4, 0.3], "dr_vul": [0.3, 0.2],
                              "crop_frac": [0.25, 0.5], "urban_frac": [0.1, 0.0], "infra": [0.33333, 0.1]}.items():
            gdf[field] = values
        _, table = finalise_district_table(gdf)
        row = table.set_index("district").loc["Alpha"]
        self.assertAlmostEqual(row["population_per_km2"], 100.0)
        self.assertAlmostEqual(row["cropland_cover_pct"], 25.0)
        self.assertAlmostEqual(row["settlement_cover_pct"], 10.0)
        self.assertAlmostEqual(row["infrastructure_index_mean"], 0.333)
        # A zero-area district has no density rather than a division error.
        self.assertTrue(math.isnan(table.set_index("district").loc["Beta", "population_per_km2"]))

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
