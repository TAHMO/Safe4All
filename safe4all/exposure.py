"""Step 6: exposure - WorldPop population, cropland, settlements and infrastructure."""

from __future__ import annotations

from .config import DEFAULT_WEIGHTS
from .ee_tools import minmax, weighted_overlay

WORLDPOP = "WorldPop/GP/100m/pop"
LANDCOVER = "COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019"
VIIRS_MONTHLY = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"
GRIP4_AFRICA = "projects/sat-io/open-datasets/GRIP4/Africa"   # GEE community catalogue


def population_grid(ee, region, year: int = 2020, scale: int = 1000):
    """WorldPop 100 m counts summed to ``scale`` m cells (people per cell, exact totals)."""
    worldpop = ee.ImageCollection(WORLDPOP).filter(ee.Filter.eq("year", year)).filterBounds(region)
    projection = worldpop.first().projection()
    return (worldpop.mosaic().unmask(0).setDefaultProjection(projection)
            .reduceResolution(ee.Reducer.sum().unweighted(), maxPixels=1024)
            .reproject(projection.atScale(scale)).rename("population").clip(region))


def landcover(ee):
    image = ee.Image(LANDCOVER)
    return image, image.select("discrete_classification").projection()


def aggregate_cover(ee, image, region, scale: int = 1000):
    """Mean of a 100 m land-cover layer within each ``scale`` m cell."""
    _, projection = landcover(ee)
    return (ee.Image(image).setDefaultProjection(projection)
            .reduceResolution(ee.Reducer.mean(), maxPixels=1024)
            .reproject(projection.atScale(scale)).clip(region))


def cover_fraction(ee, region, cover: str, scale: int = 1000):
    """0-1 fraction of a Copernicus cover type (e.g. 'crops', 'urban') per cell."""
    image, _ = landcover(ee)
    return aggregate_cover(ee, image.select(f"{cover}-coverfraction").divide(100), region, scale).rename(cover)


def nightlights(ee, year: int = 2023):
    return (ee.ImageCollection(VIIRS_MONTHLY).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .select("avg_rad").median().max(0))


def road_density(ee, region, scale: int = 1000):
    """Share of 100 m cells crossed by a GRIP4 road; None if the community asset is unavailable."""
    roads = ee.FeatureCollection(GRIP4_AFRICA)
    try:
        if roads.filterBounds(region).limit(1).size().getInfo() == 0:
            return None
    except Exception as error:  # asset missing or not shared with this project
        print(f"GRIP4 roads unavailable ({str(error)[:80]}); infrastructure uses night-time lights only.")
        return None
    # Paint the unfiltered collection: Earth Engine then reads only the roads in each
    # tile from the table's spatial index. Painting roads.filterBounds(region) forces
    # all of a country's roads into one in-memory object, which fails ("Object too
    # large") for downloads and exports of a whole country.
    painted = ee.Image(0).byte().paint(roads, 1).reproject(crs="EPSG:4326", scale=100)
    return (painted.reduceResolution(ee.Reducer.mean(), maxPixels=1024)
            .reproject(crs="EPSG:4326", scale=scale).rename("road_density").clip(region))


def exposure_indicators(ee, region, population_year: int = 2020, nightlights_year: int = 2023,
                        scale: int = 1000, use_roads: bool = True, flood_weights=None, drought_weights=None):
    """Compute Step 6 exposure layers and the flood- and drought-specific exposure indices."""
    population = population_grid(ee, region, population_year, scale)
    crops = cover_fraction(ee, region, "crops", scale)
    urban = cover_fraction(ee, region, "urban", scale)
    lights = nightlights(ee, nightlights_year)
    lights_index = minmax(ee, lights.add(1).log(), region, scale, "nightlights")
    roads = road_density(ee, region, scale) if use_roads else None
    infrastructure = lights_index if roads is None else lights_index.add(minmax(ee, roads, region, scale, "roads")).divide(2)

    index = {
        "population": minmax(ee, population.add(1).log(), region, scale, "population"),  # log: cities do not swamp rural areas
        "cropland": crops.clamp(0, 1).rename("cropland"),
        "settlements": urban.divide(0.5).clamp(0, 1).rename("settlements"),      # >= 50 % built-up = maximum
        "infrastructure": infrastructure.rename("infrastructure").clip(region),
    }
    return {
        "index": index, "population": population, "crop_fraction": crops, "urban_fraction": urban,
        "nightlights_index": lights_index,
        "flood": weighted_overlay(ee, index, flood_weights or DEFAULT_WEIGHTS["flood_exposure"], "flood_exposure", region),
        "drought": weighted_overlay(ee, index, drought_weights or DEFAULT_WEIGHTS["drought_exposure"], "drought_exposure", region),
    }
