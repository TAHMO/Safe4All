"""Steps 8-9: multi-criteria risk model, classification and district statistics.

Flood Risk   = Flood Hazard   x Flood Exposure   x Flood Vulnerability
Drought Risk = Drought Hazard x Drought Exposure x Drought Vulnerability
"""

from __future__ import annotations

import math

from .config import CLASS_LABELS, EQUAL_BREAKS, class_label
from .ee_tools import class_breaks, classify, minmax


def risk_index(ee, hazard, exposure, vulnerability, land_mask, name: str, region, scale: int = 1000):
    """H x E x V, masked to land and rescaled to 0-1 (2nd-98th percentile).

    Multiplying three 0-1 indices compresses values toward 0; rescaling keeps the
    ranking and makes the five classes usable. Risk is high only where all three
    components are high.
    """
    raw = hazard.multiply(exposure).multiply(vulnerability).updateMask(land_mask).rename("raw")
    return minmax(ee, raw, region, scale, name)


def classify_layers(ee, layers: dict, region, scale: int = 1000, method: str = "equal") -> tuple[dict, dict]:
    """Classify each 0-1 layer into classes 1-5; returns (classes, breaks)."""
    classes, breaks = {}, {}
    for name, image in layers.items():
        breaks[name] = class_breaks(ee, image, region, scale, method)
        classes[name] = classify(ee, image, breaks[name], f"{name}_class", region)
    return classes, breaks


def class_area_population(ee, class_image, population, region, scale: int = 1000) -> list[dict]:
    """Area (km2) and population in each risk class over the study area."""
    grouped = ee.Reducer.sum().group(1, "cls")
    area = (ee.Image.pixelArea().divide(1e6).addBands(class_image)
            .reduceRegion(grouped, region, scale, bestEffort=True, maxPixels=1e13, tileScale=4).get("groups"))
    # No scale: the population grid's own projection is used, so totals are exact.
    people = (population.addBands(class_image)
              .reduceRegion(grouped, region, maxPixels=1e13, tileScale=4).get("groups"))
    groups = ee.Dictionary({"area": area, "people": people}).getInfo()  # one round trip, not two
    area_by = {int(g["cls"]): g["sum"] for g in groups["area"]}
    people_by = {int(g["cls"]): g["sum"] for g in groups["people"]}
    total_area, total_people = sum(area_by.values()) or 1, sum(people_by.values()) or 1
    return [{"class": label, "area_km2": area_by.get(i, 0.0), "area_pct": 100 * area_by.get(i, 0.0) / total_area,
             "population": people_by.get(i, 0.0), "population_pct": 100 * people_by.get(i, 0.0) / total_people}
            for i, label in enumerate(CLASS_LABELS, start=1)]


def district_statistics(ee, districts, mean_layers: dict, population, high_risk_classes: dict, scale: int = 1000):
    """Per-district mean indices, share of area and people in High/Very High risk.

    ``mean_layers`` maps short field names (<= 10 characters, shapefile-safe) to
    0-1 images; ``high_risk_classes`` maps a short hazard prefix (e.g. 'fl') to
    its class image. Adds ``<prefix>_hi_frac`` and ``pop_<prefix>_hi`` fields.
    """
    means = [image.rename(field) for field, image in mean_layers.items()]
    means += [image.gte(4).rename(f"{prefix}_hi_frac") for prefix, image in high_risk_classes.items()]
    # Population bands come first so the sum runs on the population grid (exact totals).
    sums = [population.rename("pop_total")]
    sums += [population.updateMask(image.gte(4)).rename(f"pop_{prefix}_hi") for prefix, image in high_risk_classes.items()]
    with_means = ee.Image.cat(means).reduceRegions(collection=districts, reducer=ee.Reducer.mean(),
                                                   scale=scale, tileScale=4)
    return ee.Image.cat(sums).reduceRegions(collection=with_means, reducer=ee.Reducer.sum(), tileScale=4)


def assign_classes(values, method: str = "equal") -> list[str]:
    """Risk class labels for district mean indices.

    ``equal`` uses the fixed 0.2 breaks; ``quantile`` puts 20 % of districts in
    each class (falls back to ``equal`` with fewer than five districts).
    Missing values return ``"No data"``.
    """
    values = [None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v) for v in values]
    valid = sorted((v, i) for i, v in enumerate(values) if v is not None)
    labels = ["No data"] * len(values)
    if method == "quantile" and len(valid) >= 5:
        for rank, (_, i) in enumerate(valid):
            labels[i] = CLASS_LABELS[min(rank * 5 // len(valid), 4)]
    else:
        for v, i in valid:
            labels[i] = class_label(v, EQUAL_BREAKS)
    return labels


def rank_descending(values) -> list[int | None]:
    """Rank 1 = highest value; ties share the best rank; missing values get None."""
    clean = [None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v) for v in values]
    ordered = sorted((v for v in clean if v is not None), reverse=True)
    first_rank = {}
    for position, v in enumerate(ordered, start=1):
        first_rank.setdefault(v, position)
    return [None if v is None else first_rank[v] for v in clean]
