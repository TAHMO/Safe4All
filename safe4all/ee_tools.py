"""Earth Engine building blocks shared by the hazard, exposure, vulnerability and risk steps."""

from __future__ import annotations

from .config import EQUAL_BREAKS, normalise_weights

TIME_PROPS = ["system:time_start", "year", "month"]


def add_time_props(image):
    import ee

    date = ee.Date(image.get("system:time_start"))
    return image.set({"year": date.get("year"), "month": date.get("month")})


def image_available(ee, image, region, name: str, scale: int = 1000) -> bool:
    """True if Earth Engine can read pixels of ``image`` (catalogue assets can be corrupted or withdrawn)."""
    try:
        ee.Image(image).reduceRegion(ee.Reducer.first(), region.centroid(1000), scale).getInfo()
        return True
    except Exception as error:
        print(f"  {name} unavailable in Earth Engine ({str(error)[:100]}); skipped.")
        return False


def month_filter(ee, months):
    """Filter images to a list of calendar months (works for seasons crossing 31 December)."""
    return ee.Filter.Or(*[ee.Filter.calendarRange(int(m), int(m), "month") for m in months])


# Compute region statistics (percentiles for rescaling, quantile class breaks) once
# and embed them as constants. Otherwise every map tile, export and district
# statistic re-runs the whole-region reduction, and these reductions are nested
# (risk = minmax(hazard x exposure x vulnerability), each built from minmax layers).
# The numbers are identical; set False to go back to fully lazy evaluation.
PRECOMPUTE_STATS = True


def resolve_stats(ee, image, stats, suffixes, label: str = ""):
    """Fetch the first band's ``<band><suffix>`` statistics as Python floats in one request.

    Returns None (caller keeps the lazy server-side expression) if precomputing is
    off, the request fails (e.g. an interactive time-out), or any value is missing.
    """
    if not PRECOMPUTE_STATS:
        return None
    try:
        result = ee.Dictionary({"band": image.bandNames().get(0), "stats": stats}).getInfo() or {}
    except Exception as error:
        print(f"  {label or 'statistics'}: computed lazily ({str(error)[:100]})")
        return None
    values = result.get("stats") or {}
    numbers = [values.get(f"{result.get('band')}{suffix}") for suffix in suffixes]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or v != v for v in numbers):
        return None
    return [float(v) for v in numbers]


def minmax(ee, image, region, scale, name, pmin=2, pmax=98):
    """Rescale an indicator to 0-1 using its 2nd-98th percentile over the region."""
    image = ee.Image(image)
    stats = image.reduceRegion(reducer=ee.Reducer.percentile([pmin, pmax]), geometry=region,
                               scale=scale, bestEffort=True, maxPixels=1e13, tileScale=4)
    resolved = resolve_stats(ee, image, stats, [f"_p{pmin}", f"_p{pmax}"], name)
    if resolved:
        low, high = (ee.Number(v) for v in resolved)
    else:
        band = ee.String(image.bandNames().get(0))
        low = ee.Number(stats.get(band.cat(f"_p{pmin}")))
        high = ee.Number(stats.get(band.cat(f"_p{pmax}")))
    return image.subtract(low).divide(high.subtract(low).max(1e-9)).clamp(0, 1).rename(name).clip(region)


def inverse(ee, image, name):
    return ee.Image(1).subtract(image).rename(name)


def linear_decay(ee, image, maximum, name):
    """Score 1 at zero, falling linearly to 0 at ``maximum`` (e.g. slope, HAND, distance)."""
    return ee.Image(1).subtract(ee.Image(image).divide(maximum)).clamp(0, 1).rename(name)


def weighted_overlay(ee, layers: dict, weights: dict, name: str, region):
    """Weighted sum of 0-1 layers. Missing (None) layers drop out and weights re-scale."""
    available = [key for key, layer in layers.items() if layer is not None]
    used = normalise_weights(weights, available)
    dropped = [key for key, w in weights.items() if w > 0 and key not in used]
    total = ee.Image(0)
    for key, weight in used.items():
        total = total.add(ee.Image(layers[key]).multiply(weight))
    if dropped:
        print(f"  {name}: not available -> {dropped}; remaining weights re-scaled")
    print(f"  {name} weights: { {k: round(w, 3) for k, w in used.items()} }")
    return total.rename(name).clip(region)


def monthly_composites(ee, collection, start_year: int, end_year: int, reducer: str = "mean"):
    """Calendar-month composites with year/month properties; empty months are dropped."""
    start = ee.Date.fromYMD(start_year, 1, 1)
    count = (end_year - start_year + 1) * 12

    def make(offset):
        date = start.advance(offset, "month")
        subset = collection.filterDate(date, date.advance(1, "month"))
        image = subset.mean() if reducer == "mean" else subset.sum()
        return image.set({"system:time_start": date.millis(), "year": date.get("year"),
                          "month": date.get("month"), "n": subset.size()})

    return ee.ImageCollection(ee.List.sequence(0, count - 1).map(make)).filter(ee.Filter.gt("n", 0))


def standardized_anomalies(ee, collection, baseline_filter):
    """z-score of every monthly image against the same calendar month in the baseline."""
    merged = None
    for month in range(1, 13):
        monthly = collection.filter(ee.Filter.eq("month", month))
        base = monthly.filter(baseline_filter)
        mean = base.mean()
        std = base.reduce(ee.Reducer.stdDev())
        z = monthly.map(lambda img: ee.Image(
            img.subtract(mean).divide(std).updateMask(std.gt(1e-6)).rename("z").copyProperties(img, TIME_PROPS)))
        merged = z if merged is None else merged.merge(z)
    return merged


def vci_collection(ee, ndvi_monthly, baseline_filter):
    """Vegetation Condition Index = 100 (NDVI - min) / (max - min), per calendar month."""
    merged = None
    for month in range(1, 13):
        monthly = ndvi_monthly.filter(ee.Filter.eq("month", month))
        base = monthly.filter(baseline_filter)
        low, high = base.min(), base.max()
        span = high.subtract(low)
        vci = monthly.map(lambda img: ee.Image(
            img.subtract(low).divide(span).multiply(100).updateMask(span.gt(0.01)).rename("vci")
            .copyProperties(img, TIME_PROPS)))
        merged = vci if merged is None else merged.merge(vci)
    return merged


def frequency_below(ee, collection, threshold, name, projection, region):
    """Fraction of time steps at or below ``threshold`` (masked steps are ignored)."""
    return (collection.map(lambda img: img.lte(threshold)).mean()
            .rename(name).reproject(projection).clip(region))


def class_breaks(ee, image, region, scale, method: str = "equal"):
    """Return 4 class breaks: fixed 0.2 steps, or area quantiles (20 % per class)."""
    if method == "equal":
        return list(EQUAL_BREAKS)
    if method != "quantile":
        raise ValueError("CLASS_METHOD must be 'equal' or 'quantile'.")
    image = ee.Image(image)
    stats = image.reduceRegion(ee.Reducer.percentile([20, 40, 60, 80]), region, scale,
                               bestEffort=True, maxPixels=1e13, tileScale=4)
    resolved = resolve_stats(ee, image, stats, [f"_p{q}" for q in (20, 40, 60, 80)], "class breaks")
    if resolved:
        return [ee.Number(v) for v in resolved]
    band = ee.String(image.bandNames().get(0))
    return [ee.Number(stats.get(band.cat(f"_p{q}"))) for q in (20, 40, 60, 80)]


def classify(ee, image, breaks, name, region):
    """Classes 1-5 (Very Low ... Very High); a value equal to a break falls in the lower class."""
    b1, b2, b3, b4 = breaks
    return (ee.Image(1).where(image.gt(b1), 2).where(image.gt(b2), 3).where(image.gt(b3), 4)
            .where(image.gt(b4), 5).updateMask(image.mask()).rename(name).toByte().clip(region))
