/**********************************************************
 * Washington DC
 * TRUE pyramid by batches
 * sum -> normalize (De-quantization removed)
 *
 * Outputs:
 * 1) original AEF embedding count in DC
 * 2) 50m true-pyramid embedding count in DC
 **********************************************************/

// --------------------
// PARAMETERS
// --------------------
var YEAR = 2022;
var AGG_SCALE_M = 70;      // target pyramid resolution
var BATCH_SCALE_M = 2000;   // batch tile size
var SAMPLE_SCALE_M = 10;    // stable sampling scale
var TILE_SCALE = 16;

var WORK_CRS = 'EPSG:3857'; // use metric grid in a globally valid CRS

// --------------------
// LOAD DC GEOMETRY
// --------------------
var dc = ee.FeatureCollection('TIGER/2018/States')
  .filter(ee.Filter.eq('STUSPS', 'DC'))
  .first();

var dcGeom = dc.geometry();
var dcGeom3857 = dcGeom.transform(WORK_CRS, 1);

// --------------------
// LOAD AEF EMBEDDING
// IMPORTANT: use mosaic()
// --------------------
var startDate = ee.Date.fromYMD(YEAR, 1, 1);
var endDate   = startDate.advance(1, 'year');

// add filterBounds(dcGeom) to get the sampling boundary
var emb = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
  .filterBounds(dcGeom)
  .filterDate(startDate, endDate)
  .mosaic()
  .clip(dcGeom);

var bands = emb.bandNames();
var firstBand = emb.select(0);

print('Band count:', bands.size());
print('Band names:', bands);

// --------------------
// DIRECT FLOAT CONVERSION 
// (De-quantization formula removed. Kept toFloat() to prevent integer overflow during summation)
// --------------------
var embFloat = emb.toFloat();

// A band for counting valid source pixels
var srcCountBand = ee.Image.constant(1)
  .toFloat()
  .updateMask(firstBand.mask())
  .rename('src_count');

// Stack for later reduceRegions(sum)
var imgAll = embFloat.addBands(srcCountBand);

// --------------------
// HELPER: make regular square grid in WORK_CRS
// FIXED: snap origin to global integer multiples of cellSizeMeters
// --------------------
function makeGrid3857(geometry3857, cellSizeMeters) {
  var bounds = geometry3857.bounds(1, WORK_CRS);
  var ring = ee.List(bounds.coordinates().get(0));

  var ll = ee.List(ring.get(0));
  var ur = ee.List(ring.get(2));

  var xmin_raw = ee.Number(ll.get(0));
  var ymin_raw = ee.Number(ll.get(1));
  var xmax_raw = ee.Number(ur.get(0));
  var ymax_raw = ee.Number(ur.get(1));

  // Snap origin to the nearest lower global integer multiple of cellSizeMeters.
  // This guarantees all batches share the same global grid regardless of their
  // local bounding box, so cells from different batches are perfectly aligned.
  var xmin = xmin_raw.divide(cellSizeMeters).floor().multiply(cellSizeMeters);
  var ymin = ymin_raw.divide(cellSizeMeters).floor().multiply(cellSizeMeters);
  var xmax = xmax_raw.divide(cellSizeMeters).ceil().multiply(cellSizeMeters);
  var ymax = ymax_raw.divide(cellSizeMeters).ceil().multiply(cellSizeMeters);

  var xs = ee.List.sequence(xmin, xmax, cellSizeMeters);
  var ys = ee.List.sequence(ymin, ymax, cellSizeMeters);

  var cells = xs.map(function(x) {
    x = ee.Number(x);
    return ys.map(function(y) {
      y = ee.Number(y);
      var rect = ee.Geometry.Rectangle(
        [x, y, x.add(cellSizeMeters), y.add(cellSizeMeters)],
        WORK_CRS,
        false
      );
      return ee.Feature(rect, { x0: x, y0: y });
    });
  }).flatten();

  return ee.FeatureCollection(cells).filterBounds(geometry3857);
}

// --------------------
// HELPER: safe property fetch
// --------------------
function getNumberOr0(feature, key) {
  var v = feature.get(key);
  return ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(v, null), 0, v));
}

// --------------------
// HELPER: normalize summed embedding feature
// --------------------
function normalizeSummedFeature(f) {
  var vals = bands.map(function(b) {
    return getNumberOr0(f, ee.String(b));
  });

  var arr = ee.Array(vals);

  var norm = ee.Number(
    arr.pow(2).reduce(ee.Reducer.sum(), [0]).get([0])
  ).sqrt();

  var unitArr = ee.Array(
    ee.Algorithms.If(
      norm.gt(0),
      arr.divide(norm),
      arr
    )
  );

  var dict = ee.Dictionary.fromLists(bands, unitArr.toList());

  return f.set(dict)
    .set({
      norm_before: norm,
      valid: norm.gt(0)
    });
}

// --------------------
// 1) ORIGINAL AEF EMBEDDING COUNT IN DC
// --------------------
var originalCount = ee.Number(
  srcCountBand.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: dcGeom,
    scale: SAMPLE_SCALE_M,
    crs: WORK_CRS,       
    maxPixels: 1e13,
    tileScale: TILE_SCALE
  }).get('src_count')
);

print('Original embedding count (DC):', originalCount);

// --------------------
// 2) BUILD BATCH GRID
// --------------------
var batchGrid = makeGrid3857(dcGeom3857, BATCH_SCALE_M);
print('Batch tile count:', batchGrid.size());

// --------------------
// 3) PROCESS ONE BATCH
// --------------------
function processOneBatch(batchFeature) {
  var cellGrid = makeGrid3857(ee.Feature(batchFeature).geometry(), AGG_SCALE_M);
  var validCells = cellGrid.filterBounds(dcGeom3857);

  var reduced = imgAll.reduceRegions({
    collection: validCells,
    reducer: ee.Reducer.sum(),
    scale: SAMPLE_SCALE_M,
    crs: WORK_CRS,
    tileScale: TILE_SCALE
  });

  var normalized = ee.FeatureCollection(reduced.map(normalizeSummedFeature))
    .filter(ee.Filter.gt('src_count', 0));

  return normalized;
}

// --------------------
// 4) RUN ALL BATCHES (Optimized via Map/Flatten)
// --------------------
var batchList = batchGrid.toList(batchGrid.size());

var pyramidFC = ee.FeatureCollection(
  batchList.map(function(batchFeature) {
    return processOneBatch(batchFeature);
  })
).flatten();

// --------------------
// 5) FINAL COUNTS
// --------------------
var pyramidCount = pyramidFC.size();

var totalSourcePixels = ee.Number(pyramidFC.aggregate_sum('src_count'));
var avgPixelsPer100mCell = totalSourcePixels.divide(pyramidCount);

// --------------------
// EXPORT
// --------------------
Export.table.toDrive({
  collection: pyramidFC,
  description: 'DC_PYRAMID_70m_moasic_2_' + YEAR,
  fileFormat: 'CSV'
});