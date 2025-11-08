import numpy as np

import pyproj
from shapely.geometry import LineString
from shapely.ops import transform as shp_transform
import fiona
import matplotlib.tri as mtri
import rasterio as rio
import geopandas as gpd
import anuga
import os
import sys
from tqdm import tqdm

from utils.anuga_tools import anuga_tools as at
from utils import data_processing_tools as dpt

workshop_dir = os.getcwd()
data_dir = os.path.join(workshop_dir, 'data')
model_inputs_dir = os.path.join(workshop_dir, 'model_inputs')
model_outputs_dir = os.path.join(workshop_dir, 'model_outputs')
if 'google.colab' in sys.modules:
    data_dir = os.path.join(data_dir, 'collab')
    model_inputs_dir = os.path.join(model_inputs_dir, 'collab')
    model_outputs_dir = os.path.join(model_outputs_dir, 'collab')
model_visuals_dir = os.path.join(workshop_dir, 'visuals')
model_validation_dir = os.path.join(workshop_dir, 'validation')

f_DEM_tif = os.path.join(data_dir, 'DEM_MTI_PART.tif')
mesh_tri_shp = os.path.join(data_dir, "DEM_MTI_PART_USM.shp")  # your mesh triangles shapefile
pts_path  = os.path.join(data_dir, "mesh_mti_shp_pts.npy")
tris_path = os.path.join(data_dir, "mesh_mti_shp_tris.npy")

# --- Load upstream boundary line ---
f_US_BC = os.path.join(data_dir, 'ShMouth_US_BC.shp')
#us_gdf = gpd.read_file(f_US_BC)

# --- Load downstream boundary line ---
f_DS_BC = os.path.join(data_dir, 'Russel_DS_Boundary.shp')
#ds_gdf = gpd.read_file(f_DS_BC)

# ---------- helpers ----------
def finish_plot(path=None, dpi=150):
    if path:
        plt.savefig(path, bbox_inches="tight", dpi=dpi)
    elif not HEADLESS:
        plt.show()
    plt.close()

def reproject_points_xy(pts_xy: np.ndarray, src_crs, dst_crs):
    """Reproject Nx2 array of XY coordinates from src_crs -> dst_crs."""
    if src_crs is None or dst_crs is None:
        raise ValueError("CRS missing for reprojection")
    src = pyproj.CRS.from_user_input(src_crs)
    dst = pyproj.CRS.from_user_input(dst_crs)
    if src == dst:
        return pts_xy
    T = pyproj.Transformer.from_crs(src, dst, always_xy=True).transform
    x, y = T(pts_xy[:,0], pts_xy[:,1])
    out = pts_xy.copy()
    out[:,0] = x
    out[:,1] = y
    return out

def read_first_linestring_with_crs(path):
    with fiona.open(path) as src:
        crs = src.crs_wkt or src.crs
        for ft in src:
            g = ft["geometry"]
            if not g: continue
            if g["type"] == "LineString":
                return LineString(g["coordinates"]), crs
            if g["type"] == "MultiLineString":
                parts = [LineString(c) for c in g["coordinates"]]
                coords = [xy for part in parts for xy in part.coords]
                return LineString(coords), crs
    raise RuntimeError(f"No LineString in {path}")

def reproject_linestring(ls, src_crs, dst_crs):
    src = pyproj.CRS.from_user_input(src_crs)
    dst = pyproj.CRS.from_user_input(dst_crs)
    if src == dst:
        return ls
    T = pyproj.Transformer.from_crs(src, dst, always_xy=True).transform
    return shp_transform(T, ls)

# ---------- load your arrays ----------
pts  = np.load(pts_path,  allow_pickle=False)          # Nx2 float
tris = np.load(tris_path, allow_pickle=True)           # Mx3 int
if not np.issubdtype(tris.dtype, np.integer):
    tris = tris.astype(np.int64, copy=False)

# If you know the CRS of pts/tris (e.g., EPSG:26914 or 3158), set it here:
MESH_SRC_CRS = "EPSG:3158"   # <-- set to what your pts/tris are actually in

# Optional: read DS/US lines now so they plot in the same CRS as DEM
us_line_shp = os.path.join(data_dir, "ShMouth_US_BC.shp")
ds_line_shp = os.path.join(data_dir, "Russel_DS_Boundary.shp")
US_raw, US_crs = read_first_linestring_with_crs(us_line_shp)
DS_raw, DS_crs = read_first_linestring_with_crs(ds_line_shp)

# ---------- DEM overlay ----------
asc_dem = os.path.join(data_dir, os.path.basename(f_DEM_tif).replace('.tif', '_edited.asc'))
out_over_dem_png = os.path.join(data_dir, "mesh_over_dem_preview.png")

try:
    with rio.open(asc_dem) as r:
        dem = r.read(1, masked=True)
        dem_crs = r.crs  # rasterio CRS object
        bounds = r.bounds
        extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)

    # Reproject pts to DEM CRS if needed
    pts_plot = reproject_points_xy(pts, MESH_SRC_CRS, dem_crs)

    # Reproject lines to DEM CRS for visual QA
    US_plot = reproject_linestring(US_raw, US_crs, dem_crs)
    DS_plot = reproject_linestring(DS_raw, DS_crs, dem_crs)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=200)
    im = ax.imshow(dem, extent=extent, origin="upper")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Elevation")

    triang = mtri.Triangulation(pts_plot[:, 0], pts_plot[:, 1], triangles=tris)
    ax.triplot(triang, linewidth=0.25)

    # Draw US/DS lines (now in same CRS as DEM)
    ax.plot(*US_plot.xy, lw=1.5, label="US line")
    ax.plot(*DS_plot.xy, lw=1.5, label="DS line")

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Mesh over DEM (all in DEM CRS)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_over_dem_png, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_over_dem_png}")

except Exception as e:
    print(f"(Skipping DEM overlay: {e})")

# From here onward, CRS  of the DEM is used






#1) Domain built
#CHANGING TRIAGLES FROM Clock-Wise TO Counter CW

# 2) If triangles are object or float, sanitize to pure int
def sanitize_tris(arr):
    arr = np.asarray(arr)
    # If it's already a proper int array with shape (M,3), return
    if arr.ndim == 2 and arr.shape[1] == 3 and np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64, copy=False), {}

    issues = {}
    # If ragged (1D object), try to stack into (M,3)
    if arr.dtype == object and arr.ndim == 1:
        try:
            arr = np.vstack(arr)  # raise if ragged
        except Exception as e:
            raise ValueError(f"Triangles look ragged; cannot form (M,3): {e}")

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Triangles must be (M,3); got {arr.shape} and dtype={arr.dtype}")

    # Convert anything not int to float to detect NaN/inf, then to int
    if not np.issubdtype(arr.dtype, np.integer):
        arr_float = arr.astype(float, copy=False)
        bad_nan_inf = ~np.isfinite(arr_float).all(axis=1)
        if bad_nan_inf.any():
            issues["nan_inf_rows"] = int(bad_nan_inf.sum())
            arr = arr[~bad_nan_inf]
            arr_float = arr_float[~bad_nan_inf]
        arr = arr_float.astype(np.int64, copy=False)

    # Remove rows with negatives
    neg = (arr < 0).any(axis=1)
    if neg.any():
        issues["neg_index_rows"] = int(neg.sum())
        arr = arr[~neg]
    oob = (arr.max(axis=1) >= len(pts))
    if oob.any():
        issues["oob_index_rows"] = int(oob.sum())
        arr = arr[~oob]

    # Remove degenerate triangles
    deg = (arr[:,0]==arr[:,1]) | (arr[:,1]==arr[:,2]) | (arr[:,0]==arr[:,2])
    if deg.any():
        issues["degenerate_rows"] = int(deg.sum())
        arr = arr[~deg]

    # Drop exact duplicate triangles
    if arr.size:
        before = arr.shape[0]
        arr = np.unique(arr, axis=0)
        if arr.shape[0] != before:
            issues["duplicate_rows_removed"] = int(before - arr.shape[0])

    return arr, issues

tris, issues = sanitize_tris(tris)
print("Sanitized tris:", tris.shape, tris.dtype, "| issues:", issues)

# 3) Contiguity for ANUGA and flip to CCW
pts  = np.ascontiguousarray(pts,  dtype=np.float64)
tris = np.ascontiguousarray(tris, dtype=np.int64)

def signed_area2(p, t):
    a,b,c = p[t[0]], p[t[1]], p[t[2]]
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

flip = np.array([signed_area2(pts, t) < 0 for t in tris])
if flip.any():
    tmp = tris[flip, 1].copy()
    tris[flip, 1] = tris[flip, 2]
    tris[flip, 2] = tmp
    print(f"Flipped {flip.sum()} triangles to CCW")

# 4) Final assertions before ANUGA
assert tris.ndim == 2 and tris.shape[1] == 3
assert np.issubdtype(tris.dtype, np.integer)
assert (tris.min() >= 0) and (tris.max() < len(pts))


# In[21]:


# Check current dtypes 
print("pts:", getattr(pts, "dtype", None), "itemsize:", getattr(getattr(pts,"dtype",None), "itemsize", None))
print("tris:", getattr(tris, "dtype", None), "itemsize:", getattr(getattr(tris,"dtype",None), "itemsize", None))

# Force correct dtypes + contiguity 
pts  = np.ascontiguousarray(pts,  dtype=np.float64)   # coordinates in float64
tris = np.ascontiguousarray(tris, dtype=np.int64)     # triangles in int64

# If pandas DataFrame, convert .to_numpy:

# Checks
assert pts.ndim == 2 and pts.shape[1] == 2, f"pts must be (N,2), got {pts.shape}"
assert tris.ndim == 2 and tris.shape[1] == 3, f"tris must be (M,3), got {tris.shape}"
assert tris.dtype == np.int64, f"tris dtype must be int64, got {tris.dtype}"
assert pts.dtype  == np.float64, f"pts dtype must be float64, got {pts.dtype}"
assert tris.min() >= 0, "tris contains negative indices"
assert tris.max() < len(pts), "tris index exceeds number of points"

def signed_area2(p, t):
    a,b,c = p[t[0]], p[t[1]], p[t[2]]
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
flip = np.fromiter((signed_area2(pts,t) < 0 for t in tris), count=tris.shape[0], dtype=bool)
if flip.any():
    tris[flip, 1], tris[flip, 2] = tris[flip, 2].copy(), tris[flip, 1].copy()


np.save(os.path.join(model_inputs_dir, 'mesh_pts.npy'), pts)
np.save(os.path.join(model_inputs_dir, 'mesh_tris.npy'), tris)

print("Saved mesh pts/tris to model_inputs_dir")