# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:21:15 2025

@author: AlxndrLhrNc
"""
from antimeridian import fix_polygon, fix_multi_polygon
from cartopy import crs as ccrs, feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import natural_earth
from cartopy.mpl.geoaxes import GeoAxes
from geopandas import GeoDataFrame, read_file
from math import sqrt, cos, pi, fabs
from matplotlib import pyplot as plt, ticker as mticker, path as mpath
from matplotlib.axes._axes import Axes
from numpy import array, full, linspace, concatenate
from parcels import ParcelsRandom
from parcels import JITParticle, ScipyParticle, FieldSet
from typing import Union

# %% Zone class ############################################################
class Zone():
    """
    Class to handle geographical zones and mapping functionalities.
    This class allows for the definition of a geographical zone by its
    minimum and maximum longitudes and latitudes, and provides methods for
    creating base maps, cutting corners of data, and projecting boundaries.
    It also includes functionalities for reprojection and handling different
    scales of map features.
    """
#### Initiate and key finctions
    def __init__(self, minlon : float, maxlon : float, minlat : float, maxlat : float, projection:ccrs.Projection = ccrs.PlateCarree()):
        """Initialize the Zone with minimum and maximum longitudes and latitudes,
        and an optional projection.
        
        Args:
            minlon (float): Minimum longitude of the zone.
            maxlon (float): Maximum longitude of the zone.
            minlat (float): Minimum latitude of the zone.
            maxlat (float): Maximum latitude of the zone.
            projection (ccrs.Projection, optional): Cartopy projection to use. Defaults to PlateCarree.
        """
        self.minlon, self.maxlon, self.minlat, self.maxlat = minlon, maxlon, minlat, maxlat
        self.proj = projection
        self.transform = ccrs.PlateCarree() # data transformation helper
        self.lon_lat_ratio = round((maxlat-minlat)/(maxlon-minlon),2) # colorbar ratio helper
        self._valid_scales = ["auto", "coarse", "low", "intermediate", "high", "full"] # scale helper
        self.zorder_dict = {"land":10, "grid":99} # zorder helper
    
    def _reproj(self, projection:ccrs.Projection = None) -> tuple:
        """
        Reproject the zone boundaries to the specified projection.
        
        Args:
            projection (ccrs.Projection, optional): Cartopy projection to use. Defaults to None, which uses the class's projection.
        Returns:
            tuple (minlon, maxlon, minlat, maxlat): Reprojected minimum and maximum longitudes and latitudes.
        """
        projection = projection or self.proj# if projection is None else projection

        [minlon, minlat, _], [maxlon, maxlat, _] = projection.transform_points(
        ccrs.PlateCarree(), array([self.minlon, self.maxlon]), array([self.minlat, self.maxlat])
        )
        return float(minlon), float(maxlon), float(minlat), float(maxlat)
    
    def _projected_boundary(self, projection:ccrs.Projection = ccrs.PlateCarree(), n:int = 20) -> mpath.Path:
        """
        Create a projected boundary path for the zone.
        
        Args:
            n (int, optional): Number of points to interpolate along the boundary. Defaults to 20.
            projection (ccrs.Projection, optional): Cartopy projection to use. Defaults to PlateCarree.
        Returns:
            mpath.Path: A path representing the boundary of the zone.
        """
        #boundary_path = mpath.Path(
        #    list(zip(linspace(self.minlon,self.maxlon, n), full(n, self.maxlat))) + \
        #        list(zip(full(n, self.maxlon), linspace(self.maxlat, self.minlat, n))) + \
        #            list(zip(linspace(self.maxlon, self.minlon, n), full(n, self.minlat))) + \
        #                list(zip(full(n, self.minlon), linspace(self.minlat, self.maxlat, n)))
        #                )

        lon = concatenate([
            linspace(self.minlon, self.maxlon, n),
            full(n, self.maxlon),
            linspace(self.maxlon, self.minlon, n),
            full(n, self.minlon)
            ])
        lat = concatenate([
            full(n, self.maxlat),
            linspace(self.maxlat, self.minlat, n),
            full(n, self.minlat),
            linspace(self.minlat, self.maxlat, n)
            ])

        # Project lon/lat to projection space (x, y) and create Path
        xy = projection.transform_points(ccrs.PlateCarree(), lon, lat)
        boundary_path =  mpath.Path(xy[:, :2], closed=True)

        return boundary_path

#### Mapping related
    def base_map(
    self,
    ax:Union[Axes, GeoAxes] = None,
    graticule: float = 0.5,
    projection: ccrs.Projection = ccrs.PlateCarree(central_longitude=180),
    scale: str ="auto",
    land_mask: bool = True,
    ocean_mask: bool = False
    ):
        """
        Create a base map for the defined zone with specified parameters.
        
        Args:
            ax (Axes, optional): axe on which the figure should be made
            graticule (float, optional): Spacing of the grid lines in degrees. Defaults to 0.5.
            projection (ccrs.Projection, optional): Cartopy projection to use. Defaults to PlateCarree with central longitude 180.
            scale (str, optional): Scale of the land mask. Defaults to "auto". Valid options are "auto", "coarse", "low", "intermediate", "high", and "full".
            land_mask (bool, optional): Whether to add a land mask to the map. Defaults to True.
        Returns:
            tuple[matplotlib.figure.Figure, cartopy.mpl.geoaxes.GeoAxes] : A tuple containing the figure and map axis.
        Raises:
            ValueError: If the provided scale is not in the list of accepted scales.
        """
        # Ensure valid scale
        if scale not in self._valid_scales:
            raise ValueError(
                f"Scale {scale} not in the list of accepted scales: {self._valid_scales}"
            )
        
        # Reprojecting zone boundaries
        minlon, maxlon, minlat, maxlat = self._reproj(projection=projection)

        # Ax & figure setup
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": projection})
        else:
            fig = ax.figure
            # Replace plain Axes with Geoaxes
            if not isinstance(ax, GeoAxes):
                pos = ax.get_position()
                ax.remove()
                ax = fig.add_subplot(pos, projection=projection)
        
        # Set map extent with provided projection
        ax.set_extent([minlon, maxlon, minlat, maxlat], crs=projection)
        
        # Match boundary shape to zone STILL CRASH
        try:
            ax.set_boundary(self._projected_boundary(projection=projection)) #, transform=ax.transData
        except Exception as e:
            print(f"WARNING: Could not set boundary du to error: {e}")
        #Values of transform I tried: ccrs.PlateCarree-FAIL, ax.transAxes-FAIL, projection-NOT WORKING, 

        # Add land mask
        if land_mask:
            ax.add_feature(
                cfeature.GSHHSFeature(scale=scale),
                facecolor="#ADAEB3",
                edgecolor="#393A3F",
                linewidth=0.5,
                zorder=self.zorder_dict["land"],
            )
        
        # Add ocean color
        if ocean_mask:
            ax.set_facecolor("#e8f1f9")

        # Add graticule
        grd = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            x_inline=False, 
            y_inline=False,
            linewidth=0.75,
            color="#55565b",
            linestyle="dotted",
        )
        grd.set_zorder(self.zorder_dict["grid"])
        grd.top_labels = False
        grd.right_labels = False
        grd.xformatter = LONGITUDE_FORMATTER
        grd.yformatter = LATITUDE_FORMATTER
        grd.xlocator = mticker.MultipleLocator(graticule)
        grd.ylocator = mticker.MultipleLocator(graticule)

        return fig, ax

    def get_country_polygon(self, name:str = "Iceland", scale:str = "10m"):
        """
        Find a way to implement
        """
        full_geom = cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', scale)
        
#### Data modification
    def cut_corners(self, data_nc):
        """
        Cut the corners of the data based on the defined zone boundaries.
        
        Args:
            data_nc (xarray.Dataset): The dataset containing the data to be cut.
        Returns:
            xarray.Dataset: A new dataset with the corners cut based on the zone boundaries.
        """
        minlat, maxlat = min(self.minlat, self.maxlat), max(self.minlat, self.maxlat)
        minlon, maxlon = min(self.minlon, self.maxlon), max(self.minlon, self.maxlon)
        new_nc = data_nc.sel(latitude=slice(minlat, maxlat)).sel(
            longitude=slice(minlon, maxlon)
        )
        if new_nc.latitude.size == 0:
            new_nc = data_nc.sel(latitude=slice(maxlat, minlat)).sel(
                longitude=slice(minlon, maxlon)
            )
        return new_nc

#%% Particles advections ############################################################
# Smagorinsky diffusion kernel
ParticleType = Union[JITParticle, ScipyParticle]

def smagdiff(particle:ParticleType, fieldset:FieldSet, time:float) -> None:
    """
    Apply Smagorinsky-type subgrid turbulent diffusion to particle motion.

    Computes local velocity gradients using central differences and derives
    a local diffusivity (Kh) following the Smagorinsky parameterization:
        Kh = Cs * A * sqrt(dudx² + 0.5*(dudy + dvdx)² + dvdy²)

    Args:
        particle (parcels.particle.JITParticle or parcels.particle.ScipyParticle): Lagrangian particle being advected.
        fieldset (parcels.fieldset.FieldSet): fieldset containing ocean velocity, area, and Smagorinsky coefficient.
            Must include: U, V, cell_areas, and Cs.
        time (float): current model time in seconds.
    """
    # Step size for finite-difference gradient approximation (in degrees)
    dx = 0.01
    
    # velocity gradients computed using local central difference.
    updx, vpdx = fieldset.UV[time, particle.depth, particle.lat, particle.lon + dx]
    umdx, vmdx = fieldset.UV[time, particle.depth, particle.lat, particle.lon - dx]
    updy, vpdy = fieldset.UV[time, particle.depth, particle.lat + dx, particle.lon]
    umdy, vmdy = fieldset.UV[time, particle.depth, particle.lat - dx, particle.lon]

    dudx = (updx - umdx) / (2 * dx)
    dudy = (updy - umdy) / (2 * dx)

    dvdx = (vpdx - vmdx) / (2 * dx)
    dvdy = (vpdy - vmdy) / (2 * dx)

    # Compute local Smagorinsky diffusivity (Kh)
    A_deg2 = fieldset.cell_areas[time, 0, particle.lat, particle.lon]
    sq_deg_to_sq_m = (1852 * 60) ** 2 * cos(particle.lat * pi / 180)
    A_m2 = A_deg2 / sq_deg_to_sq_m
    Kh = fieldset.Cs * A_m2 * sqrt(dudx**2 + 0.5 * (dudy + dvdx) ** 2 + dvdy**2)

    # Apply random displacement based on Kh
    dlat = ParcelsRandom.normalvariate(0.0, 1.0) * sqrt(
        2 * fabs(particle.dt) * Kh
    )
    dlon = ParcelsRandom.normalvariate(0.0, 1.0) * sqrt(
        2 * fabs(particle.dt) * Kh
    )

    particle_dlat += dlat
    particle_dlon += dlon

# Wind interraction kernel
def WindInteraction(particle:ParticleType, fieldset:FieldSet, time:float) -> None:
    """
    Apply a windage correction to particle motion.

    A fraction of the wind velocity is added to the particle's motion, scaled
    by a constant "wind factor" derived empirically or theoretically.

    Args:
        particle (parcels.particle.JITParticle or parcels.particle.ScipyParticle): Lagrangian particle being advected.
        fieldset (parcels.fieldset.FieldSet): fieldset containing wind velocity.
            Must include: UWind, VWind
        time (float): current model time in seconds (unused but requiered by API).
    
    Refs:
        doi for wind_factor = 0.03: https://doi.org/10.1175/JPO-D-20-0275.1
        doi for Rwindage formula: https://doi.org/10.3390/jmse13040717
        Rwindage = sqrt((rho_air / rho_water) * (area_air / area_water) * (drag_coef_air / drag_coef_water))
      
        Typical Rwindage value for PacificGyre Carthe drifter (https://www.pacificgyre.com/carthe-drifter.aspx):
            rho_air = 1.225 kg/m^3
            rho_water = 1024.5 kg/m^3
            area_air = 0.0105 m2 estimated from drifter size assuming outside diameter≈0.35m; tube thickness≈0.08m
            area_water = 0.225 m2 estimated from drifter size s=0.42m
            drag_coef_air = 0.82 for long cylinder source https://en.wikipedia.org/wiki/Drag_coefficient
            drag_coef_water = 1.28 for flat plate source https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/shape-effects-on-drag/
    """

    wind_factor = 0.03 # adjust to control the influence of the wind 0.03
    # Alternatively, replace and fill the following formula
    #wind_factor = sqrt((1.225 / 1024.5) * (0.0105 / 0.225) * (0.82 / 1.28))

    particle_dlon += (
        fieldset.UWind[particle] * wind_factor * particle.dt
    )
    particle_dlat += (
        fieldset.VWind[particle] * wind_factor * particle.dt
    )

# delete-out-of-bound-partices kernel
def DeleteParticle(particle:ParticleType, fieldset:FieldSet, time:float) -> None:
    """
    Delete particles that have left the computational domain or returning any Error message.

    Args:
        particle (parcels.particle.JITParticle or parcels.particle.ScipyParticle): Lagrangian particle being advected.
        fieldset (parcels.fieldset.FieldSet): fieldset (unused but requiered by API).
        time (float): current model time in seconds (unused but requiered by API).
    """
    if particle.state >= 40: #== parcels.StatusCode.ErrorOutOfBounds:
        particle.delete()

#%% Utility function ############################################################
def longitude_shifter(longitude:float, mode:str = "360") -> float:
    """
    Change a longitude to [0, 360] or [-180, 180]

    Args:
        longitude (float): input longitude in degrees (DD).
        mode (str, optional): "360" for [0, 360] or "-180" for [-180, 180]. Defaults to "360".

    Raises:
        ValueError: If mode is not recognised.

    Returns:
        float: shifted longitude in degrees (DD).
    """
    if mode not in {"360", "-180"}:
        raise ValueError(f"Mode {mode} unknown. Must be either '360' ([0:360]) or '-180' ([-180:180]).")
    return longitude % 360 if mode == "360" else ((longitude - 180) % 360) - 180

def nmi_2_m(key:Union[str, int]) -> int:
    """
    Return nmi distance in km based on key or int

    Args:
        key (str or int): nautical distance key or int. Options are: 'default' (test only), '3nmi' (3 nautival miles), 'TerrSea' (territorial sea or 12 nmi), 'EEZ' (exclusive economic zone or 200 nmi)

    Raises:
        ValueError: if key is not valid

    Returns:
        int: nautical distances in km (rounded)
    """
    ref_dict = {"default": 500,
                "3nmi": round(3*1852),
                "TerrSea":round(12*1852),
                "EEZ":round(200*1852)
                }
    if isinstance(key, str) and (key not in ref_dict.keys()):
        raise ValueError(f"Key {key} not valid. Choose from {list(ref_dict.keys())}")
    elif isinstance(key, int):
        return key*1852
    else:
        return ref_dict[key]

def polygon_buffer(gdf:GeoDataFrame, buffer:int = 0, crs:Union[str, int] = None, fix_winding:bool = False) -> GeoDataFrame:
    """
    Apply a buffer to polygons in a GeoDataFrame with optional CRS transformation and polygon fixing.

    Args:
        gdf (GeoDataFrame, optional): Geodataframe contaninng the geometries to buff.
        buffer (int, optional): buffer distance in meter. Defaults to 0.
        crs (str or int, optional): CRS using metric unit, e.g. 3832 or 'EPSG:3832'. Defaults to None.
        fix_winding (bool, optional): whether to fix polygon winding order (see antimeridian library for more information). Defaults to False.

    Raises:
        ValueError: if buffer is requested but no metric unit crs is provided

    Returns:
        GeoDataFrame: geodataframe with buffered geometries 
    """
    if buffer and (crs is None):
        raise ValueError("Please provide a valid CRS when buffer > 0.")
    origin_crs = gdf.crs
    try:
        gdf_buffer = gdf.to_crs(crs).buffer(buffer).to_crs(origin_crs).apply(fix_polygon, fix_winding=fix_winding)
    except:
        gdf_buffer = gdf.to_crs(crs).buffer(buffer).to_crs(origin_crs).apply(fix_multi_polygon, fix_winding=fix_winding)
    return gdf_buffer

def polygon_list_gen(gdf:GeoDataFrame = None, countries:list = []) -> dict:
    """
    Generate a dictionary of country polygons from a GeoDataFrame.

    Args:
        gdf (GeoDataFrame, optional): GeoDataFrame of countries. If None, will try to load Natural Earth data.
        countries (list, optional): List of country names to include. Defaults to [].

    Raises:
        ValueError: if no GeoDataframe is provided and Natural Earth data cannot be loaded.
        ValueError: if no countries are provided.

    Returns:
        dict: dictionary mapping country name to GeoDataFrame like {'country': GeoDataFrame}
    """
    try: 
        gdf = gdf or read_file(natural_earth(resolution="10m", category="cultural", name="admin_0_countries"))
    except:
        raise ValueError("Please provide a GeoDataFrame or ensure cartopy installed.")
    
    if not any(countries):
        raise ValueError(f"Please provide a non-empty list of countries. Accepted countries are: {gdf['NAME'].unique().to_list()}")

    polygon_dict = {c: gdf.loc[gdf["NAME"]==c, "geometry"] for c in countries}
    return polygon_dict

