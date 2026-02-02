# app.py - Fixed version for Render deployment
# -*- coding: utf-8 -*-
"""
STBG Project Prioritization API
Fixed for Render.com deployment
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from functools import reduce
import warnings
import os
import tempfile
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

warnings.filterwarnings('ignore')

app = FastAPI(title="STBG Project Prioritization API", version="1.0.0")

origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "https://stbg-projects-highway-py.netlify.app",
    "https://stbg.onrender.com",
    "https://stbg-projects-highway-py-production.netlify.app",
    "https://stbg-projects-highway-test.netlify.app",
    "https://stbg-prioritization.onrender.com",  # Add your Render URL
    "*"  # Allow all for testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisResults(BaseModel):
    projects: List[Dict[str, Any]]
    summary: Dict[str, Any]

# =============================================================================
# 1. SAFETY - CRASH FREQUENCY
# =============================================================================

def analyze_safety_frequency(projects_gdf, crashes_gdf):
    """Analyze safety based on crash frequency within project buffers"""
    
    # Load data
    gdf = projects_gdf.copy()
    crashes = crashes_gdf.copy()
    
    # Ensure projected CRS for buffer
    gdf = gdf.to_crs(epsg=2263)  # feet
    crashes = crashes.to_crs(gdf.crs)
    
    # Add project_id if not present
    if "project_id" not in gdf.columns:
        gdf["project_id"] = range(1, len(gdf) + 1)
    
    # Create 250 ft buffer
    gdf_buffered = gdf.copy()
    gdf_buffered["geometry"] = gdf_buffered.geometry.buffer(250)
    
    # Select crashes that intersect buffer
    crashes_in_buffer = gpd.sjoin(
        crashes,
        gdf_buffered[["project_id", "geometry"]],
        how="inner",
        predicate="intersects"
    )
    
    # Summarize crash people counts per buffer
    crash_sums = crashes_in_buffer.groupby("project_id").agg({
        "K_PEOPLE": "sum",
        "A_PEOPLE": "sum",
        "B_PEOPLE": "sum",
        "C_PEOPLE": "sum"
    }).reset_index()
    
    # Merge summary back to buffers
    gdf_buffered = gdf_buffered.merge(crash_sums, on="project_id", how="left")
    gdf_buffered.fillna(0, inplace=True)
    
    # Reproject back to WGS84 for saving/visualization
    gdf_buffered = gdf_buffered.to_crs(epsg=4326)
    
    # Calculate EPDO
    gdf_buffered["EPDO"] = (
        gdf_buffered["K_PEOPLE"] * 2715000 +
        gdf_buffered["A_PEOPLE"] * 2715000 +
        gdf_buffered["B_PEOPLE"] * 300000 +
        gdf_buffered["C_PEOPLE"] * 170000
    )
    
    # Calculate benefit = EPDO * (1 - cmf)
    gdf_buffered["benefit"] = gdf_buffered["EPDO"] * (1 - gdf_buffered["cmf"])
    
    # Calculate safety score
    max_benefit = gdf_buffered["benefit"].max()
    
    # Avoid divide by zero
    if max_benefit > 0:
        gdf_buffered["safety_freq"] = (gdf_buffered["benefit"] / max_benefit) * 50
    else:
        gdf_buffered["safety_freq"] = 0
    
    safety_freq = gdf_buffered[['project_id', 'safety_freq']]
    return safety_freq

# =============================================================================
# 2. SAFETY - CRASH RATE
# =============================================================================

def analyze_safety_rate(projects_gdf, crashes_gdf):
    """Analyze safety based on crash rate (normalized by traffic volume)"""
    
    # Load and process data
    gdf = projects_gdf.copy()
    crashes = crashes_gdf.copy()
    
    # Process data similar to safety frequency analysis
    gdf_buffered = gdf.to_crs(epsg=2263)
    crashes = crashes.to_crs(gdf.crs)
    
    if "project_id" not in gdf.columns:
        gdf["project_id"] = range(1, len(gdf) + 1)
    
    gdf_buffered = gdf.copy()
    gdf_buffered["geometry"] = gdf_buffered.geometry.buffer(250)
    
    crashes_in_buffer = gpd.sjoin(
        crashes,
        gdf_buffered[["project_id", "geometry"]],
        how="inner",
        predicate="intersects"
    )
    
    crash_sums = crashes_in_buffer.groupby("project_id").agg({
        "K_PEOPLE": "sum",
        "A_PEOPLE": "sum",
        "B_PEOPLE": "sum",
        "C_PEOPLE": "sum"
    }).reset_index()
    
    gdf_buffered = gdf_buffered.merge(crash_sums, on="project_id", how="left")
    gdf_buffered.fillna(0, inplace=True)
    gdf_buffered = gdf_buffered.to_crs(epsg=4326)
    
    gdf_buffered["EPDO"] = (
        gdf_buffered["K_PEOPLE"] * 2715000 +
        gdf_buffered["A_PEOPLE"] * 2715000 +
        gdf_buffered["B_PEOPLE"] * 300000 +
        gdf_buffered["C_PEOPLE"] * 170000
    )
    
    gdf_buffered["benefit"] = gdf_buffered["EPDO"] * (1 - gdf_buffered["cmf"])
    
    # Define epdo_rate based on project type
    def calculate_epdo_rate_and_vmt(row):
        if row["type"].lower() == "highway":
            vmt = row["AADT"] * row["length"] * 365 / 100_000_000
        elif row["type"].lower() == "intersection":
            vmt = row["AADT"] * 365 / 1_000_000
        else:
            vmt = 1  # avoid division by zero
        epdo_rate = row["benefit"] / vmt if vmt != 0 else 0
        return pd.Series({"VMT": vmt, "epdo_rate": epdo_rate})
    
    # Apply function
    gdf_buffered[["VMT", "epdo_rate"]] = gdf_buffered.apply(calculate_epdo_rate_and_vmt, axis=1)
    
    # Calculate safety_rate
    max_rate = gdf_buffered["epdo_rate"].max()
    
    # Avoid division by zero
    if max_rate > 0:
        gdf_buffered["safety_rate"] = (gdf_buffered["epdo_rate"] / max_rate) * 50
    else:
        gdf_buffered["safety_rate"] = 0
    
    safety_rate = gdf_buffered[['project_id', 'safety_rate']]
    return safety_rate

# =============================================================================
# 3. CONGESTION - DEMAND
# =============================================================================

def analyze_congestion_demand(projects_gdf, aadt_gdf):
    """Analyze congestion based on traffic demand"""
    
    # Load projects
    projects = projects_gdf.copy()
    
    # Load AADT segments
    aadt = aadt_gdf.copy()
    
    # Ensure both are in the same CRS
    projects = projects.to_crs(epsg=2283)  # Virginia State Plane
    aadt = aadt.to_crs(projects.crs)
    
    buffer_distance = 0.25 * 1609.34  # meters
    projects["buffer"] = projects.geometry.buffer(buffer_distance)
    
    # Convert project buffers to GeoDataFrame
    project_buffers = projects.set_geometry("buffer")
    
    # Perform spatial join
    intersected = gpd.sjoin(aadt, project_buffers, how="inner", predicate="intersects")
    
    intersected["segment_mileage"] = intersected.geometry.length / 1609.34  # meters → miles
    intersected["vmt"] = intersected["aadt_0"] * intersected["segment_mileage"]
    
    wa_aadt = (
        intersected.groupby("project_id")
        .apply(lambda x: x["vmt"].sum() / x["segment_mileage"].sum())
        .reset_index(name="wa_aadt")
    )
    
    # Drop any existing wa_aadt columns to avoid _x/_y
    projects = projects.drop(columns=[col for col in projects.columns if "wa_aadt" in col], errors="ignore")
    
    # Merge the computed wa_aadt
    projects = projects.merge(wa_aadt, on="project_id", how="left")
    
    # Replace NaN with 0
    projects["wa_aadt"] = projects["wa_aadt"].fillna(0)
    
    # Normalize
    projects["cong_demand"] = (projects["wa_aadt"] / projects["wa_aadt"].max()) * 10
    projects = projects[['project_id', 'cong_demand']]
    
    return projects

# =============================================================================
# 4. CONGESTION - LEVEL OF SERVICE
# =============================================================================

def analyze_congestion_los(projects_gdf, aadt_gdf):
    """Analyze congestion based on Level of Service"""
    
    # Load project and AADT layers
    projects = projects_gdf.copy()
    aadt = aadt_gdf.copy()
    
    # Create cong_value column based on los_0
    los_mapping = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 3,
        "F": 3
    }
    
    aadt["cong_value"] = aadt["los_0"].map(los_mapping)
    
    # Create 0.25 mile buffer around project locations
    projects = projects.to_crs(epsg=3857)  # project to metric CRS
    aadt = aadt.to_crs(epsg=3857)
    
    projects["buffer"] = projects.geometry.buffer(402.336)
    
    # Intersect buffer and AADT segments
    projects_exploded = projects.explode(index_parts=False)
    intersected = gpd.overlay(aadt, gpd.GeoDataFrame(geometry=projects_exploded["buffer"]), how="intersection")
    
    # Sum cong_value for all segments in each project buffer
    intersected = intersected.merge(projects[["project_id", "buffer"]], left_on='geometry', right_on='buffer', how='left')
    
    project_cong = (
        intersected.groupby("project_id")["cong_value"]
        .sum()
        .reset_index(name="sum_cong_value")
    )
    
    # Merge back to projects
    projects = projects.merge(project_cong, on="project_id", how="left")
    projects["sum_cong_value"] = projects["sum_cong_value"].fillna(0)
    
    # Normalize
    normalized = (projects["sum_cong_value"] / projects["sum_cong_value"].max()) * 5
    # If indivisible (not integer), return 0
    projects["cong_los"] = normalized.where(normalized % 1 == 0, 0)
    
    projects = projects[['project_id', 'cong_los']]
    
    return projects

# =============================================================================
# 5. EQUITY/ACCESS - ACCESS TO JOBS
# =============================================================================

def analyze_equity_access_jobs(projects_gdf, popemp_gdf):
    """Analyze equity and access to jobs"""
    
    # Load datasets
    pop_emp_df = popemp_gdf.copy()
    projects = projects_gdf.copy()
    
    # Define buffer distances in meters
    fc_distances_miles = {"PA": 10, "MA": 7.5, "MC": 5}
    mile_to_meter = 1609.34
    fc_distances_m = {k: v * mile_to_meter for k, v in fc_distances_miles.items()}
    
    # Project both datasets to a projected CRS
    projects = projects.to_crs(epsg=2283)
    pop_emp_df = pop_emp_df.to_crs(epsg=2283)
    
    # Calculate TAZ centroids
    pop_emp_df["centroid"] = pop_emp_df.geometry.centroid
    
    results = []
    
    # For each project
    for _, proj in projects.iterrows():
        fc = proj["fc"]
        buffer_dist = fc_distances_m[fc]
        
        # Create buffer
        proj_buffer = proj.geometry.buffer(buffer_dist)
        
        # Select TAZs with centroid inside buffer
        selected = pop_emp_df[pop_emp_df["centroid"].within(proj_buffer)]
        
        # Aggregate employment
        sum_emp17 = selected["emp17"].sum()
        sum_emp50 = selected["emp50"].sum()
        
        # % change
        pct_change = ((sum_emp50 - sum_emp17) / sum_emp17 * 100) if sum_emp17 != 0 else 0
        
        results.append({
            "project_id": proj["project_id"],
            "sum_emp17": sum_emp17,
            "sum_emp50": sum_emp50,
            "pct_change": pct_change
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    results_df["jobs_pc"] = (results_df["pct_change"] / results_df["pct_change"].max()) * 5 if results_df["pct_change"].max() != 0 else 0
    
    results_df = results_df[['project_id', 'jobs_pc']]
    
    return results_df

# =============================================================================
# 6. EQUITY/ACCESS - ACCESS TO JOBS (ENVIRONMENTAL JUSTICE)
# =============================================================================

def analyze_equity_access_jobs_ej(projects_gdf, popemp_gdf, t6_gdf):
    """Analyze equity and access to jobs in environmental justice areas"""
    
    # Load datasets
    pop_emp_df = popemp_gdf.copy()
    ej = t6_gdf.copy()
    projects = projects_gdf.copy()
    
    # Distances in miles → meters
    fc_distances_miles = {"PA": 10, "MA": 7.5, "MC": 5}
    mile_to_meter = 1609.34
    fc_distances_m = {k: v * mile_to_meter for k, v in fc_distances_miles.items()}
    
    # Project all layers to a projected CRS
    projects = projects.to_crs(epsg=2283)
    pop_emp_df = pop_emp_df.to_crs(epsg=2283)
    ej = ej.to_crs(epsg=2283)
    
    results = []
    
    # Loop through projects
    for _, proj in projects.iterrows():
        fc = proj["fc"]
        buffer_dist = fc_distances_m[fc]
        
        # Create buffer
        proj_buffer = proj.geometry.buffer(buffer_dist)
        
        # Clip EJ polygons within buffer
        ej_clip = ej[ej.intersects(proj_buffer)]
        
        if ej_clip.empty:
            sum_emp17 = 0
            sum_emp50 = 0
        else:
            # Clip TAZ by EJ polygons (intersection)
            taz_ej_intersect = gpd.overlay(pop_emp_df, ej_clip, how="intersection")
            
            # Aggregate employment in intersected areas
            sum_emp17 = taz_ej_intersect["emp17"].sum()
            sum_emp50 = taz_ej_intersect["emp50"].sum()
        
        # % change
        pct_change = ((sum_emp50 - sum_emp17) / sum_emp17 * 100) if sum_emp17 != 0 else 0
        
        results.append({
            "project_id": proj["project_id"],
            "sum_emp17": sum_emp17,
            "sum_emp50": sum_emp50,
            "pct_change": pct_change
        })
    
    # Results DataFrame
    results_df = pd.DataFrame(results)
    
    # Normalize percent change
    results_df["jobs_pc_ej"] = (results_df["pct_change"] / results_df["pct_change"].max()) * 5 if results_df["pct_change"].max() != 0 else 0
    results_df = results_df[['project_id', 'jobs_pc_ej']]
    
    return results_df

# =============================================================================
# 7. ACCESS TO NON-WORK DESTINATIONS
# =============================================================================

def analyze_access_non_work(projects_gdf, popemp_gdf, nw_gdf):
    """Analyze access to non-work destinations"""
    
    # Load datasets
    pop_emp_df = popemp_gdf.copy()
    nw = nw_gdf.copy()
    projects = projects_gdf.copy()
    
    # Distances (miles -> meters)
    fc_distances_miles = {"PA": 10, "MA": 7.5, "MC": 5}
    mile_to_meter = 1609.34
    fc_distances_m = {k: v * mile_to_meter for k, v in fc_distances_miles.items()}
    
    # Project to planar CRS
    projects = projects.to_crs(epsg=2283)
    nw = nw.to_crs(epsg=2283)
    pop_emp_df = pop_emp_df.to_crs(epsg=2283)
    
    results = []
    
    for _, proj in projects.iterrows():
        fc = proj["fc"]
        buffer_dist = fc_distances_m[fc]
        
        # Buffer project
        proj_buffer = proj.geometry.buffer(buffer_dist)
        
        # Count NW points inside buffer
        nw_count = nw[nw.within(proj_buffer)].shape[0]
        
        # Intersect TAZs with buffer
        taz_selected = pop_emp_df[pop_emp_df.intersects(proj_buffer)]
        
        if taz_selected.empty:
            sum_emp2017 = sum_emp2050 = sum_pop2017 = sum_pop2050 = area_sqmi = 0
        else:
            sum_emp2017 = taz_selected["emp17"].sum()
            sum_emp2050 = taz_selected["emp50"].sum()
            sum_pop2017 = taz_selected["pop17"].sum()
            sum_pop2050 = taz_selected["pop50"].sum()
            
            # area in square miles
            area_sqmi = taz_selected.to_crs(epsg=3857).geometry.area.sum() / (1609.34**2)
        
        # Calculate density metrics
        if area_sqmi > 0:
            pop_emp_den_2017 = nw_count * (sum_emp2017 + sum_pop2017) / area_sqmi
            pop_emp_den_2050 = nw_count * (sum_emp2050 + sum_pop2050) / area_sqmi
        else:
            pop_emp_den_2017 = pop_emp_den_2050 = 0
        
        results.append({
            "project_id": proj["project_id"],
            "nw_count": nw_count,
            "sum_emp2017": sum_emp2017,
            "sum_emp2050": sum_emp2050,
            "sum_pop2017": sum_pop2017,
            "sum_pop2050": sum_pop2050,
            "area_sqmi": area_sqmi,
            "pop_emp_den_2017": pop_emp_den_2017,
            "pop_emp_den_2050": pop_emp_den_2050
        })
    
    # Results dataframe
    results_df = pd.DataFrame(results)
    
    # Percent change in pop_emp_den
    results_df["access_nw_pct"] = (
        ((results_df["pop_emp_den_2050"] - results_df["pop_emp_den_2017"]) / results_df["pop_emp_den_2017"] * 100)
        .fillna(0)  # handle division by zero
    )
    
    # Normalize percent change to 0–5 scale
    results_df["access_nw_norm"] = (
        (results_df["access_nw_pct"] / results_df["access_nw_pct"].max() * 5) 
        if results_df["access_nw_pct"].max() != 0 else 0
    )
    
    results_df = results_df[['project_id', 'access_nw_norm']]
    
    return results_df

# =============================================================================
# 8. ACCESS TO NON-WORK DESTINATIONS (ENVIRONMENTAL JUSTICE)
# =============================================================================

def analyze_access_non_work_ej(projects_gdf, popemp_gdf, nw_gdf, t6_gdf):
    """Analyze access to non-work destinations in environmental justice areas"""
    
    # Load datasets
    pop_emp_df = popemp_gdf.copy()
    nw = nw_gdf.copy()
    t6 = t6_gdf.copy()
    projects = projects_gdf.copy()
    
    # Distances (miles -> meters)
    fc_distances_miles = {"PA": 10, "MA": 7.5, "MC": 5}
    mile_to_meter = 1609.34
    fc_distances_m = {k: v * mile_to_meter for k, v in fc_distances_miles.items()}
    
    # Project to planar CRS
    projects = projects.to_crs(epsg=2283)
    nw = nw.to_crs(epsg=2283)
    pop_emp_df = pop_emp_df.to_crs(epsg=2283)
    t6 = t6.to_crs(epsg=2283)
    
    results = []
    
    for _, proj in projects.iterrows():
        fc = proj["fc"]
        buffer_dist = fc_distances_m[fc]
        
        # Buffer project
        proj_buffer = proj.geometry.buffer(buffer_dist)
        
        # Count NW points inside buffer AND inside T6 polygon
        nw_count = nw[nw.within(proj_buffer) & nw.within(t6.unary_union)].shape[0]
        
        # Intersect TAZs with buffer AND T6 polygon
        taz_selected = pop_emp_df[pop_emp_df.intersects(proj_buffer) & pop_emp_df.intersects(t6.unary_union)]
        
        if taz_selected.empty:
            sum_emp2017 = sum_emp2050 = sum_pop2017 = sum_pop2050 = area_sqmi = 0
        else:
            sum_emp2017 = taz_selected["emp17"].sum()
            sum_emp2050 = taz_selected["emp50"].sum()
            sum_pop2017 = taz_selected["pop17"].sum()
            sum_pop2050 = taz_selected["pop50"].sum()
            
            # area in square miles
            area_sqmi = taz_selected.to_crs(epsg=3857).geometry.area.sum() / (1609.34**2)
        
        # Calculate density metrics
        if area_sqmi > 0:
            pop_emp_den_2017 = nw_count * (sum_emp2017 + sum_pop2017) / area_sqmi
            pop_emp_den_2050 = nw_count * (sum_emp2050 + sum_pop2050) / area_sqmi
        else:
            pop_emp_den_2017 = pop_emp_den_2050 = 0
        
        results.append({
            "project_id": proj["project_id"],
            "nw_count": nw_count,
            "sum_emp2017": sum_emp2017,
            "sum_emp2050": sum_emp2050,
            "sum_pop2017": sum_pop2017,
            "sum_pop2050": sum_pop2050,
            "area_sqmi": area_sqmi,
            "pop_emp_den_2017": pop_emp_den_2017,
            "pop_emp_den_2050": pop_emp_den_2050
        })
    
    # Results dataframe
    results_df = pd.DataFrame(results)
    
    # Percent change in pop_emp_den
    results_df["access_nw_pct"] = (
        ((results_df["pop_emp_den_2050"] - results_df["pop_emp_den_2017"]) / results_df["pop_emp_den_2017"] * 100)
        .fillna(0)  # handle division by zero
    )
    
    # Normalize percent change to 0–5 scale
    max_pct = results_df["access_nw_pct"].max()
    results_df["access_nw_ej_norm"] = (results_df["access_nw_pct"] / max_pct * 5) if max_pct != 0 else 0
    
    results_df = results_df[['project_id', 'access_nw_ej_norm']]
    
    return results_df

# =============================================================================
# 9. SENSITIVE FEATURES ANALYSIS
# =============================================================================

def analyze_sensitive_features(projects_gdf, fhz_gdf, frsk_gdf, wet_gdf, con_gdf):
    """Analyze impact on sensitive environmental features"""
    
    # Load sensitive feature datasets
    fhz = fhz_gdf.copy()
    frsk = frsk_gdf.copy()
    wet = wet_gdf.copy()
    con = con_gdf.copy()
    
    # Filter AE zones
    fhz_filtered = fhz[fhz['FLD_ZONE'] == 'AE']
    
    # Buffer flood risk areas
    gdf = frsk.copy()
    if frsk.crs is None or not frsk.crs.is_projected:
        gdf = frsk.to_crs('EPSG:2264')
    
    frsk_buff = frsk.buffer(200)
    frsk_buff = gpd.GeoDataFrame(geometry=frsk_buff, crs=frsk.crs)
    
    # Combine all sensitive areas
    gdf1 = fhz_filtered.copy()
    gdf2 = frsk_buff.copy()
    gdf3 = wet.copy()
    gdf4 = con.copy()
    
    # Ensure all have the same CRS
    target_crs = gdf1.crs
    gdf2 = gdf2.to_crs(target_crs)
    gdf3 = gdf3.to_crs(target_crs)
    gdf4 = gdf4.to_crs(target_crs)
    
    # Combine all GeoDataFrames
    combined_gdf = gpd.GeoDataFrame(pd.concat([gdf1, gdf2, gdf3, gdf4], ignore_index=True))
    
    # Dissolve all geometries into one
    sen_areas = combined_gdf.dissolve()
    
    # Load projects
    gdf = projects_gdf.copy()
    
    # Create ¼-mile buffer around project points
    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)
    
    # Create ¼-mile buffer (402.336 meters) around each project point
    buffer_distance_m = 402.336
    project_buffers = gdf_utm.buffer(buffer_distance_m)
    
    # Convert back to GeoDataFrame with original project attributes
    project_buffer_gdf = gpd.GeoDataFrame(
        gdf.drop(columns='geometry'),  # Keep all attributes except original geometry
        geometry=project_buffers,
        crs=utm_crs
    )
    project_buffer_gdf = project_buffer_gdf.to_crs(gdf.crs)
    
    # Perform intersection between project buffers and sensitive areas
    if project_buffer_gdf.crs != sen_areas.crs:
        sen_areas = sen_areas.to_crs(project_buffer_gdf.crs)
    
    sen_areas_proj_buff = gpd.overlay(project_buffer_gdf, sen_areas, how='intersection')
    
    # Compute area in square miles
    sen_areas_proj_buff["sen_area_sqmi"] = sen_areas_proj_buff.geometry.area / 2.59e6  # convert from m² to mi²
    
    cols = ["project_id", "type", "syip", "county", "cmf", "AADT", "length", "fc", "cost_mil", "tier", "sen_area_sqmi"]
    sen_areas_proj_buff = sen_areas_proj_buff[cols + ["geometry"]]
    
    # Calculate impact based on project tier
    def adjust_area(row):
        if row["tier"] == "CE":
            return row["sen_area_sqmi"] * 0.9
        elif row["tier"] == "EA":
            return row["sen_area_sqmi"] * 0.7
        elif row["tier"] == "EIS":
            return row["sen_area_sqmi"] * 0.5
        else:
            return row["sen_area_sqmi"]  # no reduction if tier not listed
    
    sen_areas_proj_buff["sen_impact"] = sen_areas_proj_buff.apply(adjust_area, axis=1)
    
    # Normalize environmental impact (lower impact is better, so we invert)
    max_impact = sen_areas_proj_buff["sen_impact"].max()
    if max_impact > 0:
        sen_areas_proj_buff["env_impact_score"] = 10 - (sen_areas_proj_buff["sen_impact"] / max_impact * 10)
    else:
        sen_areas_proj_buff["env_impact_score"] = 10
    
    # Cap at 0
    sen_areas_proj_buff["env_impact_score"] = sen_areas_proj_buff["env_impact_score"].clip(lower=0)
    
    env_scores = sen_areas_proj_buff[['project_id', 'env_impact_score']]
    
    return env_scores

# =============================================================================
# 10. JOB GROWTH ANALYSIS
# =============================================================================

def analyze_job_growth(projects_gdf, popemp_gdf):
    """Analyze job growth in project areas"""
    
    # Load datasets
    projects = projects_gdf.copy()
    pop_emp = popemp_gdf.copy()
    
    # Ensure same CRS
    if projects.crs != pop_emp.crs:
        pop_emp = pop_emp.to_crs(projects.crs)
    
    # Define buffer distances (in meters)
    mile_to_meter = 1609.34
    buffer_distances = {"PA": 10 * mile_to_meter, "MA": 7.5 * mile_to_meter, "MC": 5 * mile_to_meter}
    
    # Create buffers
    projects["buffer_dist"] = projects["fc"].map(buffer_distances)
    projects_buffer = projects.copy()
    projects_buffer["geometry"] = projects_buffer.geometry.buffer(projects_buffer["buffer_dist"])
    
    # Compute centroids of pop_emp features
    pop_emp_centroids = pop_emp.copy()
    pop_emp_centroids["geometry"] = pop_emp_centroids.geometry.centroid
    
    # Spatial join (points within buffers)
    joined = gpd.sjoin(
        pop_emp_centroids,
        projects_buffer[["project_id", "geometry"]],
        predicate="within"
    )
    
    # Aggregate stats by project_id
    agg_stats = (
        joined.groupby("project_id", as_index=False)
        .agg({
            "pop17": "sum",
            "pop50": "sum",
            "emp17": "sum",
            "emp50": "sum"
        })
    )
    
    # Compute job growth
    agg_stats["job_growth"] = agg_stats["emp50"] - agg_stats["emp17"]
    
    # Normalize job growth (0-10 scale)
    max_growth = agg_stats["job_growth"].max()
    if max_growth > 0:
        agg_stats["job_growth_score"] = (agg_stats["job_growth"] / max_growth) * 10
    else:
        agg_stats["job_growth_score"] = 0
    
    job_growth_scores = agg_stats[['project_id', 'job_growth_score']]
    
    return job_growth_scores

# =============================================================================
# 11. FREIGHT JOBS ACCESS
# =============================================================================

def analyze_freight_jobs(projects_gdf, lehd_gdf):
    """Analyze access to freight jobs"""
    
    # Load datasets
    projects = projects_gdf.copy()
    lehd = lehd_gdf.copy()
    
    # Ensure same CRS
    if projects.crs != lehd.crs:
        lehd = lehd.to_crs(projects.crs)
    
    # Define buffer distances (in meters)
    mile_to_meter = 1609.34
    buffer_distances = {"PA": 10 * mile_to_meter, "MA": 7.5 * mile_to_meter, "MC": 5 * mile_to_meter}
    
    # Create buffers
    projects["buffer_dist"] = projects["fc"].map(buffer_distances)
    projects_buffer = projects.copy()
    projects_buffer["geometry"] = projects_buffer.geometry.buffer(projects_buffer["buffer_dist"])
    
    # Compute centroids of LEHD points
    lehd_centroids = lehd.copy()
    lehd_centroids["geometry"] = lehd_centroids.geometry.centroid
    
    # Spatial join (points within buffers)
    joined = gpd.sjoin(
        lehd_centroids,
        projects_buffer[["project_id", "geometry"]],
        predicate="within"
    )
    
    # Aggregate stats by project_id
    agg_stats = (
        joined.groupby("project_id", as_index=False)
        .agg({
            "freight_jobs": "sum"
        })
    )
    
    # Normalize freight jobs (0-10 scale)
    max_freight = agg_stats["freight_jobs"].max()
    if max_freight > 0:
        agg_stats["freight_score"] = (agg_stats["freight_jobs"] / max_freight) * 10
    else:
        agg_stats["freight_score"] = 0
    
    freight_scores = agg_stats[['project_id', 'freight_score']]
    
    return freight_scores

# =============================================================================
# 12. ACTIVITY CENTERS PROXIMITY
# =============================================================================

def analyze_activity_centers(projects_gdf, actv_gdf):
    """Analyze proximity to activity centers"""
    
    # Load datasets
    projects = projects_gdf.copy()
    actv = actv_gdf.copy()
    
    # Ensure same CRS
    if projects.crs != actv.crs:
        actv = actv.to_crs(projects.crs)
    
    # Define buffer distances (in meters)
    mile_to_meter = 1609.34
    buffer_distances = {"PA": 10 * mile_to_meter, "MA": 7.5 * mile_to_meter, "MC": 5 * mile_to_meter}
    
    # Create buffers
    projects["buffer_dist"] = projects["fc"].map(buffer_distances)
    projects_buffer = projects.copy()
    projects_buffer["geometry"] = projects_buffer.geometry.buffer(projects_buffer["buffer_dist"])
    
    # Spatial join (points within buffers)
    joined = gpd.sjoin(
        actv,
        projects_buffer[["project_id", "geometry"]],
        predicate="within"
    )
    
    # Count points by project_id
    agg_counts = (
        joined.groupby("project_id")
        .size()
        .reset_index(name="actv_count")
    )
    
    # Normalize activity center count (0-10 scale)
    max_count = agg_counts["actv_count"].max()
    if max_count > 0:
        agg_counts["activity_score"] = (agg_counts["actv_count"] / max_count) * 10
    else:
        agg_counts["activity_score"] = 0
    
    activity_scores = agg_counts[['project_id', 'activity_score']]
    
    return activity_scores

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_analysis(files_dict: Dict[str, str], output_dir: str):
    """Main function to run all analyses and generate final ranking"""
    
    print("Starting highway projects analysis...")
    
    print("All required files found. Starting analysis...")

    # Load all data into GeoDataFrames
    try:
        projects_gdf = gpd.read_file(files_dict['projects'])
        crashes_gdf = gpd.read_file(files_dict['crashes'])
        aadt_gdf = gpd.read_file(files_dict['aadt'])
        popemp_gdf = gpd.read_file(files_dict['popemp'])
        actv_gdf = gpd.read_file(files_dict['actv'])
        con_gdf = gpd.read_file(files_dict['con']) # This now reads congestion.geojson
        fhz_gdf = gpd.read_file(files_dict['fhz'])
        frsk_gdf = gpd.read_file(files_dict['frsk'])
        lehd_gdf = gpd.read_file(files_dict['lehd'])
        nw_gdf = gpd.read_file(files_dict['nw'])
        t6_gdf = gpd.read_file(files_dict['t6'])
        wet_gdf = gpd.read_file(files_dict['wet'])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading geospatial data: {e}")

    # Add project_id if not present
    if "project_id" not in projects_gdf.columns:
        projects_gdf["project_id"] = range(1, len(projects_gdf) + 1)
    
    # Run all analyses
    print("1. Analyzing safety frequency...")
    safety_freq = analyze_safety_frequency(projects_gdf, crashes_gdf)
    
    print("2. Analyzing safety rate...")
    safety_rate = analyze_safety_rate(projects_gdf, crashes_gdf)
    
    print("3. Analyzing congestion demand...")
    cong_demand = analyze_congestion_demand(projects_gdf, aadt_gdf)
    
    print("4. Analyzing congestion level of service...")
    cong_los = analyze_congestion_los(projects_gdf, aadt_gdf)
    
    print("5. Analyzing equity and access to jobs...")
    eq_acc_jobs = analyze_equity_access_jobs(projects_gdf, popemp_gdf)
    
    print("6. Analyzing equity and access to jobs (EJ)...")
    eq_acc_jobs_ej = analyze_equity_access_jobs_ej(projects_gdf, popemp_gdf, t6_gdf)
    
    print("7. Analyzing access to non-work destinations...")
    eq_acc_nw = analyze_access_non_work(projects_gdf, popemp_gdf, nw_gdf)
    
    print("8. Analyzing access to non-work destinations (EJ)...")
    eq_acc_nw_ej = analyze_access_non_work_ej(projects_gdf, popemp_gdf, nw_gdf, t6_gdf)
    
    print("9. Analyzing sensitive features...")
    env_scores = analyze_sensitive_features(projects_gdf, fhz_gdf, frsk_gdf, wet_gdf, con_gdf)
    
    print("10. Analyzing job growth...")
    job_growth_scores = analyze_job_growth(projects_gdf, popemp_gdf)
    
    print("11. Analyzing freight jobs access...")
    freight_scores = analyze_freight_jobs(projects_gdf, lehd_gdf)
    
    print("12. Analyzing activity centers proximity...")
    activity_scores = analyze_activity_centers(projects_gdf, actv_gdf)
    
    # Combine all dataframes
    dfs = [
        safety_freq,
        safety_rate,
        cong_demand,
        cong_los,
        eq_acc_jobs,
        eq_acc_jobs_ej,
        eq_acc_nw,
        eq_acc_nw_ej,
        env_scores,
        job_growth_scores,
        freight_scores,
        activity_scores
    ]
    
    # Merge all the regular dataframes on 'project_id'
    merged_data_df = reduce(lambda left, right: pd.merge(left, right, on="project_id", how="outer"), dfs)
    
    # Extract the non-geometry data from the GeoDataFrame
    gdf_data = projects_gdf
    
    # Merge the combined data with the GeoDataFrame's attribute data
    final_attributes_df = pd.merge(merged_data_df, gdf_data, on="project_id", how="outer")
    
    # Join this final attribute table back to the original GeoDataFrame to get the geometry
    final_gdf = projects_gdf[['project_id', 'geometry']].merge(final_attributes_df, on='project_id', how='right')
    
    # Handle multiple geometry columns - drop any extra geometry columns
    geometry_columns = [col for col in final_gdf.columns if final_gdf[col].dtype == 'geometry']
    if len(geometry_columns) > 1:
        print(f"Found multiple geometry columns: {geometry_columns}")
        # Keep only the main geometry column, drop others
        for geom_col in geometry_columns[1:]:
            final_gdf = final_gdf.drop(columns=[geom_col])
        print(f"Keeping only '{geometry_columns[0]}' geometry column")
    
    # Fill NaN values with 0 for all score columns
    score_columns = [
        'safety_freq', 'safety_rate', 'cong_demand', 'cong_los',
        'jobs_pc', 'jobs_pc_ej', 'access_nw_norm', 'access_nw_ej_norm',
        'env_impact_score', 'job_growth_score', 'freight_score', 'activity_score'
    ]
    
    for col in score_columns:
        if col in final_gdf.columns:
            final_gdf[col] = final_gdf[col].fillna(0)
    
    # Calculate the total benefit score (sum of all normalized scores)
    final_gdf['total_score'] = final_gdf[score_columns].sum(axis=1)
    
    # Calculate the Benefit-Cost Ratio (BCR)
    final_gdf['bcr'] = final_gdf['total_score'] / final_gdf['cost_mil']
    
    # Rank the projects based on the BCR (higher BCR is better)
    final_gdf['rank'] = final_gdf['bcr'].rank(ascending=False, method='dense').astype(int)
    
    # Create comprehensive results summary
    results_columns = ['project_id', 'type', 'county', 'cost_mil', 'tier'] + score_columns + ['total_score', 'bcr', 'rank']
    
    # Only include columns that exist in the dataframe
    available_columns = [col for col in results_columns if col in final_gdf.columns]
    
    results_df = final_gdf[available_columns].sort_values('rank')
    
    print("\n" + "="*80)
    print("FINAL PROJECT RANKINGS")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total projects analyzed: {len(results_df)}")
    print(f"Total score range: {results_df['total_score'].min():.2f} - {results_df['total_score'].max():.2f}")
    print(f"BCR range: {results_df['bcr'].min():.3f} - {results_df['bcr'].max():.3f}")
    print(f"Average cost: ${results_df['cost_mil'].mean():.2f}M")
    
    # Save results
    # Save the main results as a CSV file.
    results_csv_path = os.path.join(output_dir, "stbg_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"✓ Saved stbg_results.csv to {output_dir}")
    results_df.to_csv(os.path.join(output_dir, "project_rankings.csv"), index=False)
    print(f"✓ Saved project_rankings.csv to {output_dir}")

    # Convert to serializable format for JSON response
    results_list = results_df.to_dict(orient='records')
    
    # Clean up NaN/inf values for JSON serialization
    for project in results_list:
        for key, value in project.items():
            if pd.isna(value) or value == float('inf') or value == float('-inf'):
                project[key] = None # or 0, depending on desired representation

    summary = {
        "total_projects": len(results_list),
        "total_cost": final_gdf['cost_mil'].sum()
    }

    return {"projects": results_list, "summary": summary}

@app.get("/")
def read_root():
    return {"message": "Welcome to the STBG Project Prioritization API"}

@app.post("/analyze", response_model=AnalysisResults)
async def analyze_projects(
    projects_file: UploadFile = File(...),
    crashes_file: UploadFile = File(...),
    aadt_file: UploadFile = File(...),
    popemp_file: UploadFile = File(...),
    actv_file: UploadFile = File(...),
    con_file: UploadFile = File(...),
    fhz_file: UploadFile = File(...),
    frsk_file: UploadFile = File(...),
    lehd_file: UploadFile = File(...),
    nw_file: UploadFile = File(...),
    t6_file: UploadFile = File(...),
    wet_file: UploadFile = File(...),
):
    # Define the output directory relative to this script's location.
    # The script is in `public/stbg_elijah`, so we go up one level to get to `public`.
    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        files_dict = {}
        try:
            # Map incoming files to the keys used in the analysis functions
            all_files = locals()
            file_uploads = {
                "projects": projects_file, "crashes": crashes_file, "aadt": aadt_file,
                "popemp": popemp_file, "actv": actv_file, "con": con_file,
                "fhz": fhz_file, "frsk": frsk_file, "lehd": lehd_file,
                "nw": nw_file, "t6": t6_file, "wet": wet_file
            }

            for key, upload_file in file_uploads.items():
                file_path = os.path.join(temp_dir, upload_file.filename)
                with open(file_path, "wb") as f:
                    content = await upload_file.read()
                    f.write(content)
                files_dict[key] = file_path

            # Run the main analysis function with the paths to the temporary files
            results = run_analysis(files_dict, output_dir)
            
            if not results or "projects" not in results:
                 raise HTTPException(status_code=500, detail="Analysis failed to produce results.")

            return results

        except HTTPException as he:
            raise he
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# =============================================================================
# EXECUTION - FIXED FOR RENDER
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")
