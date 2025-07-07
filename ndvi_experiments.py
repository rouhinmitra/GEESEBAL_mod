#%%
import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
sys.path.append('D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\Code\\dT_changes\\geeSEBAL_copy_edits_dT\\etbrasil\\')
import geesebal
from geesebal import (tools_ndvi_adj, landsatcollection, masks, meteorology, endmembers, 
                     evapotranspiration, collection, timeseries, image, ET_Collection_mod)
#%%
class EndmemberConfig:
    def __init__(self, 
                 cold_ndvi_percentile=5,  # Top NDVI percentile for cold pixel
                 cold_lst_percentile=20,  # Coldest LST percentile for cold pixel
                 hot_ndvi_percentile=10,  # Lowest NDVI percentile for hot pixel
                 hot_lst_percentile=20,   # Hottest LST percentile for hot pixel
                 use_max_rn=False,        # Whether to use max Rn as additional criterion
                 rn_percentile=10,        # Rn percentile if using max Rn criterion
                 use_static_zom=False,    # Whether to use static zom value
                 use_observed_wind=False,  # Whether to use observed wind speed
                 use_observed_air_temp=False,  # Whether to use observed air temperature
                 vegetation_height=3,     # Vegetation height in meters
                 use_deterministic_selection=False,  # Whether to use deterministic pixel selection
                 cold_lst_deterministic_percentile=None,  # Specific LST percentile for cold pixel (if deterministic)
                 hot_lst_deterministic_percentile=None):  # Specific LST percentile for hot pixel (if deterministic)
        self.cold_ndvi_percentile = cold_ndvi_percentile
        self.cold_lst_percentile = cold_lst_percentile
        self.hot_ndvi_percentile = hot_ndvi_percentile
        self.hot_lst_percentile = hot_lst_percentile
        self.use_max_rn = use_max_rn
        self.rn_percentile = rn_percentile
        self.use_static_zom = use_static_zom
        self.use_observed_wind = use_observed_wind
        self.use_observed_air_temp = use_observed_air_temp
        self.vegetation_height = vegetation_height
        self.use_deterministic_selection = use_deterministic_selection
        self.cold_lst_deterministic_percentile = cold_lst_deterministic_percentile
        self.hot_lst_deterministic_percentile = hot_lst_deterministic_percentile
#%%
def et_collection_SR( config , start_date,end_date,lat,lon,scale, NDVI_soil=0.2, NDVI_veg=1):
    print(f"DEBUG: Starting et_collection_SR for {start_date} to {end_date}")
    print(f"DEBUG: NDVI_soil={NDVI_soil}, NDVI_veg={NDVI_veg}")
    
    geometry=ee.Geometry.Point([lon,lat])
    ls=ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate(start_date,end_date).filterMetadata('CLOUD_COVER', 'less_than',80 ).filterBounds(geometry)
    def applyScaleFactors(image):
        opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
        thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);
        return image.addBands(opticalBands, None, True).addBands(thermalBands, None, True)
    ls=ls.map(applyScaleFactors)
    ls_list=ls.aggregate_array('system:index').getInfo()
    print(f"DEBUG: Found scenes: {ls_list}")
    count = ls.size().getInfo()
    print("Number of scenes: ", count)
    n=0
    k=0
    lon_cold_pixel=[]
    lat_cold_pixel=[]
    ts_cold_scene=[]
    cold_pixel_lat,cold_pixel_lon,cold_pixel_ndvi,cold_pixel_temp,cold_pixel_sum,cold_pixel_Rn,cold_pixel_G=[],[],[],[],[],[],[]
    hot_pixel_lat,hot_pixel_lon,hot_pixel_ndvi,hot_pixel_temp,hot_pixel_sum,hot_pixel_Rn,hot_pixel_G=[],[],[],[],[],[],[]
    zenith_angle=[]
    while n < count:
        print(f"DEBUG: Processing scene {n+1}/{count}")
        image= ls.filterMetadata('system:index','equals',ls_list[n]).first()
        image.getInfo()
        image=ee.Image(image)
        NDVI_cold=5
        Ts_cold=20
        NDVI_hot=10
        Ts_hot=20
        index=image.get('system:index')
        cloud_cover=image.get('CLOUD_COVER')
        LANDSAT_ID=image.get('L1_LANDSAT_PRODUCT_ID').getInfo()
        print(f"DEBUG: Processing LANDSAT_ID: {LANDSAT_ID}")
        landsat_version=image.get('SATELLITE').getInfo()
        sun_elevation=image.get("SUN_ELEVATION")
        print(f"DEBUG: Sun elevation: {sun_elevation.getInfo()}")
        time_start=image.get('system:time_start')
        date=ee.Date(time_start)
        year=ee.Number(date.get('year'))
        month=ee.Number(date.get('month'))
        day=ee.Number(date.get('day'))
        hour=ee.Number(date.get('hour'))
        minuts = ee.Number(date.get('minutes'))
        print(f"DEBUG: Time: {str(hour.getInfo())}:{str(minuts.getInfo())}")
        crs = image.projection().crs()
        transform=ee.List(ee.Dictionary(ee.Algorithms.Describe(image.projection())).get('transform'))
        date_string=date.format('YYYY-MM-dd').getInfo()
        p_top_NDVI=ee.Number(NDVI_cold)
        p_coldest_Ts=ee.Number(Ts_cold)
        p_lowest_NDVI=ee.Number(NDVI_hot)
        p_hottest_Ts=ee.Number(Ts_hot)
        ls_trial=image.select([0,1,2,3,4,5,6,8,17], ["UB","B","GR","R","NIR","SWIR_1","SWIR_2","ST_B10","pixel_qa"])
        print(f"DEBUG: Band names: {ls_trial.bandNames().getInfo()}")
        ls_trial=masks.f_cloudMaskL8_SR(ls_trial)
        print(f"DEBUG: After cloud mask: {ls_trial.bandNames().getInfo()}")
        ls_trial=masks.f_albedoL8(ls_trial)
        print(f"DEBUG: After albedo: {ls_trial.bandNames().getInfo()}")
        geometryReducer=ls_trial.geometry().bounds().getInfo()
        geometry_download=geometryReducer['coordinates']
        print(f"DEBUG: Geometry bounds: {geometryReducer}")
        col_meteorology= meteorology.get_meteorology(ls_trial,time_start);
        T_air = col_meteorology.select('AirT_G');
        print(f"DEBUG: Meteorology bands: {T_air.bandNames().getInfo()}")
        # Always use ERA5 data
        ux = col_meteorology.select('ux_G');
        print("DEBUG: Using ERA5 wind speed")
        UR = col_meteorology.select('RH_G');
        Rn24hobs = col_meteorology.select('Rn24h_G');
        SRTM_ELEVATION ='USGS/SRTMGL1_003'
        srtm = ee.Image(SRTM_ELEVATION).clip(geometryReducer);
        z_alt = srtm.select('elevation')
        print("DEBUG: Starting spectral indices calculation")
        ls_trial=tools_ndvi_adj.fexp_spec_ind(ls_trial)
        print("DEBUG: Spectral indices done")
        print("DEBUG: Starting LST DEM correction")
        ls_trial=tools_ndvi_adj.LST_DEM_correction(ls_trial, z_alt, T_air, UR,sun_elevation,hour,minuts)
        print("DEBUG: LST DEM correction done")
        
        # Step 1: Select cold pixel for temperature only (needed for radiation calculations)
        print("DEBUG: Starting initial cold pixel selection for radiation calculations")
        if config.use_deterministic_selection and config.cold_lst_deterministic_percentile is not None:
            temp_cold_pixel=endmembers.fexp_cold_pixel_deterministic(ls_trial, geometryReducer, config.cold_ndvi_percentile, config.cold_lst_percentile, config.cold_lst_deterministic_percentile)
        else:
            temp_cold_pixel=endmembers.fexp_cold_pixel(ls_trial, geometryReducer, config.cold_ndvi_percentile, config.cold_lst_percentile)
        if temp_cold_pixel is None:
            print("Initial cold pixel selection failed - skipping scene")
            continue
        n_Ts_cold_temp = ee.Number(temp_cold_pixel.get('temp').getInfo())
        print(f"DEBUG: Initial cold pixel temperature: {n_Ts_cold_temp.getInfo()}")
        
        # Step 2: Complete radiation calculations
        print("DEBUG: Starting radiation calculations")
        ls_trial=tools_ndvi_adj.fexp_radlong_up(ls_trial)
        ls_trial=tools_ndvi_adj.fexp_radshort_down(ls_trial,z_alt,T_air,UR, sun_elevation)
        ls_trial=tools_ndvi_adj.fexp_radlong_down(ls_trial, n_Ts_cold_temp)
        ls_trial=tools_ndvi_adj.fexp_radbalance(ls_trial)
        ls_trial=tools_ndvi_adj.fexp_soil_heat(ls_trial)
        print("DEBUG: Radiation calculations done")
        
        # Step 3: Select final cold pixel with Rn and G values
        print("DEBUG: Starting final cold pixel selection with Rn and G")
        if config.use_deterministic_selection and config.cold_lst_deterministic_percentile is not None:
            d_cold_pixel=endmembers.fexp_cold_pixel_rn_g(ls_trial, geometryReducer, config.cold_ndvi_percentile, config.cold_lst_percentile, True, config.cold_lst_deterministic_percentile)
        else:
            d_cold_pixel=endmembers.fexp_cold_pixel_rn_g(ls_trial, geometryReducer, config.cold_ndvi_percentile, config.cold_lst_percentile, False)
        if d_cold_pixel is None:
            print("Final cold pixel selection failed - skipping scene")
            continue
        print(f"DEBUG: Final cold pixel selection successful: {d_cold_pixel.getInfo()}")
        print(f"DEBUG: Cold pixel temperature: {d_cold_pixel.get('temp').getInfo()}")
        print(f"DEBUG: Cold pixel Rn: {d_cold_pixel.get('Rn').getInfo()}")
        print(f"DEBUG: Cold pixel G: {d_cold_pixel.get('G').getInfo()}")
        n_Ts_cold = ee.Number(d_cold_pixel.get('temp').getInfo())
        
        # Step 4: Select hot pixel with Rn and G values
        print("DEBUG: Starting hot pixel selection with Rn and G")
        if config.use_deterministic_selection and config.hot_lst_deterministic_percentile is not None:
            d_hot_pixel=endmembers.fexp_hot_pixel_rn_g(ls_trial, geometryReducer, config.hot_ndvi_percentile, config.hot_lst_percentile, config.use_max_rn, config.rn_percentile, True, config.hot_lst_deterministic_percentile)
        else:
            d_hot_pixel=endmembers.fexp_hot_pixel_rn_g(ls_trial, geometryReducer, config.hot_ndvi_percentile, config.hot_lst_percentile, config.use_max_rn, config.rn_percentile, False)
        if d_hot_pixel is None:
            print("Hot pixel selection failed - skipping scene")
            continue
        print(f"DEBUG: Hot pixel selection successful: {d_hot_pixel.getInfo()}")
        print(f"DEBUG: Hot pixel temperature: {d_hot_pixel.get('temp').getInfo()}")
        print(f"DEBUG: Hot pixel Rn: {d_hot_pixel.get('Rn').getInfo()}")
        print(f"DEBUG: Hot pixel G: {d_hot_pixel.get('G').getInfo()}")
        print("DEBUG: About to call fexp_sensible_heat_flux")
        
        # Use NDVI-based dT sensible heat flux
        try:
            print(f"DEBUG: Using ERA5 air temperature and wind speed")
            
            ls_trial=tools_ndvi_adj.fexp_sensible_heat_flux(ls_trial, ux, UR,Rn24hobs,n_Ts_cold,
                                           d_hot_pixel, d_cold_pixel, date_string,geometryReducer, NDVI_soil, NDVI_veg, config.use_static_zom, config.vegetation_height)
            print("DEBUG: Sensible heat flux calculation successful")
        except Exception as e:
            print(f"DEBUG: Error in sensible heat flux: {str(e)}")
            raise e
            
        print("Sensible heat flux done")
        ls_trial=evapotranspiration.fexp_et(ls_trial,Rn24hobs)
        cold_pixel_lat.append(d_cold_pixel.get("y").getInfo())
        cold_pixel_lon.append(d_cold_pixel.get("x").getInfo())
        cold_pixel_temp.append(d_cold_pixel.get("temp").getInfo())
        cold_pixel_ndvi.append(d_cold_pixel.get("ndvi").getInfo())
        cold_pixel_sum.append(d_cold_pixel.get("sum").getInfo())
        cold_pixel_Rn.append(d_cold_pixel.get("Rn").getInfo())
        cold_pixel_G.append(d_cold_pixel.get("G").getInfo())
        hot_pixel_lat.append(d_hot_pixel.get("y").getInfo())
        hot_pixel_lon.append(d_hot_pixel.get("x").getInfo())
        hot_pixel_temp.append(d_hot_pixel.get("temp").getInfo())
        hot_pixel_ndvi.append(d_hot_pixel.get("ndvi").getInfo())
        hot_pixel_sum.append(d_hot_pixel.get("sum").getInfo())
        hot_pixel_Rn.append(d_hot_pixel.get("Rn").getInfo())
        hot_pixel_G.append(d_hot_pixel.get("G").getInfo())
        zenith_angle.append(90-sun_elevation.getInfo())
        NAME_FINAL=LANDSAT_ID[:5]+LANDSAT_ID[10:17]+LANDSAT_ID[17:25]
        if k ==0:
            new_ls=ee.List([])
            met=ee.List([])
            new_ls=new_ls.add(ls_trial)
            met=met.add(col_meteorology.select("Rn24h_G","AirT_G","RH_G","ux_G","SW_Down"))
        else:
            new_ls=new_ls.add(ls_trial)
            met=met.add(col_meteorology.select("Rn24h_G","AirT_G","RH_G","ux_G","SW_Down"))
        k=k+1
        print(f"DEBUG: Scene {n+1} processed successfully")
        n=n+1
        print(n)
        et_collection=ee.ImageCollection(new_ls)
        met_collection=ee.ImageCollection(met)
        region = et_collection.getRegion(geometry, int(scale)).getInfo()
        era5_met=met_collection.getRegion(geometry, int(scale)).getInfo()
    df = pd.DataFrame.from_records(region[1:len(region)])
    df_met = pd.DataFrame.from_records(era5_met[1:len(era5_met)])
    if df.shape == (0,0):
        return pd.DataFrame()
    else:
        df.columns = region[0]
        df_met.columns=era5_met[0]
        df_met=df_met.drop(["time"],axis=1)
        print(df_met)
        df=pd.concat([df,df_met],axis=1)
        df["cold_pixel_lat"]=cold_pixel_lat
        df["cold_pixel_lon"]=cold_pixel_lon
        df["cold_pixel_ndvi"]=cold_pixel_ndvi
        df["cold_pixel_sum"]=cold_pixel_sum
        df["cold_pixel_temp"]=cold_pixel_temp
        df["cold_pixel_Rn"]=cold_pixel_Rn
        df["cold_pixel_G"]=cold_pixel_G
        df["hot_pixel_sum"]=hot_pixel_sum
        df["hot_pixel_lat"]=hot_pixel_lat
        df["hot_pixel_lon"]=hot_pixel_lon
        df["hot_pixel_ndvi"]=hot_pixel_ndvi
        df["hot_pixel_Rn"]=hot_pixel_Rn
        df["hot_pixel_G"]=hot_pixel_G
        df["hot_pixel_temp"]=hot_pixel_temp
        df.time = df.time / 1000
        df['time'] = pd.to_datetime(df['time'], unit = 's')
        df.rename(columns = {'time': 'date'}, inplace = True)
        df.sort_values(by = 'date')
        return df,et_collection,met_collection
#%%
from datetime import datetime, timedelta
from collections import OrderedDict
from calendar import monthrange
def split_days(start_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = start + timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
a,b=split_days("2015-01-01")       
print(a,b)
#%%
def call_et_func(config, lon, lat, start_date, scale, name, NDVI_soil=0.15, NDVI_veg=0.85):
        start,end = split_days(start_date)
        print("Processing data from", start,end)
        output_dir = f"D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\csv_combined\\dT_edits\\ndvi_dT_tests\\US-Tw3\\{name}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        concat = pd.DataFrame()
        concat_et = None
        concat_met = None
        try:
            print(f"Processing period: {start} to {end}")
            df_sub, et, met = et_collection_SR(config, start, end, lat, lon, scale, NDVI_soil, NDVI_veg)
            df_sub.to_csv(f"{output_dir}\\{start}.csv")
            print(f"Saved data for period {start}")
            concat = df_sub
            concat_et = et
            concat_met = met
        except Exception as e:
            print(f"Error processing period {start}: {str(e)}")
            concat = pd.DataFrame()
            concat_et = None 
            concat_met = None
        return concat, concat_et, concat_met
# Example usage:
if __name__ == "__main__":
    configs = [
        EndmemberConfig(use_max_rn=True, rn_percentile=10, use_observed_wind=True),
    ]
    os.chdir("D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\csv_combined\\Af_dT_regular_volk_processed\\")
    file_list = os.listdir()
    data = []
    for i in range(len(file_list)):
        print(file_list[i])
        data.append(pd.read_csv(file_list[i]))
    for i in range(len(file_list)):
        if file_list[i] == "US-Tw3.csv":
            print(i)
    for i, config in enumerate(configs):
        config_name = f"config_{i+1}"
        print(f"Running configuration {config_name}")
        try:
            for j in range(data[103].shape[0]):
                print(data[103].iloc[j]["Name"], data[103].iloc[j]["latitude.1"], 
                      data[103].iloc[j]["longitude.1"], data[103].iloc[j]["Date"])
                et_df, et_image, met = call_et_func(
                    config=config,
                    lon=data[103]["longitude.1"].iloc[j],
                    lat=data[103]["latitude.1"].iloc[j],
                    start_date=data[103]["Date"].iloc[j],
                    scale=30,
                    name=data[103]["Name"].iloc[j]
                )
                print(et_df)
        except Exception as e:
            print(f"Error processing configuration {config_name}: {str(e)}")
# %% 