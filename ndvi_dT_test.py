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
from ndvi_experiments import EndmemberConfig, et_collection_SR, split_days, call_et_func
#%%
def run_ndvi_dT_tests():
    """
    Run GEESEBAL with different endmember selection configurations for Tw3 station using NDVI-based dT correction
    """
    configs = [
        EndmemberConfig(
            cold_ndvi_percentile=5,
            cold_lst_percentile=20,
            hot_ndvi_percentile=10,
            hot_lst_percentile=20,
            use_static_zom=False,
            vegetation_height=3
        )
    ]
    os.chdir("D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\csv_combined\\Af_dT_regular_volk_processed\\")
    file_list = os.listdir()
    data = []
    for i in range(len(file_list)):
        print(file_list[i])
        data.append(pd.read_csv(file_list[i]))
    tw3_index = None
    for i, df in enumerate(data):
        if "US-Tw3" in df["Name"].iloc[0]:
            tw3_index = i
            break
    if tw3_index is None:
        print("Error: Tw3 station data not found")
        return
    base_output_dir = "D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\csv_combined\\dT_edits\\ndvi_dT_tests"
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    for i, config in enumerate(configs):
        if i == 0:
            config_name = "Regular_fv"
        else:
            config_name = f"config_{i+1}"
        print(f"\nRunning configuration: {config_name}")
        print(f"Cold pixel settings: NDVI percentile={config.cold_ndvi_percentile}, LST percentile={config.cold_lst_percentile}")
        print(f"Hot pixel settings: NDVI percentile={config.hot_ndvi_percentile}, LST percentile={config.hot_lst_percentile}")
        print(f"Using static zom: {config.use_static_zom}")
        print(f"Vegetation height: {config.vegetation_height} m")
        config_output_dir = os.path.join(base_output_dir, config_name)
        if not os.path.exists(config_output_dir):
            os.makedirs(config_output_dir)
        try:
            for j in range(data[tw3_index].shape[0]):
                try:
                    print(f"\nProcessing {data[tw3_index].iloc[j]['Name']} for date {data[tw3_index].iloc[j]['Date']}")
                    print(f"Location: lat={data[tw3_index].iloc[j]['latitude.1']}, lon={data[tw3_index].iloc[j]['longitude.1']}")
                    print(f"Using ERA5 meteorology data")
                    et_df, et_image, met = call_et_func(
                        config=config,
                        lon=data[tw3_index]["longitude.1"].iloc[j],
                        lat=data[tw3_index]["latitude.1"].iloc[j],
                        start_date=data[tw3_index]["Date"].iloc[j],
                        scale=30,
                        name=f"{data[tw3_index]['Name'].iloc[j]}_{config_name}"
                    )
                    if not et_df.empty:
                        print(f"Successfully processed data for {data[tw3_index].iloc[j]['Date']}")
                        print(f"Cold pixel temperature: {et_df['cold_pixel_temp'].iloc[0]:.2f} K")
                        print(f"Hot pixel temperature: {et_df['hot_pixel_temp'].iloc[0]:.2f} K")
                        print(f"dT: {et_df['hot_pixel_temp'].iloc[0] - et_df['cold_pixel_temp'].iloc[0]:.2f} K")
                        date_str = data[tw3_index].iloc[j]['Date'].replace('-', '_')
                        output_file = os.path.join(config_output_dir, f"results_{date_str}.csv")
                        et_df.to_csv(output_file, index=False)
                        print(f"Saved results to {output_file}")
                    else:
                        print(f"No data available for {data[tw3_index].iloc[j]['Date']}")
                except Exception as e:
                    print(f"Error processing date {data[tw3_index].iloc[j]['Date']}: {str(e)}")
                    print(f"Continuing with next date...")
                    continue
        except Exception as e:
            print(f"Error processing configuration {config_name}: {str(e)}")
if __name__ == "__main__":
    run_ndvi_dT_tests()
# %% 