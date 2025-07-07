#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
#%%
def merge_endmember_configs(ec_dir, config_dirs, file_name, output_dir):
    """
    Merge results from different endmember configurations with EC data
    
    Parameters:
    -----------
    ec_dir : str
        Directory containing the EC data file
    config_dirs : dict
        Dictionary mapping configuration names to their respective directories
        e.g., {'coldest_pixel_cold': 'path/to/coldest_pixel_cold', ...}
    file_name : str
        Base name of the files (e.g., 'US-Tw3')
    output_dir : str
        Directory where merged results will be saved
    """
    # Read EC data
    os.chdir(ec_dir)
    ec_file = pd.read_csv(file_name + ".csv")
    if "Date" in ec_file.columns:
        ec_file["Date"] = pd.to_datetime(ec_file["Date"])
    
    # Initialize merged data with EC data
    merged_data = ec_file.copy()
    
    # Process each configuration
    for config_name, config_dir in config_dirs.items():
        print(f"\nProcessing configuration: {config_name}")
        
        # Read all CSV files from the configuration directory
        os.chdir(config_dir)
        run_list = [f for f in os.listdir() if f.endswith(".csv")]
        
        if not run_list:
            print(f"No CSV files found in {config_dir}")
            continue
            
        # Combine all runs for this configuration
        run_data = []
        for file in run_list:
            df = pd.read_csv(file)
            run_data.append(df)
        
        if not run_data:
            print(f"No data found for configuration {config_name}")
            continue
            
        run_file = pd.concat(run_data)
        
        # Remove duplicates based on 'id' column if it exists
        if 'id' in run_file.columns:
            run_file = run_file.drop_duplicates(subset=['id']).reset_index(drop=True)
            print(f"Removed duplicates for {config_name}, remaining rows: {len(run_file)}")
        
        # Process dates if date column exists
        if "date" in run_file.columns:
            run_file["Date"] = pd.to_datetime(pd.to_datetime(run_file["date"]).dt.date)
            run_file = run_file.sort_values(by="Date")
        
        # Rename columns to avoid conflicts
        run_file = run_file.rename(columns={
            "H": "Hinst",
            "LE": "LEinst",
            "G": "Ginst"
        })
        
        # Select columns to merge
        columns_to_merge = [
            "Rn", "dT", "rah", "rah_first", "Hinst", "Ginst", "LEinst", "ET_24h",
            'cold_pixel_lat', 'cold_pixel_lon', 'cold_pixel_ndvi', 'cold_pixel_sum',
            'cold_pixel_temp', 'hot_pixel_sum', 'hot_pixel_lat', 'hot_pixel_lon',
            'hot_pixel_ndvi', 'hot_pixel_Rn', 'hot_pixel_G', 'hot_pixel_temp',"zom","fv","fstress"
        ]
        
        # Filter columns that actually exist in run_file
        available_columns = [col for col in columns_to_merge if col in run_file.columns]
        
        # Add suffix to all columns before merging
        run_file = run_file.rename(columns={col: f"{col}_{config_name}" for col in available_columns})
        
        # Determine merge key - prefer 'id' over 'Date'
        merge_key = None
        if 'id' in run_file.columns and 'id' in merged_data.columns:
            merge_key = 'id'
        elif "Date" in run_file.columns and "Date" in merged_data.columns:
            merge_key = "Date"
        else:
            print(f"Warning: No common merge key found for {config_name}")
            continue
        
        # Prepare columns for merging
        merge_columns = [merge_key] + [f"{col}_{config_name}" for col in available_columns]
        merge_columns = [col for col in merge_columns if col in run_file.columns]
        
        # Merge with existing data
        merged_data = pd.merge(
            merged_data,
            run_file[merge_columns],
            on=merge_key,
            how="left"
        )
        
        print(f"Successfully merged {config_name} configuration using {merge_key} as key")
        print(f"Merged data shape: {merged_data.shape}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the merged data
    os.chdir(output_dir)
    merged_data.to_csv(f"{file_name}_endmember_merged.csv", index=False)
    print(f"\nMerged data saved to {output_dir}/{file_name}_endmember_merged.csv")
    
    return merged_data

#%%
if __name__ == "__main__":
    # Define directories for each configuration
    base_dir = "D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\csv_combined\\dT_edits\\ndvi_dT_tests\\US-ARM\\"
    config_dirs = {
        # 'Rn_edit': os.path.join(base_dir, 'Rn_edit'),
        'fv_fstress': os.path.join(base_dir, 'Regular_fv_fstress'),


    }
    
   # Merge configurations
    merged_data = merge_endmember_configs(
        ec_dir="D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\csv_combined\\Af_dT_regular_volk_processed\\",
        config_dirs=config_dirs,
        file_name="US-ARM",
        output_dir=os.path.join(base_dir, "merged_results")
    )
    
    # Print summary of merged data
    print("\nSummary of merged data:")
    print(f"Total number of rows: {len(merged_data)}")
    print("\nColumns in merged data:")
    for col in merged_data.columns:
        print(f"- {col}")
    
    # Print sample of merged data
    print("\nSample of merged data:")
    print(merged_data.head())
# %% 
merged_data
# %%
