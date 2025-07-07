#%%
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
#%%
# os.chdir("D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\csv_combined\\dT_edits\\ndvi_dT_tests\\merged_results\\")
os.chdir("D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\csv_combined\\dT_edits\\ndvi_dT_tests\\US-ARM\\merged_results\\")
tw3=pd.read_csv("US-ARM_endmember_merged.csv")
tw3["closure_ratio"] = (tw3["H_inst_af"] + tw3["LE_inst_af"]) / (tw3["Rn_inst_af"] - tw3["G_inst_af"]) * 100
# tw3 = tw3[(tw3["closure_ratio"] >= 75) & (tw3["closure_ratio"] <= 125)].copy()
tw3["Date"]=pd.to_datetime(tw3["Date"])
tw3 = tw3[(tw3["ET_24h_fv_fstress"]>=0) & (tw3["ET_24h"]>=0)]
# tw3=tw3[(tw3["Hinst"]<600) & (tw3["Hinst"]>=0) ]
# tw3["TA_orig"]=tw3["T_LST_DEM"]-273.13-tw3["dT"]
# tw3["TA_Rn_edit"]=tw3["T_LST_DEM"]-273.13-tw3["dT_Rn_edit"]
# tw3["TA_coldest_pixel_cold"]=tw3["T_LST_DEM"]-273.13-tw3["dT_coldest_pixel_cold"]
# tw3["TA_hottest_pixel_hot"]=tw3["T_LST_DEM"]-273.13-tw3["dT_hottest_pixel_hot"]
# tw3["TA_coldest_hottest_pixels"]=tw3["T_LST_DEM"]-273.13-tw3["dT_coldest_hottest_pixels"]
# tw3["TA_cdl_hot_regular_cold"]=tw3["T_LST_DEM"]-273.13-tw3["dT_cdl_hot_regular_cold"]
# tw3["TA_regular_hot_cdl_cold"]=tw3["T_LST_DEM"]-273.13-tw3["dT_regular_hot_cdl_cold"]
# tw3["TA_cdl_hot_cdl_cold"]=tw3["T_LST_DEM"]-273.13-tw3["dT_cdl_hot_cdl_cold"]
# tw3["TA_coldest_pixel_cold_static_zom"]=tw3["T_LST_DEM"]-273.13-tw3["dT_coldest_pixel_cold_static_zom"]
# tw3["TA_hottest_pixel_hot_static_zom"]=tw3["T_LST_DEM"]-273.13-tw3["dT_hottest_pixel_hot_static_zom"]
# tw3["TA_coldest_hottest_pixels_static_zom"]=tw3["T_LST_DEM"]-273.13-tw3["dT_coldest_hottest_pixels_static_zom"]
# tw3["TA_original_pixel_selection_static_zom"]=tw3["T_LST_DEM"]-273.13-tw3["dT_original_pixel_selection_static_zom"]
# tw3["dT_measured"]=tw3["T_LST_DEM"]-273.13-tw3["TA"]
tw3.columns
# Statistics for Hinst values outside the range 0-800 for each configuration
print("Statistics for Hinst values outside the range 0-800:")
print("=" * 60)
tw3.columns.tolist()
tw3["Hinst_fv_fstress"].describe()
tw3[["ET_24h_fv_fstress","ET_24h"]].describe()
tw3.columns
#%%



#%%
"""OPenET read"""
os.chdir("D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML_H\\OpenET\\H_trainset")
# os.chdir("D:\\Backup\\Rouhin_Lenovo\\US_project\\GEE_SEBAL_Project\\ML\\OpenET_New")
openet=pd.read_csv("US-ARM.csv")
openet["Date"]=pd.to_datetime(openet["Date"])
# openet["Date"].iloc[0]
os.chdir("D:\\Backup\\Rouhin_Lenovo\\US_project\\Untitled_Folder\\Data\\Volk_insitu\\flux_ET_dataset\\daily_data_files")
volk=pd.read_csv("US-ARM_daily_data.csv")
volk["Date"]=pd.to_datetime(volk["date"])
tw3= pd.merge(tw3, openet, on="Date", how="left", suffixes=('', '_openet'))
tw3= pd.merge(tw3, volk[["Date","ET_corr"]], on="Date", how="left", suffixes=('', '_volk'))
# tw3[["ETa","ET_corr"]]
# tw3
#%%


# Define the configurations to check
hinst_columns = [
    'Hinst',  # regular/original
    'Hinst_Rn_edit',
    'Hinst_fv_fstress',
]

# Check each configuration
for col in hinst_columns:
    if col in tw3.columns:
        # Get non-null values
        hinst_values = tw3[col].dropna()
        
        # Count values outside range 0-800
        outside_range = hinst_values[(hinst_values < 0) | (hinst_values > 800)]
        below_zero = hinst_values[hinst_values < 0]
        above_800 = hinst_values[hinst_values > 800]
        
        # Calculate percentages
        total_count = len(hinst_values)
        outside_count = len(outside_range)
        below_zero_count = len(below_zero)
        above_800_count = len(above_800)
        
        outside_pct = (outside_count / total_count * 100) if total_count > 0 else 0
        below_zero_pct = (below_zero_count / total_count * 100) if total_count > 0 else 0
        above_800_pct = (above_800_count / total_count * 100) if total_count > 0 else 0
        
        print(f"\n{col.replace('_', ' ').title()}:")
        print(f"  Total values: {total_count}")
        print(f"  Outside range (0-800): {outside_count} ({outside_pct:.1f}%)")
        print(f"    Below 0: {below_zero_count} ({below_zero_pct:.1f}%)")
        print(f"    Above 800: {above_800_count} ({above_800_pct:.1f}%)")
        
        if len(outside_range) > 0:
            print(f"  Min outside value: {outside_range.min():.1f}")
            print(f"  Max outside value: {outside_range.max():.1f}")
    else:
        print(f"\n{col}: Column not found in dataset")

print("\n" + "=" * 60)
# Filter out points where Hinst values are outside the range 0-800 for all configurations
print("Filtering out points where Hinst values are outside range 0-800...")
print(f"Original dataset size: {len(tw3)}")

# Create a mask for valid Hinst values across all configurations
valid_mask = pd.Series(True, index=tw3.index)

for col in hinst_columns:
    if col in tw3.columns:
        # Add condition that values should be between 0 and 800
        col_mask = (tw3[col] >= 0) & (tw3[col] <= 800)
        valid_mask = valid_mask & (tw3[col].isna() | col_mask)  # Allow NaN values or valid range

# Apply the filter
tw3 = tw3[valid_mask].copy()

print(f"Filtered dataset size: {len(tw3)}")
print(f"Removed {len(tw3) - len(tw3[valid_mask])} points with Hinst values outside 0-800 range")

""""
Compare how hot pixel and cold pixel selection in the 1% affects ET results """
tw3["LE_closed"]
#%%

#%%
# Plot Rn vs different configurations
rn_configs = [col for col in tw3.columns if col.startswith('Rn_') and col != 'Rn_inst_af']

# Add regular Rn column if it exists
if 'Rn' in tw3.columns:
    rn_configs.append('Rn')

if 'Rn_fv_fstress' in tw3.columns:
    rn_configs.append('Rn_fv_fstress')

# Check if we have Rn_inst_af column
if 'Rn_inst_af' not in tw3.columns:
    print("Warning: Rn_inst_af column not found in dataset")
else:
    if len(rn_configs) > 0:
        # Calculate grid dimensions based on number of plots
        n_plots = len(rn_configs)
        if n_plots <= 6:
            nrows, ncols = 2, 3
        elif n_plots <= 8:
            nrows, ncols = 2, 4
        elif n_plots <= 9:
            nrows, ncols = 3, 3
        else:
            nrows, ncols = 3, 4
        
        plt.figure(figsize=(5*ncols, 5*nrows))
        
        for i, config_col in enumerate(rn_configs):
            plt.subplot(nrows, ncols, i+1)
            
            # Get common valid data points
            common_idx = tw3['Rn_inst_af'].notna() & tw3[config_col].notna()
            x_common = tw3.loc[common_idx, 'Rn_inst_af']
            y_common = tw3.loc[common_idx, config_col]
            
            # Create scatter plot
            plt.scatter(x_common, y_common, alpha=0.6, s=20)
            
            # Add 1:1 line
            plt.plot([0, 800], [0, 800], 'r--', alpha=0.8, label='1:1 line')
            
            # Calculate correlation coefficient and R²
            if len(x_common) > 1:
                from sklearn.metrics import r2_score
                correlation = np.corrcoef(x_common, y_common)[0, 1]
                r2 = r2_score(x_common, y_common)
                plt.text(0.05, 0.95, f'r = {correlation:.3f}', transform=plt.gca().transAxes, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                plt.text(0.05, 0.87, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Set labels and title
            plt.xlabel('Rn_inst_af (W/m²)')
            plt.ylabel(f'{config_col} (W/m²)')
            if config_col == 'Rn':
                config_name = 'Regular'
            else:
                config_name = config_col.replace('Rn_', '').replace('_', ' ').title()
            plt.title(f'Rn_inst_af vs {config_name}')
            plt.grid(True, alpha=0.3)
            
            # Set axis limits
            # plt.xlim(0, 800)
            # plt.ylim(0, 800)

        plt.tight_layout()
        plt.show()

        # Print correlation statistics
        print("\nCorrelation Analysis: Rn_inst_af vs Different Configurations")
        print("=" * 70)
        for config_col in rn_configs:
            common_idx = tw3['Rn_inst_af'].notna() & tw3[config_col].notna()
            if common_idx.sum() > 1:
                from sklearn.metrics import r2_score
                x_common = tw3.loc[common_idx, 'Rn_inst_af']
                y_common = tw3.loc[common_idx, config_col]
                
                correlation = np.corrcoef(x_common, y_common)[0, 1]
                r2 = r2_score(x_common, y_common)
                rmse = np.sqrt(np.mean((x_common - y_common)**2))
                bias = np.mean(y_common - x_common)
                
                if config_col == 'Rn':
                    config_name = 'Regular'
                else:
                    config_name = config_col.replace('Rn_', '').replace('_', ' ').title()
                print(f"\n{config_name}:")
                print(f"  Correlation (r): {correlation:.4f}")
                print(f"  R-squared (R²): {r2:.4f}")
                print(f"  RMSE: {rmse:.2f} W/m²")
                print(f"  Bias: {bias:.2f} W/m²")
                print(f"  Sample size: {common_idx.sum()} observations")
    else:
        print("No Rn configuration columns found in dataset")


# %%

# Create scatter plots comparing H_inst_af with different configurations
plt.figure(figsize=(20, 12))

# Get the configuration names from the columns in the desired order
h_inst_configs = []

# Add regular Hinst column first if it exists
if 'Hinst' in tw3.columns:
    h_inst_configs.append('Hinst')

# Add Rn_edit configuration
if 'Hinst_Rn_edit' in tw3.columns:
    h_inst_configs.append('Hinst_Rn_edit')

# Add cdl_hot_regular_cold configuration
if 'Hinst_fv_fstress' in tw3.columns:
    h_inst_configs.append('Hinst_fv_fstress')





# Add any remaining Hinst configurations
remaining_configs = [col for col in tw3.columns if col.startswith('Hinst_') and col not in h_inst_configs and col != 'H_inst_af']
h_inst_configs.extend(remaining_configs)

# Limit to 8 configurations for 2x4 format
h_inst_configs = h_inst_configs[:14]

for i, config_col in enumerate(h_inst_configs):
    plt.subplot(3, 5, i + 1)
    
    # Get data for this configuration
    x_data = tw3['H_inst_af'].dropna()
    y_data = tw3[config_col].dropna()
    
    # Find common indices (where both values exist)
    common_idx = tw3['H_inst_af'].notna() & tw3[config_col].notna()
    x_common = tw3.loc[common_idx, 'H_inst_af']
    y_common = tw3.loc[common_idx, config_col]
    
    # Create scatter plot
    plt.scatter(x_common, y_common, alpha=0.6, s=30)
    
    # Add 1:1 line
    plt.plot([0, 800], [0, 800], 'r--', alpha=0.8, label='1:1 line')
    
    # Calculate correlation coefficient, bias and RMSE
    if len(x_common) > 1:
        correlation = np.corrcoef(x_common, y_common)[0, 1]
        rmse = np.sqrt(np.mean((x_common - y_common)**2))
        bias = np.mean(y_common - x_common)
        plt.text(0.05, 0.95, f'r = {correlation:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=14)
        plt.text(0.05, 0.87, f'RMSE = {rmse:.1f} W/m²', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=14)
        plt.text(0.05, 0.79, f'Bias = {bias:.1f} W/m²', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=14)
    
    # Set labels and title
    plt.xlabel('Measured Hinst (W/m²)', fontsize=16)
    plt.ylabel('Hinst Estimated (W/m²)', fontsize=16)
    if config_col == 'Hinst':
        config_name = 'Regular'
    else:
        config_name = config_col.replace('Hinst_', '').replace('_', ' ').title()
    plt.title(f'{config_name}', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', labelsize=14)
    
    # Set axis limits
    plt.xlim(0, 600)
    plt.ylim(0, 600)

plt.tight_layout()
plt.show()

# Print correlation statistics
print("\nCorrelation Analysis: H_inst_af vs Different Configurations")
print("=" * 70)
for config_col in h_inst_configs:
    common_idx = tw3['H_inst_af'].notna() & tw3[config_col].notna()
    if common_idx.sum() > 1:
        from sklearn.metrics import r2_score
        x_common = tw3.loc[common_idx, 'H_inst_af']
        y_common = tw3.loc[common_idx, config_col]
        
        correlation = np.corrcoef(x_common, y_common)[0, 1]
        r2 = r2_score(x_common, y_common)
        rmse = np.sqrt(np.mean((x_common - y_common)**2))
        bias = np.mean(y_common - x_common)
        
        if config_col == 'Hinst':
            config_name = 'Regular'
        else:
            config_name = config_col.replace('Hinst_', '').replace('_', ' ').title()
        print(f"\n{config_name}:")
        print(f"  Correlation (r): {correlation:.4f}")
        print(f"  R-squared (R²): {r2:.4f}")
        print(f"  RMSE: {rmse:.2f} W/m²")
        print(f"  Bias: {bias:.2f} W/m²")
        print(f"  Sample size: {common_idx.sum()} observations")
#%%
# Create scatter plots comparing dT_measured with different configurations
plt.figure(figsize=(20, 12))

# Get the configuration names from the columns in the desired order
dt_configs = []

# Add regular dT column first if it exists
if 'dT' in tw3.columns:
    dt_configs.append('dT')

# Add Rn_edit configuration
if 'dT_Rn_edit' in tw3.columns:
    dt_configs.append('dT_Rn_edit')

# Add cdl_hot_regular_cold configuration
if 'dT_fv_fstress' in tw3.columns:
    dt_configs.append('dT_fv_fstress')

# Add any remaining dT configurations
remaining_configs = [col for col in tw3.columns if col.startswith('dT_') and col not in dt_configs and col != 'dT_measured']
dt_configs.extend(remaining_configs)

# Limit to 12 configurations for 3x4 format
dt_configs = dt_configs[:12]

for i, config_col in enumerate(dt_configs):
    plt.subplot(3, 4, i + 1)
    
    # Get data for this configuration
    x_data = tw3['dT_measured'].dropna()
    y_data = tw3[config_col].dropna()
    
    # Find common indices (where both values exist)
    common_idx = tw3['dT_measured'].notna() & tw3[config_col].notna()
    x_common = tw3.loc[common_idx, 'dT_measured']
    y_common = tw3.loc[common_idx, config_col]
    
    # Create scatter plot
    plt.scatter(x_common, y_common, alpha=0.6, s=30)
    
    # Add 1:1 line
    max_val = max(x_common.max(), y_common.max()) if len(x_common) > 0 else 10
    min_val = min(x_common.min(), y_common.min()) if len(x_common) > 0 else 0
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='1:1 line')
    
    # Calculate correlation coefficient, bias and RMSE
    if len(x_common) > 1:
        correlation = np.corrcoef(x_common, y_common)[0, 1]
        rmse = np.sqrt(np.mean((x_common - y_common)**2))
        bias = np.mean(y_common - x_common)
        plt.text(0.05, 0.95, f'r = {correlation:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=14)
        plt.text(0.05, 0.87, f'RMSE = {rmse:.2f} °C', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=14)
        plt.text(0.05, 0.79, f'Bias = {bias:.2f} °C', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=14)
    
    # Set labels and title
    plt.xlabel('Measured dT (°C)', fontsize=16)
    plt.ylabel('dT Estimated (°C)', fontsize=16)
    if config_col == 'dT':
        config_name = 'Regular'
    else:
        config_name = config_col.replace('dT_', '').replace('_', ' ').title()
    plt.title(f'{config_name}', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', labelsize=14)
        # Set axis limits
    plt.xlim(0, 20)
    plt.ylim(0, 20)
plt.tight_layout()
plt.show()

# Print correlation statistics
print("\nCorrelation Analysis: dT_measured vs Different Configurations")
print("=" * 70)
for config_col in dt_configs:
    common_idx = tw3['dT_measured'].notna() & tw3[config_col].notna()
    if common_idx.sum() > 1:
        from sklearn.metrics import r2_score
        x_common = tw3.loc[common_idx, 'dT_measured']
        y_common = tw3.loc[common_idx, config_col]
        
        correlation = np.corrcoef(x_common, y_common)[0, 1]
        r2 = r2_score(x_common, y_common)
        rmse = np.sqrt(np.mean((x_common - y_common)**2))
        bias = np.mean(y_common - x_common)
        
        if config_col == 'dT':
            config_name = 'Regular'
        else:
            config_name = config_col.replace('dT_', '').replace('_', ' ').title()
        print(f"\n{config_name}:")
        print(f"  Correlation (r): {correlation:.4f}")
        print(f"  R-squared (R²): {r2:.4f}")
        print(f"  RMSE: {rmse:.3f} °C")
        print(f"  Bias: {bias:.3f} °C")
        print(f"  Sample size: {common_idx.sum()} observations")

#%%
# Create scatter plots comparing LE_inst_af with different configurations
plt.figure(figsize=(20, 12))

# Get the configuration names from the columns in the desired order
le_inst_configs = []

# Add regular LEinst column first if it exists
if 'LEinst' in tw3.columns:
    le_inst_configs.append('LEinst')

# Add Rn_edit configuration
if 'LEinst_Rn_edit' in tw3.columns:
    le_inst_configs.append('LEinst_Rn_edit')

# Add coldest pixel cold configuration
if 'LEinst_fv_fstress' in tw3.columns:
    le_inst_configs.append('LEinst_fv_fstress')






# Add any remaining LEinst configurations
remaining_configs = [col for col in tw3.columns if col.startswith('LEinst_') and col not in le_inst_configs and col != 'LE_inst_af']
le_inst_configs.extend(remaining_configs)

# Limit to 6 configurations for 2x3 format
le_inst_configs = le_inst_configs[:14]

for i, config_col in enumerate(le_inst_configs):
    plt.subplot(3, 5, i + 1)
    
    # Get data for this configuration
    x_data = tw3['LE_inst_af'].dropna()
    y_data = tw3[config_col].dropna()
    
    # Find common indices (where both values exist)
    common_idx = tw3['LE_inst_af'].notna() & tw3[config_col].notna()
    x_common = tw3.loc[common_idx, 'LE_inst_af']
    y_common = tw3.loc[common_idx, config_col]
    
    # Create scatter plot
    plt.scatter(x_common, y_common, alpha=0.6, s=30)
    
    # Add 1:1 line
    plt.plot([0, 800], [0, 800], 'r--', alpha=0.8, label='1:1 line')
    
    # Calculate correlation coefficient, bias and RMSE
    if len(x_common) > 1:
        correlation = np.corrcoef(x_common, y_common)[0, 1]
        rmse = np.sqrt(np.mean((x_common - y_common)**2))
        bias = np.mean(y_common - x_common)
        plt.text(0.05, 0.95, f'r = {correlation:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=14)
        plt.text(0.05, 0.87, f'RMSE = {rmse:.1f} W/m²', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=14)
        plt.text(0.05, 0.79, f'Bias = {bias:.1f} W/m²', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=14)
    
    # Set labels and title
    plt.xlabel('Measured LEinst (W/m²)', fontsize=16)
    plt.ylabel('LEinst Estimated (W/m²)', fontsize=16)
    if config_col == 'LEinst':
        config_name = 'Regular'
    else:
        config_name = config_col.replace('LEinst_', '').replace('_', ' ').title()
    plt.title(f'{config_name}', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', labelsize=14)
    
    # Set axis limits
    plt.xlim(0, 600)
    plt.ylim(0, 600)

plt.tight_layout()
plt.show()

# Print correlation statistics
print("\nCorrelation Analysis: LE_inst_af vs Different Configurations")
print("=" * 70)
for config_col in le_inst_configs:
    common_idx = tw3['LE_inst_af'].notna() & tw3[config_col].notna()
    if common_idx.sum() > 1:
        from sklearn.metrics import r2_score
        x_common = tw3.loc[common_idx, 'LE_inst_af']
        y_common = tw3.loc[common_idx, config_col]
        
        correlation = np.corrcoef(x_common, y_common)[0, 1]
        r2 = r2_score(x_common, y_common)
        rmse = np.sqrt(np.mean((x_common - y_common)**2))
        bias = np.mean(y_common - x_common)
        
        if config_col == 'LEinst':
            config_name = 'Regular'
        else:
            config_name = config_col.replace('LEinst_', '').replace('_', ' ').title()
        print(f"\n{config_name}:")
        print(f"  Correlation (r): {correlation:.4f}")
        print(f"  R-squared (R²): {r2:.4f}")
        print(f"  RMSE: {rmse:.2f} W/m²")
        print(f"  Bias: {bias:.2f} W/m²")
        print(f"  Sample size: {common_idx.sum()} observations")
#%%
# Compare LE_closed with ET_24h and ETa
if 'LE_closed' in tw3.columns:
    print("\nComparing LE_closed with ET_24h configurations and ETa")
    print("=" * 50)
    
    # Convert LE_closed from W/m² to mm/day by dividing by 28.36
    le_closed_mm = tw3['LE_closed'] / 28.36
    
    # Find all ET_24h configuration columns
    et_24h_configs = [col for col in tw3.columns if col.startswith('ET_24h')]
    
    # Add ETa if it exists
    all_configs = et_24h_configs.copy()
    if 'ETa' in tw3.columns:
        all_configs.append('ETa')
    
    if all_configs:
        # Limit to 12 configurations for 3x4 format
        all_configs = all_configs[:12]
        
        plt.figure(figsize=(15, 10))
        
        for i, config_col in enumerate(all_configs, 1):
            plt.subplot(3, 4, i)
            
            # Filter out NaN values for both variables
            common_idx = le_closed_mm.notna() & tw3[config_col].notna()
            
            if common_idx.sum() > 0:
                x_common = le_closed_mm.loc[common_idx]
                y_common = tw3.loc[common_idx, config_col]
                
                # Create scatter plot
                plt.scatter(x_common, y_common, alpha=0.6, s=20)
                
                # Add 1:1 line
                max_val = max(x_common.max(), y_common.max()) if len(x_common) > 0 else 10
                plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.8, label='1:1 line')
                
                # Calculate correlation coefficient and R²
                if len(x_common) > 1:
                    from sklearn.metrics import r2_score
                    correlation = np.corrcoef(x_common, y_common)[0, 1]
                    r2 = r2_score(x_common, y_common)
                    plt.text(0.05, 0.95, f'r = {correlation:.3f}', transform=plt.gca().transAxes, 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    plt.text(0.05, 0.87, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Set labels and title
                plt.xlabel('LE_closed (mm/day)')
                plt.ylabel(f'{config_col} (mm/day)')
                if config_col == 'ET_24h':
                    config_name = 'Regular'
                elif config_col == 'ETa':
                    config_name = 'ETa'
                else:
                    config_name = config_col.replace('ET_24h_', '').replace('_', ' ').title()
                plt.title(f'LE_closed vs {config_name}')
                plt.grid(True, alpha=0.3)
                
                # Set axis limits based on data range
                if len(x_common) > 0 and len(y_common) > 0:
                    max_lim = max(x_common.max(), y_common.max()) * 1.1
                    plt.xlim(0, max_lim)
                    plt.ylim(0, max_lim)

        plt.tight_layout()
        plt.show()

        # Print correlation statistics
        print("\nCorrelation Analysis: LE_closed vs Different ET Configurations")
        print("=" * 70)
        for config_col in all_configs:
            common_idx = le_closed_mm.notna() & tw3[config_col].notna()
            if common_idx.sum() > 1:
                from sklearn.metrics import r2_score
                x_common = le_closed_mm.loc[common_idx]
                y_common = tw3.loc[common_idx, config_col]
                
                correlation = np.corrcoef(x_common, y_common)[0, 1]
                r2 = r2_score(x_common, y_common)
                rmse = np.sqrt(np.mean((x_common - y_common)**2))
                bias = np.mean(y_common - x_common)
                
                if config_col == 'ET_24h':
                    config_name = 'Regular'
                elif config_col == 'ETa':
                    config_name = 'ETa'
                else:
                    config_name = config_col.replace('ET_24h_', '').replace('_', ' ').title()
                print(f"\n{config_name}:")
                print(f"  Correlation (r): {correlation:.4f}")
                print(f"  R-squared (R²): {r2:.4f}")
                print(f"  RMSE: {rmse:.2f} mm/day")
                print(f"  Bias: {bias:.2f} mm/day")
                print(f"  Sample size: {common_idx.sum()} observations")
    else:
        print("No ET_24h configuration columns or ETa found in dataset")
else:
    print("Missing column in dataset: LE_closed")
#%%
# Create scatter plots comparing ET_corr with different ET configurations
if 'ET_corr' in tw3.columns:
    # Convert ET_corr from W/m² to mm/day (divide by 28.4)
    et_corr_mm = tw3['ET_corr']
    
    # Get all ET configuration columns
    et_configs = []
    
    # Add regular ET_24h column first if it exists
    if 'ET_24h' in tw3.columns:
        et_configs.append('ET_24h')
    
    # Add ETa if it exists
    if 'ETa' in tw3.columns:
        et_configs.append('ETa')
    
    # Add any remaining ET_24h configurations
    remaining_configs = [col for col in tw3.columns if col.startswith('ET_24h_') and col not in et_configs]
    et_configs.extend(remaining_configs)
    
    if et_configs:
        # Determine subplot layout based on number of configurations
        n_configs = len(et_configs)
        if n_configs <= 6:
            rows, cols = 2, 3
        elif n_configs <= 12:
            rows, cols = 3, 4
        else:
            rows, cols = 4, 4
            et_configs = et_configs[:16]  # Limit to 16 configurations
        
        plt.figure(figsize=(15, 12))
        
        for i, config_col in enumerate(et_configs):
            plt.subplot(rows, cols, i + 1)
            
            # Find common indices (where both values exist)
            common_idx = et_corr_mm.notna() & tw3[config_col].notna()
            x_common = et_corr_mm.loc[common_idx]
            y_common = tw3.loc[common_idx, config_col]
            
            # Create scatter plot
            plt.scatter(x_common, y_common, alpha=0.6, s=30)
            
            # Add 1:1 line
            if len(x_common) > 0 and len(y_common) > 0:
                max_val = max(x_common.max(), y_common.max())
                min_val = min(x_common.min(), y_common.min())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='1:1 line')
            
            # Calculate correlation coefficient, bias and RMSE
            if len(x_common) > 1:
                correlation = np.corrcoef(x_common, y_common)[0, 1]
                rmse = np.sqrt(np.mean((x_common - y_common)**2))
                bias = np.mean(y_common - x_common)
                plt.text(0.05, 0.95, f'r = {correlation:.3f}', transform=plt.gca().transAxes, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12)
                plt.text(0.05, 0.87, f'RMSE = {rmse:.2f} mm/day', transform=plt.gca().transAxes, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12)
                plt.text(0.05, 0.79, f'Bias = {bias:.2f} mm/day', transform=plt.gca().transAxes, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12)
            
            # Set labels and title
            plt.xlabel('ET_corr (mm/day)', fontsize=14)
            plt.ylabel('ET Estimated (mm/day)', fontsize=14)
            if config_col == 'ET_24h':
                config_name = 'Regular'
            elif config_col == 'ETa':
                config_name = 'ETa'
            else:
                config_name = config_col.replace('ET_24h_', '').replace('_', ' ').title()
            plt.title(f'ET_corr vs {config_name}', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.tick_params(axis='both', labelsize=12)
            
            # Set axis limits based on data range
            if len(x_common) > 0 and len(y_common) > 0:
                max_lim = max(x_common.max(), y_common.max()) * 1.1
                plt.xlim(0, max_lim)
                plt.ylim(0, max_lim)

        plt.tight_layout()
        plt.show()

        # Print correlation statistics
        print("\nCorrelation Analysis: ET_corr vs Different ET Configurations")
        print("=" * 70)
        for config_col in et_configs:
            common_idx = et_corr_mm.notna() & tw3[config_col].notna()
            if common_idx.sum() > 1:
                from sklearn.metrics import r2_score
                x_common = et_corr_mm.loc[common_idx]
                y_common = tw3.loc[common_idx, config_col]
                
                correlation = np.corrcoef(x_common, y_common)[0, 1]
                r2 = r2_score(x_common, y_common)
                rmse = np.sqrt(np.mean((x_common - y_common)**2))
                bias = np.mean(y_common - x_common)
                
                if config_col == 'ET_24h':
                    config_name = 'Regular'
                elif config_col == 'ETa':
                    config_name = 'ETa'
                else:
                    config_name = config_col.replace('ET_24h_', '').replace('_', ' ').title()
                print(f"\n{config_name}:")
                print(f"  Correlation (r): {correlation:.4f}")
                print(f"  R-squared (R²): {r2:.4f}")
                print(f"  RMSE: {rmse:.2f} mm/day")
                print(f"  Bias: {bias:.2f} mm/day")
                print(f"  Sample size: {common_idx.sum()} observations")
    else:
        print("No ET_24h configuration columns or ETa found in dataset")
else:
    print("Missing column in dataset: ET_corr")

# %%
# Plot timeseries of different net radiation values
# Extract net radiation columns for each configuration
# Sort dataframe by date first
tw3_sorted = tw3.sort_values('Date').copy()

# Plot timeseries of different net radiation values
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Check which Rn columns are available
rn_columns_to_plot = []
rn_labels = []

if 'Rn_inst_af' in tw3_sorted.columns:
    rn_columns_to_plot.append('Rn_inst_af')
    rn_labels.append('Measured')

if 'Rn' in tw3_sorted.columns:
    rn_columns_to_plot.append('Rn')
    rn_labels.append('Rn (Original)')

if 'Rn_Rn_edit' in tw3_sorted.columns:
    rn_columns_to_plot.append('Rn_Rn_edit')
    rn_labels.append('Rn Adj')

# Plot the timeseries
colors = ['k', 'tab:blue', 'tab:orange']
markers = ['o', 's', '^']
for i, (col, label) in enumerate(zip(rn_columns_to_plot, rn_labels)):
    ax.plot(tw3_sorted['Date'], tw3_sorted[col], label=label, color=colors[i % len(colors)], 
            alpha=0.7, marker=markers[i % len(markers)], markersize=4, linewidth=2.5)

ax.set_xlabel('Date', fontsize=20)
ax.set_ylabel('Net Radiation (W/m²)', fontsize=20)
ax.set_title('Net Radiation Values', fontsize=20)
ax.legend(fontsize=18)
ax.grid(True)
ax.tick_params(axis='both', labelsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45,fontsize=20)
plt.yticks(fontsize=20)

plt.tight_layout()
plt.show()

#%%


#%%
# Plot timeseries of different sensible heat flux values
fig, ax = plt.subplots(1, 1, figsize=(20, 6))

# Check which H columns are available
h_columns_to_plot = []
h_labels = []

if 'H_inst_af' in tw3_sorted.columns:
    h_columns_to_plot.append('H_inst_af')
    h_labels.append('Measured')

if 'Hinst' in tw3_sorted.columns:
    h_columns_to_plot.append('Hinst')
    h_labels.append('Hinst (Original)')


# if 'Hinst_Rn_edit' in tw3_sorted.columns:
#     h_columns_to_plot.append('Hinst_Rn_edit')
#     h_labels.append('Rn Adj')

# if 'Hinst_coldest_pixel_cold' in tw3_sorted.columns:
#     h_columns_to_plot.append('Hinst_coldest_pixel_cold')
#     h_labels.append('Cold Pixel Cold')

if 'Hinst_fv_fstress' in tw3_sorted.columns:
    h_columns_to_plot.append('Hinst_fv_fstress')
    h_labels.append('Fv Fstress')

# if 'Hinst_coldest_hottest_pixels' in tw3_sorted.columns:
#     h_columns_to_plot.append('Hinst_coldest_hottest_pixels')
#     h_labels.append('Coldest Hottest Pixels')

# Plot the timeseries
colors = ['k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
markers = ['o', 's', '^', 'D', 'v', 'p']
for i, (col, label) in enumerate(zip(h_columns_to_plot, h_labels)):
    ax.plot(tw3_sorted['Date'], tw3_sorted[col], label=label, color=colors[i % len(colors)], 
            alpha=0.7, marker=markers[i % len(markers)], markersize=4, linewidth=2.5)

ax.set_xlabel('Date', fontsize=20)
ax.set_ylabel('Sensible Heat Flux (W/m²)', fontsize=20)
ax.set_title('Sensible Heat Flux Values', fontsize=20)
ax.legend(fontsize=18, bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True)
ax.tick_params(axis='both', labelsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45,fontsize=20)
plt.yticks(fontsize=20)

plt.tight_layout()
plt.show()

#%%
# Plot timeseries of different latent heat flux values
fig, ax = plt.subplots(1, 1, figsize=(20, 6))

# Check which LE columns are available
le_columns_to_plot = []
le_labels = []

if 'LE_inst_af' in tw3_sorted.columns:
    le_columns_to_plot.append('LE_inst_af')
    le_labels.append('Measured')

if 'LEinst' in tw3_sorted.columns:
    le_columns_to_plot.append('LEinst')
    le_labels.append('LEinst (Original)')

# if 'LEinst_Rn_edit' in tw3_sorted.columns:
#     le_columns_to_plot.append('LEinst_Rn_edit')
#     le_labels.append('Rn Adj')

# if 'LEinst_coldest_pixel_cold' in tw3_sorted.columns:
#     le_columns_to_plot.append('LEinst_coldest_pixel_cold')
#     le_labels.append('Cold Pixel Cold')

if 'LEinst_fv_fstress' in tw3_sorted.columns:
    le_columns_to_plot.append('LEinst_fv_fstress')
    le_labels.append('Fv Fstress')

# Plot the timeseries
colors = ['k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
markers = ['o', 's', '^', 'D', 'v', 'p', 'X']
for i, (col, label) in enumerate(zip(le_columns_to_plot, le_labels)):
    ax.plot(tw3_sorted['Date'], tw3_sorted[col], label=label, color=colors[i % len(colors)], 
            alpha=0.7, marker=markers[i % len(markers)], markersize=4, linewidth=2.5)

ax.set_xlabel('Date', fontsize=20)
ax.set_ylabel('Latent Heat Flux (W/m²)', fontsize=20)
ax.set_title('Latent Heat Flux Values', fontsize=20)
ax.legend(fontsize=18, bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True)
ax.tick_params(axis='both', labelsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45,fontsize=20)
plt.yticks(fontsize=20)

plt.tight_layout()
plt.show()

# %%
# Create scatter plot of different TA methods vs measured TA
fig, ax = plt.subplots(figsize=(12, 8))

# Check which TA columns are available
ta_columns_to_plot = []
ta_labels = []

if 'TA' in tw3_sorted.columns:
    measured_ta = tw3_sorted['TA']
    
    if 'TA_orig' in tw3_sorted.columns:
        ta_columns_to_plot.append('TA_orig')
        ta_labels.append('TA (Original)')
    
    if 'TA_Rn_edit' in tw3_sorted.columns:
        ta_columns_to_plot.append('TA_Rn_edit')
        ta_labels.append('TA (Rn Adj)')

    if 'TA_coldest_pixel_cold' in tw3_sorted.columns:
        ta_columns_to_plot.append('TA_coldest_pixel_cold')
        ta_labels.append('TA (Cold Pixel Cold)')

    if 'TA_hottest_pixel_hot' in tw3_sorted.columns:
        ta_columns_to_plot.append('TA_hottest_pixel_hot')
        ta_labels.append('TA (Hot Pixel Hot)')

    if 'TA_cdl_hot_regular_cold' in tw3_sorted.columns:
        ta_columns_to_plot.append('TA_cdl_hot_regular_cold')
        ta_labels.append('TA (CDL Hot Regular Cold)')

    if 'TA_regular_hot_cdl_cold' in tw3_sorted.columns:
        ta_columns_to_plot.append('TA_regular_hot_cdl_cold')
        ta_labels.append('TA (Regular Hot CDL Cold)')

    if 'TA_cdl_hot_cdl_cold' in tw3_sorted.columns:
        ta_columns_to_plot.append('TA_cdl_hot_cdl_cold')
        ta_labels.append('TA (CDL Hot CDL Cold)')   

    if 'TA_coldest_hottest_pixels' in tw3_sorted.columns:

        ta_columns_to_plot.append('TA_coldest_hottest_pixels')
        ta_labels.append('TA (Coldest Hottest Pixels)')


    # Plot scatter plots
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    markers = ['o', 's', '^', 'D', 'v', 'X']
    
    for i, (col, label) in enumerate(zip(ta_columns_to_plot, ta_labels)):
        ax.scatter(measured_ta, tw3_sorted[col], label=label, 
                  color=colors[i % len(colors)], alpha=0.7, 
                  s=60, marker=markers[i % len(markers)])
    
    # Add 1:1 line
    min_val = min(measured_ta.min(), min([tw3_sorted[col].min() for col in ta_columns_to_plot]))
    max_val = max(measured_ta.max(), max([tw3_sorted[col].max() for col in ta_columns_to_plot]))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='1:1 Line')
    
    ax.set_xlabel('Measured TA (°C)', fontsize=20)
    ax.set_ylabel('Modeled TA (°C)', fontsize=20)
    ax.set_title('Air Temperature: Measured vs Modeled', fontsize=20)
    ax.legend(fontsize=18, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=16)
    
    plt.tight_layout()
    plt.show()
#%%
# Plot timeseries of different air temperature values
fig, ax = plt.subplots(1, 1, figsize=(20, 6))

# Check which TA columns are available
ta_columns_to_plot = []
ta_labels = []

if 'TA' in tw3_sorted.columns:
    ta_columns_to_plot.append('TA')
    ta_labels.append('Measured')

if 'TA_orig' in tw3_sorted.columns:
    ta_columns_to_plot.append('TA_orig')
    ta_labels.append('TA (Original)')

if 'TA_Rn_edit' in tw3_sorted.columns:
    ta_columns_to_plot.append('TA_Rn_edit')
    ta_labels.append('TA (Rn Adj)')

if 'TA_hottest_pixel_hot' in tw3_sorted.columns:
    ta_columns_to_plot.append('TA_hottest_pixel_hot')
    ta_labels.append('TA (Hot Pixel Hot)')

if 'TA_cdl_hot_regular_cold' in tw3_sorted.columns:
    ta_columns_to_plot.append('TA_cdl_hot_regular_cold')
    ta_labels.append('TA (CDL Hot Regular Cold)')

if 'TA_regular_hot_cdl_cold' in tw3_sorted.columns:
    ta_columns_to_plot.append('TA_regular_hot_cdl_cold')
    ta_labels.append('TA (Regular Hot CDL Cold)')

if 'TA_cdl_hot_cdl_cold' in tw3_sorted.columns:
    ta_columns_to_plot.append('TA_cdl_hot_cdl_cold')
    ta_labels.append('TA (CDL Hot CDL Cold)')


# Plot the timeseries
colors = ['k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
markers = ['o', 's', '^', 'D', 'v', 'p', 'X']
for i, (col, label) in enumerate(zip(ta_columns_to_plot, ta_labels)):
    ax.plot(tw3_sorted['Date'], tw3_sorted[col], label=label, color=colors[i % len(colors)], 
            alpha=0.7, marker=markers[i % len(markers)], markersize=4, linewidth=2.5)

ax.set_xlabel('Date', fontsize=20)
ax.set_ylabel('Air Temperature (°C)', fontsize=20)
ax.set_title('Air Temperature Values', fontsize=20)
ax.legend(fontsize=18, bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True)
ax.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.show()

# %%
# Create scatter plot of different rah methods vs measured rah
fig, ax = plt.subplots(figsize=(12, 8))

# Check which rah columns are available
rah_columns_to_plot = []
rah_labels = []

if 'rah' in tw3_sorted.columns:
    measured_rah = tw3_sorted['rah']
    
    if 'rah_orig' in tw3_sorted.columns:
        rah_columns_to_plot.append('rah_orig')
        rah_labels.append('rah (Original)')
    
    if 'rah_Rn_edit' in tw3_sorted.columns:
        rah_columns_to_plot.append('rah_Rn_edit')
        rah_labels.append('rah (Rn Adj)')

    if 'rah_hottest_pixel_hot' in tw3_sorted.columns:
        rah_columns_to_plot.append('rah_hottest_pixel_hot')
        rah_labels.append('rah (Hot Pixel Hot)')
    
    if 'rah_coldest_pixel_cold' in tw3_sorted.columns:
        rah_columns_to_plot.append('rah_coldest_pixel_cold')
        rah_labels.append('rah (Cold Pixel Cold)')
    
    if 'rah_coldest_hottest_pixels' in tw3_sorted.columns:
        rah_columns_to_plot.append('rah_coldest_hottest_pixels')
        rah_labels.append('rah (Coldest Hottest Pixels)')

    # Plot scatter plots
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (col, label) in enumerate(zip(rah_columns_to_plot, rah_labels)):
        ax.scatter(measured_rah, tw3_sorted[col], label=label, 
                  color=colors[i % len(colors)], alpha=0.7, 
                  s=60, marker=markers[i % len(markers)])
    
    # Add 1:1 line
    min_val = min(measured_rah.min(), min([tw3_sorted[col].min() for col in rah_columns_to_plot]))
    max_val = max(measured_rah.max(), max([tw3_sorted[col].max() for col in rah_columns_to_plot]))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='1:1 Line')
    
    ax.set_xlabel('Measured rah (s/m)', fontsize=20)
    ax.set_ylabel('Modeled rah (s/m)', fontsize=20)
    ax.set_title('Aerodynamic Resistance: Measured vs Modeled', fontsize=20)
    ax.legend(fontsize=18, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=16)
    
    plt.tight_layout()
    plt.show()

#%%
# Create scatter plot of different dT methods vs measured dT
fig, ax = plt.subplots(figsize=(12, 8))

# Check which dT columns are available
dT_columns_to_plot = []
dT_labels = []

if 'dT_measured' in tw3_sorted.columns:
    measured_dT = tw3_sorted['dT_measured']
    
    if 'dT' in tw3_sorted.columns:
        dT_columns_to_plot.append('dT')
        dT_labels.append('dT (Original)')
    
    if 'dT_Rn_edit' in tw3_sorted.columns:
        dT_columns_to_plot.append('dT_Rn_edit')
        dT_labels.append('dT (Rn Adj)')

    if 'dT_hottest_pixel_hot' in tw3_sorted.columns:
        dT_columns_to_plot.append('dT_hottest_pixel_hot')
        dT_labels.append('dT (Hot Pixel Hot)')

    # if 'dT_coldest_pixel_cold' in tw3_sorted.columns:
    #     dT_columns_to_plot.append('dT_coldest_pixel_cold')
    #     dT_labels.append('dT (Cold Pixel Cold)')

    # if 'dT_coldest_hottest_pixels' in tw3_sorted.columns:
    #     dT_columns_to_plot.append('dT_coldest_hottest_pixels')
    #     dT_labels.append('dT (Coldest Hottest Pixels)')

    if 'dT_cdl_hot_regular_cold' in tw3_sorted.columns:
        dT_columns_to_plot.append('dT_cdl_hot_regular_cold')
        dT_labels.append('dT (CDL Hot Regular Cold)')

    if 'dT_regular_hot_cdl_cold' in tw3_sorted.columns: 
        dT_columns_to_plot.append('dT_regular_hot_cdl_cold')
        dT_labels.append('dT (Regular Hot CDL Cold)')

    if 'dT_cdl_hot_cdl_cold' in tw3_sorted.columns:
        dT_columns_to_plot.append('dT_cdl_hot_cdl_cold')
        dT_labels.append('dT (CDL Hot CDL Cold)')


    # Plot scatter plots
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    markers = ['o', 's', '^', 'D', 'v', 'X']
    
    for i, (col, label) in enumerate(zip(dT_columns_to_plot, dT_labels)):
        ax.scatter(measured_dT, tw3_sorted[col], label=label, 
                  color=colors[i % len(colors)], alpha=0.7, 
                  s=60, marker=markers[i % len(markers)])
    
    # Add 1:1 line
    min_val = min(measured_dT.min(), min([tw3_sorted[col].min() for col in dT_columns_to_plot]))
    max_val = max(measured_dT.max(), max([tw3_sorted[col].max() for col in dT_columns_to_plot]))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='1:1 Line')
    
    ax.set_xlabel('Measured dT (K)', fontsize=20)
    ax.set_ylabel('Modeled dT (K)', fontsize=20)
    ax.set_title('Temperature Difference: Measured vs Modeled', fontsize=20)
    ax.legend(fontsize=18, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=16)
    
    plt.tight_layout()
    plt.show()

    # Print correlation statistics for dT
    print("\nCorrelation Analysis: Measured dT vs Different Configurations")
    print("=" * 70)
    for i, (config_col, config_name) in enumerate(zip(dT_columns_to_plot, dT_labels)):
        common_idx = measured_dT.notna() & tw3_sorted[config_col].notna()
        if common_idx.sum() > 1:
            x_common = measured_dT.loc[common_idx]
            y_common = tw3_sorted.loc[common_idx, config_col]
            
            correlation = np.corrcoef(x_common, y_common)[0, 1]
            rmse = np.sqrt(np.mean((x_common - y_common)**2))
            bias = np.mean(y_common - x_common)
            
            print(f"\n{config_name}:")
            print(f"  Correlation (r): {correlation:.4f}")
            print(f"  RMSE: {rmse:.4f} K")
            print(f"  Bias: {bias:.4f} K")
            print(f"  Sample size: {common_idx.sum()} observations")
else:
    print("Missing column in dataset: dT")

#%%
# Plot timeseries of different air temperature values
fig, ax = plt.subplots(1, 1, figsize=(20, 6))

# Check which TA columns are available
ta_columns_to_plot = []
ta_labels = []

if 'dT_measured' in tw3_sorted.columns:
    ta_columns_to_plot.append('dT_measured')
    ta_labels.append('Measured')

if 'dT' in tw3_sorted.columns:
    ta_columns_to_plot.append('dT')
    ta_labels.append('dT (Original)')

if 'dT_Rn_edit' in tw3_sorted.columns:
    ta_columns_to_plot.append('dT_Rn_edit')
    ta_labels.append('dT (Rn Adj)')

if 'dT_hottest_pixel_hot' in tw3_sorted.columns:
    ta_columns_to_plot.append('dT_hottest_pixel_hot')
    ta_labels.append('dT (Hot Pixel Hot)')

# Plot the timeseries
colors = ['k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
markers = ['o', 's', '^', 'D', 'v', 'p']
for i, (col, label) in enumerate(zip(ta_columns_to_plot, ta_labels)):
    ax.plot(tw3_sorted['Date'], tw3_sorted[col], label=label, color=colors[i % len(colors)], 
            alpha=0.7, marker=markers[i % len(markers)], markersize=4, linewidth=2.5)

ax.set_xlabel('Date', fontsize=20)
ax.set_ylabel('dT (°C)', fontsize=20)
ax.set_title('dT Values', fontsize=20)
ax.legend(fontsize=18, bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True)
ax.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.show()
#%%
# Create scatter plots comparing measured rah with different configurations
plt.figure(figsize=(15, 12))

# Check which rah columns are available (excluding measured)
rah_configs = []

if 'rah' in tw3.columns:
    rah_configs.append('rah')

if 'rah_Rn_edit' in tw3.columns:
    rah_configs.append('rah_Rn_edit')

if 'rah_coldest_pixel_cold' in tw3.columns:
    rah_configs.append('rah_coldest_pixel_cold')

if 'rah_hottest_pixel_hot' in tw3.columns:
    rah_configs.append('rah_hottest_pixel_hot')

if 'rah_coldest_hottest_pixels' in tw3.columns:
    rah_configs.append('rah_coldest_hottest_pixels')

# Add any remaining rah configurations
remaining_configs = [col for col in tw3.columns if col.startswith('rah_') and col not in rah_configs and col != 'rah_measured']
rah_configs.extend(remaining_configs)

# Limit to 6 configurations for 2x3 format
rah_configs = rah_configs[:6]

for i, config_col in enumerate(rah_configs):
    plt.subplot(2, 3, i + 1)
    
    # Find common indices (where both values exist)
    if 'rah' in tw3.columns:
        common_idx = tw3['rah'].notna() & tw3[config_col].notna()
        x_common = tw3.loc[common_idx, 'rah']
        y_common = tw3.loc[common_idx, config_col]
        
        # Create scatter plot
        plt.scatter(x_common, y_common, alpha=0.6, s=30)
        
        # Add 1:1 line
        max_val = max(x_common.max(), y_common.max()) if len(x_common) > 0 else 100
        min_val = min(x_common.min(), y_common.min()) if len(x_common) > 0 else 0
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        # Calculate and display statistics
        if len(x_common) > 1:
            correlation = np.corrcoef(x_common, y_common)[0, 1]
            rmse = np.sqrt(np.mean((x_common - y_common)**2))
            bias = np.mean(y_common - x_common)
            
            plt.text(0.05, 0.95, f'r = {correlation:.3f}', transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=14)
            plt.text(0.05, 0.87, f'RMSE = {rmse:.1f} s/m', transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=14)
            plt.text(0.05, 0.79, f'Bias = {bias:.1f} s/m', transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=14)
    
    # Set labels and title
    plt.xlabel('rah SEBAL original(s/m)', fontsize=16)
    plt.ylabel(f'{config_col} (s/m)', fontsize=14)
    if config_col == 'rah':
        config_name = 'Regular'
    else:
        config_name = config_col.replace('rah_', '').replace('_', ' ').title()
    plt.title(f'rah_measured vs {config_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()
plt.show()
#%%
plt.plot(tw3["dT"]/tw3["rah"],tw3["zom"],"o")
tw3.columns.tolist()
#%%
# plt.plot(tw3["WS"],tw3["ux_G"],"o")

# plt.plot(np.arange(0,7,1),np.arange(0,7,1),"k--")
# plt.plot((tw3["dT"]-tw3["T_LST_DEM"]+273.13+tw3["TA"]),tw3["Hinst"]-tw3["H_inst_af"],'o')
plt.plot((tw3["rah_first"]),tw3["rah"],'o')
#%%
tw3[tw3["id"]=="LC08_044033_20130619"][["Date","rah_first","rah","dT","zom","Hinst","H_inst_af","Hinst_Rn_edit","Hinst_coldest_pixel_cold","Hinst_hottest_pixel_hot","Hinst_coldest_hottest_pixels","Hinst_cdl_hot_regular_cold","Hinst_regular_hot_cdl_cold","Hinst_cdl_hot_cdl_cold","NDVI_model"]].iloc[0]
tw3[tw3["id"]=="LC08_044033_20130619"][["Date","rah_first_Rn_edit","rah_Rn_edit","dT_Rn_edit","zom_Rn_edit","Hinst_Rn_edit","H_inst_af","NDVI_model","TA","T_LST_DEM"]].iloc[0]

tw3["id"].unique()
tw3[tw3["id"]=="LC08_044033_20180601"][["Date","rah_first","rah","dT","zom","Hinst","H_inst_af","Hinst_Rn_edit","Hinst_coldest_pixel_cold","Hinst_hottest_pixel_hot","Hinst_coldest_hottest_pixels","Hinst_cdl_hot_regular_cold","Hinst_regular_hot_cdl_cold","Hinst_cdl_hot_cdl_cold","NDVI_model"]].iloc[0]
# %%
tw3[(tw3["Hinst"]-tw3["H_inst_af"])<0].id.unique()
tw3[tw3["id"]=="LC08_044033_20150727"][["Date","rah_first","rah","dT","zom","Hinst","H_inst_af","Hinst_Rn_edit","Hinst_coldest_pixel_cold","Hinst_hottest_pixel_hot","Hinst_coldest_hottest_pixels","Hinst_cdl_hot_regular_cold","Hinst_regular_hot_cdl_cold","Hinst_cdl_hot_cdl_cold","NDVI_model"]].iloc[0]
tw3[tw3["id"]=="LC08_044033_20150727"][["Date","rah_first_Rn_edit","rah_Rn_edit","dT_Rn_edit","zom_Rn_edit","Hinst_Rn_edit","H_inst_af","NDVI_model","TA","T_LST_DEM"]].iloc[0]
#%%
tw3.loc[(tw3["Hinst"] - tw3["H_inst_af"]) > 0, ["Date","id","NDVI_model", "Hinst", "Hinst_Rn_edit", "H_inst_af"]].drop_duplicates()
# Plot NDVI_model histogram after removing duplicate NDVI values for the specified condition
tw3.loc[(tw3["Hinst"] - tw3["H_inst_af"]) > 0, "NDVI_model"].drop_duplicates().hist()
# %%
plt.plot(tw3["WS"],tw3["ux_G"],"o")

plt.plot(np.arange(0,8,1),np.arange(0,8,1),"k--")
plt.xlabel('Measured Wind Speed (m/s)', fontsize=25)
plt.ylabel('ERA5 Wind Speed (m/s)', fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlim(0,7)
plt.ylim(0,7)

# Calculate bias, RMSE and correlation
common_idx = tw3['WS'].notna() & tw3['ux_G'].notna()
x_common = tw3.loc[common_idx, 'WS']
y_common = tw3.loc[common_idx, 'ux_G']

if len(x_common) > 1:
    correlation = np.corrcoef(x_common, y_common)[0, 1]
    rmse = np.sqrt(np.mean((x_common - y_common)**2))
    bias = np.mean(y_common - x_common)
    
    plt.text(0.05, 0.95, f'r = {correlation:.3f}', transform=plt.gca().transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=20)
    plt.text(0.05, 0.80, f'RMSE = {rmse:.2f} m/s', transform=plt.gca().transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=20)
    plt.text(0.05, 0.65, f'Bias = {bias:.2f} m/s', transform=plt.gca().transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=20)
#%%
plt.plot(tw3["TA"],tw3["AirT_G"],"o",c="r")
plt.plot(tw3["TA"],tw3["T_LST_DEM"]-273.13-tw3["dT"],"o",c="b")
plt.plot(np.arange(10,40,1),np.arange(10,40,1),"k--")
#%%
# plt.plot(tw3["T_LST_DEM"]-273.13-tw3["TA"],tw3["AirT_G"],"o",c="r")
plt.plot((tw3["T_LST_DEM"]-273.13-tw3["TA"]),tw3["dT"],"o",c="tab:blue")
plt.plot(np.arange(0,20,1),np.arange(0,20,1),"k--")
plt.xlabel('Measured dT (C)', fontsize=25)
plt.ylabel('Estimated dT (C)', fontsize=25)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(0,20)
plt.ylim(0,20)

# Calculate bias, RMSE and correlation
common_idx = (tw3["T_LST_DEM"]-273.13-tw3["TA"]).notna() & tw3["dT"].notna()
x_common = tw3.loc[common_idx, "T_LST_DEM"]-273.13-tw3.loc[common_idx, "TA"]
y_common = tw3.loc[common_idx, "dT"]

if len(x_common) > 1:
    correlation = np.corrcoef(x_common, y_common)[0, 1]
    rmse = np.sqrt(np.mean((x_common - y_common)**2))
    bias = np.mean(y_common - x_common)
    
    plt.text(0.05, 0.95, f'r = {correlation:.3f}', transform=plt.gca().transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=20)
    plt.text(0.05, 0.80, f'RMSE = {rmse:.2f} °C', transform=plt.gca().transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=20)
    plt.text(0.05, 0.65, f'Bias = {bias:.2f} °C', transform=plt.gca().transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=20)
#%% Calculate H using insitu dT and rah using in situ wind speed
tw3["rho_insitu"]= (-0.0046 * (tw3["TA"]+273.13)) + 2.5538
tw3["H_insitu"]= tw3["rho_insitu"] * 1004 *(tw3["T_LST_DEM"] - 273.13 - tw3["TA"])/tw3["rah_original_pixel_selection_observed_wind"]
tw3["rah_original_pixel_selection_observed_wind"].describe()
plt.plot(tw3["rah_first"],tw3["rah_first_original_pixel_selection_observed_wind"],"o")
plt.plot(np.arange(0,100,2),np.arange(0,100,2),"k--")
plt.xlabel('rah SEBAL original(s/m)', fontsize=25)
plt.ylabel('rah SEBAL observed wind(s/m)', fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlim(0,40)
plt.ylim(0,40)
#%%
tw3["H_insitu"].describe()
#%%
plt.figure(figsize=(8, 6))
plt.plot(tw3["H_inst_af"], tw3["H_insitu"], "o")
plt.plot(np.arange(0, 800, 50), np.arange(0, 800, 50), "k--")
plt.xlabel('Measured H (W/m²)', fontsize=25)
plt.ylabel('H calculated (insitu data) (W/m²)', fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
# plt.xlim(0, 1000)
# plt.ylim(0, 1000)

# Calculate bias, RMSE and correlation
common_idx = tw3["H_inst_af"].notna() & tw3["H_insitu"].notna()
x_common = tw3.loc[common_idx, "H_inst_af"]
y_common = tw3.loc[common_idx, "H_insitu"]

if len(x_common) > 1:
    correlation = np.corrcoef(x_common, y_common)[0, 1]
    rmse = np.sqrt(np.mean((x_common - y_common)**2))
    bias = np.mean(y_common - x_common)
    
    plt.text(0.95, 0.95, f'r = {correlation:.3f}', transform=plt.gca().transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=20)
    plt.text(0.95, 0.80, f'RMSE = {rmse:.1f} W/m²', transform=plt.gca().transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=20)
    plt.text(0.95, 0.65, f'Bias = {bias:.1f} W/m²', transform=plt.gca().transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=20)


# %%
tw3[["longitude.1","latitude.1"]]
