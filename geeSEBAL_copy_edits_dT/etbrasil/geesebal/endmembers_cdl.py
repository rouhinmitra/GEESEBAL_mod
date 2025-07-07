#----------------------------------------------------------------------------------------#
#---------------------------------------//GEESEBAL//-------------------------------------#
#GEESEBAL - GOOGLE EARTH ENGINE APP FOR SURFACE ENERGY BALANCE ALGORITHM FOR LAND (SEBAL)
#CREATE BY: LEONARDO LAIPELT, RAFAEL KAYSER, ANDERSON RUHOFF AND AYAN FLEISCHMANN
#PROJECT - ET BRASIL https://etbrasil.org/
#LAB - HIDROLOGIA DE GRANDE ESCALA [HGE] website: https://www.ufrgs.br/hge/author/hge/
#UNIVERSITY - UNIVERSIDADE FEDERAL DO RIO GRANDE DO SUL - UFRGS
#RIO GRANDE DO SUL, BRAZIL

#DOI
#VERSION 0.1.1
#CONTACT US: leonardo.laipelt@ufrgs.br

#----------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------#

#PYTHON PACKAGES
#Call EE
import ee

#A SIMPLIFIED VERSION OF
#CALIBRATION USING INVERSE MODELING AT EXTREME CONDITIONS (CIMEC)
#FROM ALLEN ET AL. (2013) FOR METRIC
#SEE MORE: LAIPELT ET AL. (2020)

#DEFAULT PARAMETERS
#NDVI COLD = 5%
#TS COLD = 20%
#NDVI HOT = 10%
#TS HOT = 20%

#SELECT COLD PIXEL
def fexp_cold_pixel(image, refpoly, p_top_NDVI, p_coldest_Ts, use_cdl_cold=False):
    try:
        print(f"Starting cold pixel selection. Using CDL: {use_cdl_cold}")
        if use_cdl_cold:
            print("Cold pixel selection using CDL")
            # Get CDL data for the region and time period
            image_date = ee.Date(image.get('system:time_start'))
            year = image_date.get('year')
            print("Image year:", year.getInfo())
            
            # Create date range for the year
            start_date = ee.Date.fromYMD(year, 1, 1)
            end_date = ee.Date.fromYMD(year, 12, 1)
            
            # Get CDL for the specific year using date range
            cdl = ee.ImageCollection('USDA/NASS/CDL') \
                .filterDate(start_date, end_date) \
                .first()
            print("CDL bands available:", cdl.bandNames().getInfo())
            
            # Select water pixels (class 83 or 111)
            water_mask = cdl.select('cropland').eq(83).Or(cdl.select('cropland').eq(111))
            print("Water mask created")
            
            # Apply water mask to the image
            water_masked_image = image.updateMask(water_mask)
            print("Water mask applied to image")
            
            # Get the 10th percentile LST value from water areas
            lst_percentile = water_masked_image.select('T_LST_DEM').reduceRegion(
                reducer=ee.Reducer.percentile([10]),
                geometry=refpoly,
                scale=30,
                maxPixels=9e14
            )
            lst_10th_percentile = ee.Number(lst_percentile.get('T_LST_DEM'))
            print("10th percentile LST value from water areas:", lst_10th_percentile.getInfo())
            
            # Create a mask for pixels below the 10th percentile
            cold_mask = water_masked_image.select('T_LST_DEM').lte(lst_10th_percentile)
            c_lst_cold = water_masked_image.updateMask(cold_mask)
            print("Cold mask applied (pixels below 10th percentile)")
            
            # Create binary mask for counting
            binary_mask = cold_mask.rename('binary')
            c_lst_cold = c_lst_cold.addBands(binary_mask)
            
            # Count pixels
            pixel_count = c_lst_cold.select('binary').reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=refpoly,
                scale=30,
                maxPixels=9e14
            ).get('binary')
            print("Number of cold pixels:", pixel_count.getInfo())
            
            if pixel_count.getInfo() == 0:
                print("Warning: No cold pixels found after masking")
                return None
            
            # Randomly select a cold pixel
            def function_def_pixel(f):
                return f.setGeometry(ee.Geometry.Point([f.get('longitude'), f.get('latitude')]))
            
            fc_cold_pix = c_lst_cold.stratifiedSample(1, "binary", refpoly, 30).map(function_def_pixel)
            n_Ts_cold = ee.Number(fc_cold_pix.aggregate_first('T_LST_DEM'))
            n_long_cold = ee.Number(fc_cold_pix.aggregate_first('longitude'))
            n_lat_cold = ee.Number(fc_cold_pix.aggregate_first('latitude'))
            n_ndvi_cold = ee.Number(fc_cold_pix.aggregate_first('NDVI'))
            
            # Print cold pixel values for debugging
            print("Selected cold pixel values:")
            print("LST:", n_Ts_cold.getInfo())
            print("Longitude:", n_long_cold.getInfo())
            print("Latitude:", n_lat_cold.getInfo())
            print("NDVI:", n_ndvi_cold.getInfo())
            
            #CREATE A DICTIONARY WITH THOSE RESULTS
            d_cold_pixel = ee.Dictionary({
                'temp': ee.Number(n_Ts_cold),
                'ndvi': ee.Number(n_ndvi_cold),
                'x': ee.Number(n_long_cold),
                'y': ee.Number(n_lat_cold),
                'sum': ee.Number(pixel_count)
            })
            
            return d_cold_pixel
            
        else:
            print("Using default cold pixel selection method")
            # Original cold pixel selection logic
            #IDENTIFY THE TOP % NDVI PIXELS
            d_perc_top_NDVI = image.select('NDVI_neg').reduceRegion(
                reducer=ee.Reducer.percentile([p_top_NDVI]),
                geometry=refpoly,
                scale=30,
                maxPixels=9e14
            )
            
            #GET VALUE
            n_perc_top_NDVI = ee.Number(d_perc_top_NDVI.get('NDVI_neg'))
            print("NDVI percentile:", n_perc_top_NDVI.getInfo())
            
            #UPDATE MASK WITH NDVI VALUES
            i_top_NDVI = image.updateMask(image.select('NDVI_neg').lte(n_perc_top_NDVI))
            
            #SELECT THE COLDEST TS FROM PREVIOUS NDVI GROUP
            d_perc_low_LST = i_top_NDVI.select('LST_NW').reduceRegion(
                reducer=ee.Reducer.percentile([p_coldest_Ts]),
                geometry=refpoly,
                scale=30,
                maxPixels=9e14
            )
            
            #GET VALUE
            n_perc_low_LST = ee.Number(d_perc_low_LST.get('LST_NW'))
            print("LST percentile:", n_perc_low_LST.getInfo())
            
            c_lst_cold = i_top_NDVI.updateMask(i_top_NDVI.select('LST_NW').lte(n_perc_low_LST))
            
            #FILTERS
            c_lst_cold20 = c_lst_cold.updateMask(image.select('LST_NW').gte(200))
            c_lst_cold20_int = c_lst_cold20.select('LST_NW').int().rename('int')
            c_lst_cold20 = c_lst_cold20.addBands(c_lst_cold20_int)
            
            #COUNT NUMBER OF PIXELS
            count_final_cold_pix = c_lst_cold20.select('int').reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=refpoly,
                scale=30,
                maxPixels=9e14
            )
            n_count_final_cold_pix = ee.Number(count_final_cold_pix.get('int'))
            print("Number of cold pixels:", n_count_final_cold_pix.getInfo())
            
            if n_count_final_cold_pix.getInfo() == 0:
                print("Warning: No cold pixels found after masking")
                return None
            
            #SELECT COLD PIXEL RANDOMLY (FROM PREVIOUS SELECTION)
            def function_def_pixel(f):
                return f.setGeometry(ee.Geometry.Point([f.get('longitude'), f.get('latitude')]))
            
            fc_cold_pix = c_lst_cold20.stratifiedSample(1, "int", refpoly, 30).map(function_def_pixel)
            n_Ts_cold = ee.Number(fc_cold_pix.aggregate_first('LST_NW'))
            n_long_cold = ee.Number(fc_cold_pix.aggregate_first('longitude'))
            n_lat_cold = ee.Number(fc_cold_pix.aggregate_first('latitude'))
            n_ndvi_cold = ee.Number(fc_cold_pix.aggregate_first('NDVI'))
            
            # Print cold pixel values for debugging
            print("Cold pixel values:")
            print("LST:", n_Ts_cold.getInfo())
            print("Longitude:", n_long_cold.getInfo())
            print("Latitude:", n_lat_cold.getInfo())
            print("NDVI:", n_ndvi_cold.getInfo())
            
            #CREATE A DICTIONARY WITH THOSE RESULTS
            d_cold_pixel = ee.Dictionary({
                'temp': ee.Number(n_Ts_cold),
                'ndvi': ee.Number(n_ndvi_cold),
                'x': ee.Number(n_long_cold),
                'y': ee.Number(n_lat_cold),
                'sum': ee.Number(n_count_final_cold_pix)
            })
            
            return d_cold_pixel
        
    except Exception as e:
        print(f"Error in fexp_cold_pixel: {str(e)}")
        return None

#SELECT HOT PIXEL
def fexp_hot_pixel(image, refpoly, p_lowest_NDVI, p_hottest_Ts, use_max_rn=False, rn_percentile=90, use_cdl_hot=False):
    try:
        print(f"Starting hot pixel selection. Using CDL: {use_cdl_hot}")
        if use_cdl_hot:
            print("Hot pixel selection using CDL")
            # Get CDL data for the region and time period
            image_date = ee.Date(image.get('system:time_start'))
            year = image_date.get('year')
            print("Image year:", year.getInfo())
            
            # Create date range for the year
            start_date = ee.Date.fromYMD(year, 1, 1)
            end_date = ee.Date.fromYMD(year, 12, 1)
            
            # Get CDL for the specific year using date range
            cdl = ee.ImageCollection('USDA/NASS/CDL') \
                .filterDate(start_date, end_date) \
                .first()
            print("CDL bands available:", cdl.bandNames().getInfo())
            
            # Select cultivated crops (class 2)
            cultivated_mask = cdl.select('cropland').eq(2)
            print("Cultivated mask created")
            
            # Apply cultivated mask to the image
            image = image.updateMask(cultivated_mask)
            print("Cultivated mask applied to image")

            # Get the maximum LST value from cultivated areas
            max_lst = image.select('T_LST_DEM').reduceRegion(
                reducer=ee.Reducer.max(),
                geometry=refpoly,
                scale=30,
                maxPixels=9e14
            )
            max_lst_value = ee.Number(max_lst.get('T_LST_DEM'))
            print("Maximum LST value from cultivated areas:", max_lst_value.getInfo())

            # Create a mask for pixels with maximum LST
            max_lst_mask = image.select('T_LST_DEM').eq(max_lst_value)
            c_lst_hotpix = image.updateMask(max_lst_mask)
            print("Maximum LST mask applied")

            # Create binary mask for counting
            binary_mask = max_lst_mask.rename('binary')
            c_lst_hotpix = c_lst_hotpix.addBands(binary_mask)

            # Count pixels
            pixel_count = c_lst_hotpix.select('binary').reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=refpoly,
                scale=30,
                maxPixels=9e14
            ).get('binary')
            print("Number of hot pixels:", pixel_count.getInfo())

            if pixel_count.getInfo() == 0:
                print("Warning: No hot pixels found after masking")
                return None

            #SELECT HOT PIXEL RANDOMLY
            def function_def_pixel(f):
                return f.setGeometry(ee.Geometry.Point([f.get('longitude'), f.get('latitude')]))

            fc_hot_pix = c_lst_hotpix.stratifiedSample(1, "binary", refpoly, 30, seed=3).map(function_def_pixel)
            n_Ts_hot = ee.Number(fc_hot_pix.aggregate_first('T_LST_DEM'))
            n_long_hot = ee.Number(fc_hot_pix.aggregate_first('longitude'))
            n_lat_hot = ee.Number(fc_hot_pix.aggregate_first('latitude'))
            n_ndvi_hot = ee.Number(fc_hot_pix.aggregate_first('NDVI'))
            n_Rn_hot = ee.Number(fc_hot_pix.aggregate_first('Rn'))
            n_G_hot = ee.Number(fc_hot_pix.aggregate_first('G'))

            # Print hot pixel values for debugging
            print("Hot pixel values:")
            print("LST:", n_Ts_hot.getInfo())
            print("Longitude:", n_long_hot.getInfo())
            print("Latitude:", n_lat_hot.getInfo())
            print("NDVI:", n_ndvi_hot.getInfo())
            print("Rn:", n_Rn_hot.getInfo())
            print("G:", n_G_hot.getInfo())

            #CREATE A DICTIONARY WITH THOSE RESULTS
            d_hot_pixel = ee.Dictionary({
                'temp': ee.Number(n_Ts_hot),
                'x': ee.Number(n_long_hot),
                'y': ee.Number(n_lat_hot),
                'Rn': ee.Number(n_Rn_hot),
                'G': ee.Number(n_G_hot),
                'ndvi': ee.Number(n_ndvi_hot),
                'sum': ee.Number(pixel_count)
            })

            return d_hot_pixel

        else:
            print("Using default hot pixel selection method")
            # Default GEESEBAL method for hot pixel selection
            #IDENTIFY THE DOWN % NDVI PIXELS
            d_perc_down_ndvi = image.select('pos_NDVI').reduceRegion(
                reducer=ee.Reducer.percentile([p_lowest_NDVI]),
                geometry=refpoly,
                scale=30,
                maxPixels=9e14
            )
            #GET VALUE
            n_perc_low_NDVI = ee.Number(d_perc_down_ndvi.get('pos_NDVI'))
            print("NDVI percentile:", n_perc_low_NDVI.getInfo())

            #UPDATE MASK WITH NDVI VALUES
            i_low_NDVI = image.updateMask(image.select('pos_NDVI').lte(n_perc_low_NDVI))

            #SELECT THE HOTTEST TS FROM PREVIOUS NDVI GROUP
            d_perc_top_lst = i_low_NDVI.select('LST_neg').reduceRegion(
                reducer=ee.Reducer.percentile([p_hottest_Ts]),
                geometry=refpoly,
                scale=30,
                maxPixels=9e14
            )

            #GET VALUE
            n_perc_top_lst = ee.Number(d_perc_top_lst.get('LST_neg'))
            print("LST percentile:", n_perc_top_lst.getInfo())

            c_lst_hotpix = i_low_NDVI.updateMask(i_low_NDVI.select('LST_neg').lte(n_perc_top_lst))

            # If using max Rn as additional criterion
            if use_max_rn:
                print("Applying max Rn criterion")
                # Get Rn percentile
                d_perc_rn = c_lst_hotpix.select('Rn').reduceRegion(
                    reducer=ee.Reducer.percentile([rn_percentile]),
                    geometry=refpoly,
                    scale=30,
                    maxPixels=9e14
                )
                n_perc_rn = ee.Number(d_perc_rn.get('Rn'))
                print("Rn percentile:", n_perc_rn.getInfo())
                # Update mask with Rn values
                c_lst_hotpix = c_lst_hotpix.updateMask(c_lst_hotpix.select('Rn').lte(n_perc_rn))

            c_lst_hotpix_int = c_lst_hotpix.select('LST_NW').int().rename('int')

            #COUNT NUMBER OF PIXELS
            count_final_hot_pix = c_lst_hotpix_int.select('int').reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=refpoly,
                scale=30,
                maxPixels=9e14
            )
            n_count_final_hot_pix = ee.Number(count_final_hot_pix.get('int'))
            print("Number of hot pixels:", n_count_final_hot_pix.getInfo())

            if n_count_final_hot_pix.getInfo() == 0:
                print("Warning: No hot pixels found after masking")
                return None

            #SELECT HOT PIXEL RANDOMLY (FROM PREVIOUS SELECTION)
            def function_def_pixel(f):
                return f.setGeometry(ee.Geometry.Point([f.get('longitude'), f.get('latitude')]))

            fc_hot_pix = c_lst_hotpix.stratifiedSample(1, "int", refpoly, 30, seed=3).map(function_def_pixel)
            n_Ts_hot = ee.Number(fc_hot_pix.aggregate_first('LST_NW'))
            n_long_hot = ee.Number(fc_hot_pix.aggregate_first('longitude'))
            n_lat_hot = ee.Number(fc_hot_pix.aggregate_first('latitude'))
            n_ndvi_hot = ee.Number(fc_hot_pix.aggregate_first('NDVI'))
            n_Rn_hot = ee.Number(fc_hot_pix.aggregate_first('Rn'))
            n_G_hot = ee.Number(fc_hot_pix.aggregate_first('G'))

            # Print hot pixel values for debugging
            print("Hot pixel values:")
            print("LST:", n_Ts_hot.getInfo())
            print("Longitude:", n_long_hot.getInfo())
            print("Latitude:", n_lat_hot.getInfo())
            print("NDVI:", n_ndvi_hot.getInfo())
            print("Rn:", n_Rn_hot.getInfo())
            print("G:", n_G_hot.getInfo())

            #CREATE A DICTIONARY WITH THOSE RESULTS
            d_hot_pixel = ee.Dictionary({
                'temp': ee.Number(n_Ts_hot),
                'x': ee.Number(n_long_hot),
                'y': ee.Number(n_lat_hot),
                'Rn': ee.Number(n_Rn_hot),
                'G': ee.Number(n_G_hot),
                'ndvi': ee.Number(n_ndvi_hot),
                'sum': ee.Number(n_count_final_hot_pix)
            })

            return d_hot_pixel
        
    except Exception as e:
        print(f"Error in fexp_hot_pixel: {str(e)}")
        return None
