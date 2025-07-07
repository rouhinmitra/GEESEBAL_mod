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
def fexp_cold_pixel(image, refpoly, p_top_NDVI, p_coldest_Ts):

  #IDENTIFY THE TOP % NDVI PIXELS
  d_perc_top_NDVI=image.select('NDVI_neg').reduceRegion(
      reducer=ee.Reducer.percentile([p_top_NDVI]),
      geometry= refpoly,
      scale= 30,
      maxPixels=9e14)

  #GET VALUE
  n_perc_top_NDVI= ee.Number(d_perc_top_NDVI.get('NDVI_neg'))
#   print(n_perc_top_NDVI.getInfo())
  #UPDATE MASK WITH NDVI VALUES
  i_top_NDVI=image.updateMask(image.select('NDVI_neg').lte(n_perc_top_NDVI));

  #SELECT THE COLDEST TS FROM PREVIOUS NDVI GROUP
  d_perc_low_LST = i_top_NDVI.select('LST_NW').reduceRegion(
    reducer= ee.Reducer.percentile([p_coldest_Ts]),
    geometry=refpoly,
    scale= 30,
    maxPixels=9e14
    )
  #GET VALUE
  n_perc_low_LST = ee.Number(d_perc_low_LST.get('LST_NW'))
  i_cold_lst = i_top_NDVI.updateMask(i_top_NDVI.select('LST_NW').lte(n_perc_low_LST));
#   print(n_perc_low_LST)

  #FILTERS
  c_lst_cold20 =  i_cold_lst.updateMask(image.select('LST_NW').gte(200))
  c_lst_cold20_int=c_lst_cold20.select('LST_NW').int().rename('int')
  c_lst_cold20=c_lst_cold20.addBands(c_lst_cold20_int)

  #COUNT NUNMBER OF PIXELS
  count_final_cold_pix = c_lst_cold20.select('int').reduceRegion(
        reducer=  ee.Reducer.count(),
        geometry= refpoly,
        scale= 30,
        maxPixels=9e14)
  n_count_final_cold_pix = ee.Number(count_final_cold_pix.get('int'))

  #SELECT COLD PIXEL RANDOMLY (FROM PREVIOUS SELECTION)
  def function_def_pixel(f):
      return f.setGeometry(ee.Geometry.Point([f.get('longitude'), f.get('latitude')]));

  fc_cold_pix = c_lst_cold20.stratifiedSample(1, "int", refpoly, 30).map(function_def_pixel)
  n_Ts_cold = ee.Number(fc_cold_pix.aggregate_first('LST_NW'))
  n_long_cold = ee.Number(fc_cold_pix.aggregate_first('longitude'))
  n_lat_cold = ee.Number(fc_cold_pix.aggregate_first('latitude'))
  n_ndvi_cold = ee.Number(fc_cold_pix.aggregate_first('NDVI'))
  n_Rn_cold = ee.Number(fc_cold_pix.aggregate_first('Rn'))
  n_G_cold = ee.Number(fc_cold_pix.aggregate_first('G'))

  #CREATE A DICTIONARY WITH THOSE RESULTS
  d_cold_pixel = ee.Dictionary({
          'temp': ee.Number(n_Ts_cold),
          'ndvi': ee.Number(n_ndvi_cold),
          'x': ee.Number(n_long_cold),
          'y': ee.Number(n_lat_cold),
          'Rn': ee.Number(n_Rn_cold),
          'G': ee.Number(n_G_cold),
          'sum': ee.Number(n_count_final_cold_pix)})

  #RETURN DICTIONARY
  return d_cold_pixel

#SELECT HOT PIXEL
def fexp_hot_pixel(image, refpoly, p_lowest_NDVI, p_hottest_Ts, use_max_rn=False, rn_percentile=90):

  #IDENTIFY THE DOWN % NDVI PIXELS
  d_perc_down_ndvi=image.select('pos_NDVI').reduceRegion(
      reducer=ee.Reducer.percentile([p_lowest_NDVI]),
      geometry= refpoly,
      scale= 30,
      maxPixels=9e14
       );
  #GET VALUE
  n_perc_low_NDVI= ee.Number(d_perc_down_ndvi.get('pos_NDVI'))

  #UPDATE MASK WITH NDVI VALUES
  i_low_NDVI = image.updateMask(image.select('pos_NDVI').lte(n_perc_low_NDVI));

  #SELECT THE HOTTEST TS FROM PREVIOUS NDVI GROUP
  d_perc_top_lst = i_low_NDVI.select('LST_neg').reduceRegion(
    reducer= ee.Reducer.percentile([p_hottest_Ts]),
    geometry=refpoly,
    scale= 30,
    maxPixels=9e14
    );

  #GET VALUE
  n_perc_top_lst = ee.Number(d_perc_top_lst.get('LST_neg'))

  c_lst_hotpix = i_low_NDVI.updateMask(i_low_NDVI.select('LST_neg').lte(n_perc_top_lst))

  # If using max Rn as additional criterion
  if use_max_rn:
      # Get Rn percentile
      d_perc_rn = c_lst_hotpix.select('Rn').reduceRegion(
          reducer=ee.Reducer.percentile([rn_percentile]),
          geometry=refpoly,
          scale=30,
          maxPixels=9e14
      )
      n_perc_rn = ee.Number(d_perc_rn.get('Rn'))
      # Update mask with Rn values
      c_lst_hotpix = c_lst_hotpix.updateMask(c_lst_hotpix.select('Rn').lte(n_perc_rn))

  c_lst_hotpix_int=c_lst_hotpix.select('LST_NW').int().rename('int')

  #COUNT NUNMBER OF PIXELS
  count_final_hot_pix = c_lst_hotpix_int.select('int').reduceRegion(
        reducer=  ee.Reducer.count(),
        geometry= refpoly,
        scale= 30,
        maxPixels=9e14)
  n_count_final_hot_pix = ee.Number(count_final_hot_pix.get('int'))

  #SELECT HOT PIXEL RANDOMLY (FROM PREVIOUS SELECTION)
  def function_def_pixel(f):
     return f.setGeometry(ee.Geometry.Point([f.get('longitude'), f.get('latitude')]))

  fc_hot_pix = c_lst_hotpix.stratifiedSample(1, "int", refpoly, 30,seed=3).map(function_def_pixel)
  n_Ts_hot = ee.Number(fc_hot_pix.aggregate_first('LST_NW'));
  n_long_hot = ee.Number(fc_hot_pix.aggregate_first('longitude'))
  n_lat_hot = ee.Number(fc_hot_pix.aggregate_first('latitude'))
  n_ndvi_hot = ee.Number(fc_hot_pix.aggregate_first('NDVI'))
  n_Rn_hot = ee.Number(fc_hot_pix.aggregate_first('Rn'))
  n_G_hot = ee.Number(fc_hot_pix.aggregate_first('G'))

  #CREATE A DICTIONARY WITH THOSE RESULTS
  d_hot_pixel = ee.Dictionary({
        'temp': ee.Number(n_Ts_hot),
        'x': ee.Number(n_long_hot),
        'y': ee.Number(n_lat_hot),
        'Rn': ee.Number(n_Rn_hot),
        'G': ee.Number(n_G_hot),
        'ndvi': ee.Number(n_ndvi_hot),
        'sum': ee.Number(n_count_final_hot_pix)})

  #RETURN DICTIONARY
  return d_hot_pixel

#SELECT COLD PIXEL WITH DETERMINISTIC LST PERCENTILE
def fexp_cold_pixel_deterministic(image, refpoly, p_top_NDVI, p_coldest_Ts, cold_lst_percentile):

  #IDENTIFY THE TOP % NDVI PIXELS
  d_perc_top_NDVI=image.select('NDVI_neg').reduceRegion(
      reducer=ee.Reducer.percentile([p_top_NDVI]),
      geometry= refpoly,
      scale= 30,
      maxPixels=9e14)

  #GET VALUE
  n_perc_top_NDVI= ee.Number(d_perc_top_NDVI.get('NDVI_neg'))
#   print(n_perc_top_NDVI.getInfo())
  #UPDATE MASK WITH NDVI VALUES
  i_top_NDVI=image.updateMask(image.select('NDVI_neg').lte(n_perc_top_NDVI));

  #SELECT THE COLDEST TS FROM PREVIOUS NDVI GROUP
  d_perc_low_LST = i_top_NDVI.select('LST_NW').reduceRegion(
    reducer= ee.Reducer.percentile([p_coldest_Ts]),
    geometry=refpoly,
    scale= 30,
    maxPixels=9e14
    )
  #GET VALUE
  n_perc_low_LST = ee.Number(d_perc_low_LST.get('LST_NW'))
  i_cold_lst = i_top_NDVI.updateMask(i_top_NDVI.select('LST_NW').lte(n_perc_low_LST));
#   print(n_perc_low_Lst)

  #FILTERS
  c_lst_cold20 =  i_cold_lst.updateMask(image.select('LST_NW').gte(200))
  c_lst_cold20_int=c_lst_cold20.select('LST_NW').int().rename('int')
  c_lst_cold20=c_lst_cold20.addBands(c_lst_cold20_int)

  #COUNT NUNMBER OF PIXELS
  count_final_cold_pix = c_lst_cold20.select('int').reduceRegion(
        reducer=  ee.Reducer.count(),
        geometry= refpoly,
        scale= 30,
        maxPixels=9e14)
  n_count_final_cold_pix = ee.Number(count_final_cold_pix.get('int'))

  # Check if we have any pixels after initial filtering
  if n_count_final_cold_pix.getInfo() == 0:
      # Return None to indicate no valid pixels found
      return None
  else:
      #SELECT COLD PIXEL USING SPECIFIC LST PERCENTILE (DETERMINISTIC)
      # Get the specific LST percentile from the filtered pixels
      d_cold_lst_percentile = c_lst_cold20.select('LST_NW').reduceRegion(
          reducer=ee.Reducer.percentile([cold_lst_percentile]),
          geometry=refpoly,
          scale=30,
          maxPixels=9e14
      )
      n_cold_lst_percentile = ee.Number(d_cold_lst_percentile.get('LST_NW'))
      
      # Select pixels within a small range around the target percentile (more robust)
      # Use a tolerance of ±0.5K to ensure we get pixels
      tolerance = ee.Number(0.5)
      c_cold_exact = c_lst_cold20.updateMask(
          c_lst_cold20.select('LST_NW').gte(n_cold_lst_percentile.subtract(tolerance))
          .And(c_lst_cold20.select('LST_NW').lte(n_cold_lst_percentile.add(tolerance)))
      )
      
      # If no pixels found with tolerance, fall back to original random selection
      pixel_count = c_cold_exact.select('int').reduceRegion(
          reducer=ee.Reducer.count(),
          geometry=refpoly,
          scale=30,
          maxPixels=9e14
      )
      
      # Use original random selection if no pixels found with deterministic approach
      if pixel_count.get('int').getInfo() == 0:
          c_cold_exact = c_lst_cold20
      
      # Get the first pixel from this selection (deterministic)
      def function_def_pixel(f):
          return f.setGeometry(ee.Geometry.Point([f.get('longitude'), f.get('latitude')]));

      fc_cold_pix = c_cold_exact.stratifiedSample(1, "int", refpoly, 30, seed=1).map(function_def_pixel)

      # Get pixel values
      n_Ts_cold = ee.Number(fc_cold_pix.aggregate_first('LST_NW'))
      n_long_cold = ee.Number(fc_cold_pix.aggregate_first('longitude'))
      n_lat_cold = ee.Number(fc_cold_pix.aggregate_first('latitude'))
      n_ndvi_cold = ee.Number(fc_cold_pix.aggregate_first('NDVI'))
      n_Rn_cold = ee.Number(fc_cold_pix.aggregate_first('Rn'))
      n_G_cold = ee.Number(fc_cold_pix.aggregate_first('G'))

      #CREATE A DICTIONARY WITH THOSE RESULTS
      d_cold_pixel = ee.Dictionary({
              'temp': ee.Number(n_Ts_cold),
              'ndvi': ee.Number(n_ndvi_cold),
              'x': ee.Number(n_long_cold),
              'y': ee.Number(n_lat_cold),
              'Rn': ee.Number(n_Rn_cold),
              'G': ee.Number(n_G_cold),
              'sum': ee.Number(n_count_final_cold_pix)})

      #RETURN DICTIONARY
      return d_cold_pixel

#SELECT HOT PIXEL WITH DETERMINISTIC LST PERCENTILE
def fexp_hot_pixel_deterministic(image, refpoly, p_lowest_NDVI, p_hottest_Ts, hot_lst_percentile, use_max_rn=False, rn_percentile=90):

  #IDENTIFY THE DOWN % NDVI PIXELS
  d_perc_down_ndvi=image.select('pos_NDVI').reduceRegion(
      reducer=ee.Reducer.percentile([p_lowest_NDVI]),
      geometry= refpoly,
      scale= 30,
      maxPixels=9e14
       );
  #GET VALUE
  n_perc_low_NDVI= ee.Number(d_perc_down_ndvi.get('pos_NDVI'))

  #UPDATE MASK WITH NDVI VALUES
  i_low_NDVI = image.updateMask(image.select('pos_NDVI').lte(n_perc_low_NDVI));

  #SELECT THE HOTTEST TS FROM PREVIOUS NDVI GROUP
  d_perc_top_lst = i_low_NDVI.select('LST_neg').reduceRegion(
    reducer= ee.Reducer.percentile([p_hottest_Ts]),
    geometry=refpoly,
    scale= 30,
    maxPixels=9e14
    );

  #GET VALUE
  n_perc_top_lst = ee.Number(d_perc_top_lst.get('LST_neg'))

  c_lst_hotpix = i_low_NDVI.updateMask(i_low_NDVI.select('LST_neg').lte(n_perc_top_lst))

  # If using max Rn as additional criterion
  if use_max_rn:
      # Get Rn percentile
      d_perc_rn = c_lst_hotpix.select('Rn').reduceRegion(
          reducer=ee.Reducer.percentile([rn_percentile]),
          geometry=refpoly,
          scale=30,
          maxPixels=9e14
      )
      n_perc_rn = ee.Number(d_perc_rn.get('Rn'))
      # Update mask with Rn values
      c_lst_hotpix = c_lst_hotpix.updateMask(c_lst_hotpix.select('Rn').lte(n_perc_rn))

  c_lst_hotpix_int=c_lst_hotpix.select('LST_NW').int().rename('int')

  #COUNT NUNMBER OF PIXELS
  count_final_hot_pix = c_lst_hotpix_int.select('int').reduceRegion(
        reducer=  ee.Reducer.count(),
        geometry= refpoly,
        scale= 30,
        maxPixels=9e14)
  n_count_final_hot_pix = ee.Number(count_final_hot_pix.get('int'))

  # Check if we have any pixels after initial filtering
  if n_count_final_hot_pix.getInfo() == 0:
      # Return None to indicate no valid pixels found
      return None
  else:
      #SELECT HOT PIXEL USING SPECIFIC LST PERCENTILE (DETERMINISTIC)
      # Get the specific LST percentile from the filtered pixels
      d_hot_lst_percentile = c_lst_hotpix.select('LST_NW').reduceRegion(
          reducer=ee.Reducer.percentile([hot_lst_percentile]),
          geometry=refpoly,
          scale=30,
          maxPixels=9e14
      )
      n_hot_lst_percentile = ee.Number(d_hot_lst_percentile.get('LST_NW'))
      
      # Select pixels within a small range around the target percentile (more robust)
      # Use a tolerance of ±0.5K to ensure we get pixels
      tolerance = ee.Number(0.5)
      c_hot_exact = c_lst_hotpix.updateMask(
          c_lst_hotpix.select('LST_NW').gte(n_hot_lst_percentile.subtract(tolerance))
          .And(c_lst_hotpix.select('LST_NW').lte(n_hot_lst_percentile.add(tolerance)))
      )
      
      # If no pixels found with tolerance, fall back to original random selection
      pixel_count = c_hot_exact.select('int').reduceRegion(
          reducer=ee.Reducer.count(),
          geometry=refpoly,
          scale=30,
          maxPixels=9e14
      )
      
      # Use original random selection if no pixels found with deterministic approach
      if pixel_count.get('int').getInfo() == 0:
          c_hot_exact = c_lst_hotpix
      
      # Get the first pixel from this selection (deterministic)
      def function_def_pixel(f):
         return f.setGeometry(ee.Geometry.Point([f.get('longitude'), f.get('latitude')]))

      fc_hot_pix = c_hot_exact.stratifiedSample(1, "int", refpoly, 30, seed=2).map(function_def_pixel)

      # Get pixel values
      n_Ts_hot = ee.Number(fc_hot_pix.aggregate_first('LST_NW'));
      n_long_hot = ee.Number(fc_hot_pix.aggregate_first('longitude'))
      n_lat_hot = ee.Number(fc_hot_pix.aggregate_first('latitude'))
      n_ndvi_hot = ee.Number(fc_hot_pix.aggregate_first('NDVI'))
      n_Rn_hot = ee.Number(fc_hot_pix.aggregate_first('Rn'))
      n_G_hot = ee.Number(fc_hot_pix.aggregate_first('G'))

      #CREATE A DICTIONARY WITH THOSE RESULTS
      d_hot_pixel = ee.Dictionary({
            'temp': ee.Number(n_Ts_hot),
            'x': ee.Number(n_long_hot),
            'y': ee.Number(n_lat_hot),
            'Rn': ee.Number(n_Rn_hot),
            'G': ee.Number(n_G_hot),
            'ndvi': ee.Number(n_ndvi_hot),
            'sum': ee.Number(n_count_final_hot_pix)})

      #RETURN DICTIONARY
      return d_hot_pixel

#SELECT COLD PIXEL RN AND G VALUES (AFTER RADIATION CALCULATIONS)
def fexp_cold_pixel_rn_g(image, refpoly, p_top_NDVI, p_coldest_Ts, use_deterministic=False, cold_lst_percentile=None):
    """
    Get cold pixel Rn and G values from an image that already has Rn and G bands calculated.
    This function should be called after radiation calculations are complete.
    """
    if use_deterministic and cold_lst_percentile is not None:
        # Use deterministic selection
        d_cold_pixel = fexp_cold_pixel_deterministic(image, refpoly, p_top_NDVI, p_coldest_Ts, cold_lst_percentile)
    else:
        # Use regular selection
        d_cold_pixel = fexp_cold_pixel(image, refpoly, p_top_NDVI, p_coldest_Ts)
    
    return d_cold_pixel

#SELECT HOT PIXEL RN AND G VALUES (AFTER RADIATION CALCULATIONS)
def fexp_hot_pixel_rn_g(image, refpoly, p_lowest_NDVI, p_hottest_Ts, use_max_rn=False, rn_percentile=90, use_deterministic=False, hot_lst_percentile=None):
    """
    Get hot pixel Rn and G values from an image that already has Rn and G bands calculated.
    This function should be called after radiation calculations are complete.
    """
    if use_deterministic and hot_lst_percentile is not None:
        # Use deterministic selection
        d_hot_pixel = fexp_hot_pixel_deterministic(image, refpoly, p_lowest_NDVI, p_hottest_Ts, hot_lst_percentile, use_max_rn, rn_percentile)
    else:
        # Use regular selection
        d_hot_pixel = fexp_hot_pixel(image, refpoly, p_lowest_NDVI, p_hottest_Ts, use_max_rn, rn_percentile)
    
    return d_hot_pixel
