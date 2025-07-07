#!/usr/bin/env python
# coding: utf-8

# In[1]:




def et_collection_SR(start_date,end_date,path,row):
	
    sys.path.append('C:\\Rouhin_Lenovo\\US_project\\Alfalfa\\ET_Code\\geeSEBAL-master\\etbrasil\\')
    print(1)
    import geesebal
    from geesebal import (tools,landsatcollection,masks,meteorology,endmembers, 
    evapotranspiration,collection,timeseries,image)
    import ee
    ls=ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterDate(start_date,end_date).filterMetadata('WRS_PATH', 'equals', path).filterMetadata('WRS_ROW', 'equals',row).filterMetadata('CLOUD_COVER', 'less_than', 10);
#     ls = ls.filter(ee.Filter.neq('system:index','LC08_038037_20210903'));
    ls_list=ls.aggregate_array('system:index').getInfo()
    print(ls_list)
    count = ls.size().getInfo()
    print("Number of scenes: ", count)
    n=0
    k=0
    #====== ITERATIVE PROCESS ======#
    #FOR EACH IMAGE ON THE LIST
    #ESTIMATE ET DAILY IMAGE
    while n < count:
        #GET IMAGE
        image= ls.filterMetadata('system:index','equals',ls_list[n]).first()
        image.getInfo()
        image=ee.Image(image)
            # et=image.Image(image)
        NDVI_cold=5
        Ts_cold=20
        NDVI_hot=10
        Ts_hot=20
        index=image.get('system:index')
        cloud_cover=image.get('CLOUD_COVER')
        LANDSAT_ID=image.get('LANDSAT_ID').getInfo()
        print(LANDSAT_ID)
        landsat_version=image.get('SATELLITE').getInfo()
        azimuth_angle=image.get('SOLAR_ZENITH_ANGLE')
        time_start=image.get('system:time_start')
        date=ee.Date(time_start)
        year=ee.Number(date.get('year'))
        month=ee.Number(date.get('month'))
        day=ee.Number(date.get('day'))
        hour=ee.Number(date.get('hour'))
        minuts = ee.Number(date.get('minutes'))
        crs = image.projection().crs()
        transform=ee.List(ee.Dictionary(ee.Algorithms.Describe(image.projection())).get('transform'))
        date_string=date.format('YYYY-MM-dd').getInfo()
        #ENDMEMBERS
        p_top_NDVI=ee.Number(NDVI_cold)
        p_coldest_Ts=ee.Number(Ts_cold)
        p_lowest_NDVI=ee.Number(NDVI_hot)
        p_hottest_Ts=ee.Number(Ts_hot)
        ls_trial=image.select([0,1,2,3,4,5,6,7,10], ["UB","B","GR","R","NIR","SWIR_1","SWIR_2","BRT","pixel_qa"])
        ls.first_toa=ee.Image('LANDSAT/LC08/C01/T1/'+index.getInfo())

        col_rad = ee.Algorithms.Landsat.calibratedRadiance(ls.first_toa)
        col_rad = ls_trial.addBands(col_rad.select([9],["T_RAD"]))
        #CLOUD REMOVAL
        # ls_trial=ee.ImageCollection(col_rad).map(masks.f_cloudMaskL8_SR)
        ls_trial=masks.f_cloudMaskL8_SR(ls_trial)
        #ALBEDO TASUMI ET AL. (2008) METHOD WITH KE ET AL. (2016) COEFFICIENTS
        # ls_trial=ls_trial.map(masks.f_albedoL8)
        ls_trial=masks.f_albedoL8(ls_trial)
        #------ Meteorology
          #GEOMETRY
        geometryReducer=ls_trial.geometry().bounds().getInfo()
        geometry_download=geometryReducer['coordinates']
        # camada_clip=ls_trial.select('BRT').first()
        camada_clip=ls_trial.select('BRT')
        sun_elevation=ee.Number(90).subtract(ee.Number(azimuth_angle))
        #METEOROLOGY PARAMETERS
        col_meteorology= meteorology.get_meteorology(ls_trial,time_start);

        #AIR TEMPERATURE [C]
        T_air = col_meteorology.select('AirT_G');

        #WIND SPEED [M S-1]
        ux= col_meteorology.select('ux_G');
        #RELATIVE HUMIDITY [%]
        UR = col_meteorology.select('RH_G');

        #NET RADIATION 24H [W M-2]
        Rn24hobs = col_meteorology.select('Rn24h_G');
        #------
        #------ Elevation
        #SRTM DATA ELEVATION
        SRTM_ELEVATION ='USGS/SRTMGL1_003'
        srtm = ee.Image(SRTM_ELEVATION).clip(geometryReducer);
        z_alt = srtm.select('elevation')

    #     GET IMAGE
    #     ls_trial=ls_trial.first()

    #     SPECTRAL IMAGES (NDVI, EVI, SAVI, LAI, T_LST, e_0, e_NB, long, lat)
        ls_trial=tools.fexp_spec_ind(ls_trial)
        ls_trial=tools.LST_DEM_correction(ls_trial, z_alt, T_air, UR,sun_elevation,hour,minuts)
        #COLD PIXEL
        d_cold_pixel=endmembers.fexp_cold_pixel(ls_trial, geometryReducer, p_top_NDVI, p_coldest_Ts)
        #COLD PIXEL NUMBER
        n_Ts_cold = ee.Number(d_cold_pixel.get('temp').getInfo())
        #INSTANTANEOUS OUTGOING LONG-WAVE RADIATION [W M-2]
        ls_trial=tools.fexp_radlong_up(ls_trial)

        #INSTANTANEOUS INCOMING SHORT-WAVE RADIATION [W M-2]
        ls_trial=tools.fexp_radshort_down(ls_trial,z_alt,T_air,UR, sun_elevation)

        #INSTANTANEOUS INCOMING LONGWAVE RADIATION [W M-2]
        ls_trial=tools.fexp_radlong_down(ls_trial, n_Ts_cold)
        #INSTANTANEOUS NET RADIATON BALANCE [W M-2]
    #     print(ls_trial.select('Rs_down').getInfo())
    #     print(ls_trial.select('Rl_down').getInfo())
    #     print(ls_trial.select('Rl_up').getInfo())

        ls_trial=tools.fexp_radbalance(ls_trial)
    #     print(ls_trial.select('Rn').getInfo())

        #SOIL HEAT FLUX (G) [W M-2]
        ls_trial=tools.fexp_soil_heat(ls_trial)
        #HOT PIXEL
        d_hot_pixel=endmembers.fexp_hot_pixel(ls_trial, geometryReducer,p_lowest_NDVI, p_hottest_Ts)
        #SENSIBLE HEAT FLUX (H) [W M-2]
        ls_trial=tools.fexp_sensible_heat_flux(ls_trial, ux, UR,Rn24hobs,n_Ts_cold,
                                       d_hot_pixel, date_string,geometryReducer)

        #DAILY EVAPOTRANSPIRATION (ET_24H) [MM DAY-1]
        ls_trial=evapotranspiration.fexp_et(ls_trial,Rn24hobs)

        NAME_FINAL=LANDSAT_ID[:5]+LANDSAT_ID[10:17]+LANDSAT_ID[17:25]
        ls_trial=ls_trial.addBands([ls_trial.select('ET_24h').rename(NAME_FINAL)])
        ET_daily=ls_trial.select(['ET_24h'],[NAME_FINAL])
        if k ==0:
            Collection_ET=ET_daily
            Collection_H=ls_trial.select(['H'],[NAME_FINAL])
            Collection_Rn_daily=Rn24hobs.select(['Rn24h_G'],[NAME_FINAL])
              #AIR TEMPERATURE [C]
            Collection_Ta=T_air.select(['AirT_G'],[NAME_FINAL]);

            #WIND SPEED [M S-1]
            Collection_ux=ux.select(['ux_G'],[NAME_FINAL]);
            #RELATIVE HUMIDITY [%]
            Collection_rh = UR.select(['RH_G'],[NAME_FINAL]);

        else:
            Collection_ET=Collection_ET.addBands(ET_daily)
            Collection_H=Collection_H.addBands(ls_trial.select(['H'],[NAME_FINAL]))
            Collection_Rn_daily=Collection_Rn_daily.addBands(Rn24hobs.select(['Rn24h_G'],[NAME_FINAL]))
            Collection_Ta=Collection_Ta.addBands(T_air.select(['AirT_G'],[NAME_FINAL]))
            Collection_ux=Collection_ux.addBands(ux.select(['ux_G'],[NAME_FINAL]))
            Collection_rh=Collection_rh.addBands(UR.select(['RH_G'],[NAME_FINAL]))
       
        k=k+1
        print(k)
        n=n+1
        print(n)
    return(Collection_ET,Collection_H,Collection_Rn_daily)




