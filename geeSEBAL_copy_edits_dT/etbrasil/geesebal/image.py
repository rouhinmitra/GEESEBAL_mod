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
#ee.Initialize()

#FOLDERS
from .masks import (
f_cloudMaskL457_SR,f_cloudMaskL8_SR,
 f_albedoL5L7,f_albedoL8)
from .meteorology import get_meteorology
from .tools import (fexp_spec_ind, fexp_lst_export,fexp_radlong_up, LST_DEM_correction,
fexp_radshort_down, fexp_radlong_down, fexp_radbalance, fexp_soil_heat,fexp_sensible_heat_flux)
from .endmembers import fexp_cold_pixel, fexp_hot_pixel
from .evapotranspiration import fexp_et


#IMAGE FUNCTION
class Image():

    #ENDMEMBERS DEFAULT
    #ALLEN ET AL. (2013)
    def __init__(self,
                 image,
                 NDVI_cold=5,
                 Ts_cold=20,
                 NDVI_hot=10,
                 Ts_hot=20):

        #GET INFORMATIONS FROM IMAGE
        self.image = ee.Image(image)
        self._index=self.image.get('system:index')
        self.cloud_cover=self.image.get('CLOUD_COVER')
        self.LANDSAT_ID=self.image.get('LANDSAT_ID').getInfo()
        self.landsat_version=self.image.get('SATELLITE').getInfo()
        self.azimuth_angle=self.image.get('SOLAR_ZENITH_ANGLE')
        self.time_start=self.image.get('system:time_start')
        self._date=ee.Date(self.time_start)
        self._year=ee.Number(self._date.get('year'))
        self._month=ee.Number(self._date.get('month'))
        self._day=ee.Number(self._date.get('day'))
        self._hour=ee.Number(self._date.get('hour'))
        self._minuts = ee.Number(self._date.get('minutes'))
        self.crs = self.image.projection().crs()
        self.transform = ee.List(ee.Dictionary(ee.Algorithms.Describe(self.image.projection())).get('transform'))
        self.date_string=self._date.format('YYYY-MM-dd').getInfo()
        print(1)
        #ENDMEMBERS
        self.p_top_NDVI=ee.Number(NDVI_cold)
        self.p_coldest_Ts=ee.Number(Ts_cold)
        self.p_lowest_NDVI=ee.Number(NDVI_hot)
        self.p_hottest_Ts=ee.Number(Ts_hot)

        #LANDSAT IMAGE
        if self.landsat_version == 'LANDSAT_5':
             self.image=self.image.select([0,1,2,3,4,5,6,9], ["B","GR","R","NIR","SWIR_1","BRT","SWIR_2", "pixel_qa"])
             self.image_toa=ee.Image('LANDSAT/LT05/C01/T1/'+ self._index.getInfo())

         #GET CALIBRATED RADIANCE
             self.col_rad = ee.Algorithms.Landsat.calibratedRadiance(self.image_toa);
             self.col_rad = self.image.addBands(self.col_rad.select([5],["T_RAD"]))

         #CLOUD REMOTION
             self.image=ee.ImageCollection(self.image).map(f_cloudMaskL457_SR)

         #ALBEDO TASUMI ET AL. (2008)
             self.image=self.image.map(f_albedoL5L7)

        elif self.landsat_version == 'LANDSAT_7':
             self.image=self.image.select([0,1,2,3,4,5,6,9], ["B","GR","R","NIR","SWIR_1","BRT","SWIR_2", "pixel_qa"])
             self.image_toa=ee.Image('LANDSAT/LE07/C01/T1/'+ self._index.getInfo())

         #GET CALIBRATED RADIANCE
             self.col_rad = ee.Algorithms.Landsat.calibratedRadiance(self.image_toa);
             self.col_rad = self.image.addBands(self.col_rad.select([5],["T_RAD"]))

         #CLOUD REMOVAL
             self.image=ee.ImageCollection(self.image).map(f_cloudMaskL457_SR)

         #ALBEDO TASUMI ET AL. (2008)
             self.image=self.image.map(f_albedoL5L7)

        else:
            self.image = self.image.select([0,1,2,3,4,5,6,7,10],["UB","B","GR","R","NIR","SWIR_1","SWIR_2","BRT","pixel_qa"])
            self.image_toa=ee.Image('LANDSAT/LC08/C01/T1/'+self._index.getInfo())

         #GET CALIBRATED RADIANCE
            self.col_rad = ee.Algorithms.Landsat.calibratedRadiance(self.image_toa)
            self.col_rad = self.image.addBands(self.col_rad.select([9],["T_RAD"]))
            print(2)
         #CLOUD REMOVAL
#             self.image=ee.ImageCollection(self.image).map(f_cloudMaskL8_SR)
            self.image=f_cloudMaskL8_SR(self.image)
         #ALBEDO TASUMI ET AL. (2008) METHOD WITH KE ET AL. (2016) COEFFICIENTS
#             self.image=self.image.map(f_albedoL8)
            self.image=f_albedoL8(self.image)


        #GEOMETRY
        self.geometryReducer=self.image.geometry().bounds().getInfo()
        self.geometry_download=self.geometryReducer['coordinates']
#         self.camada_clip=self.image.select('BRT').first()
        self.camada_clip=self.image.select('BRT')
        print(3)
        self.sun_elevation=ee.Number(90).subtract(ee.Number(self.azimuth_angle))
        print(4)
        #METEOROLOGY PARAMETERS
        col_meteorology= get_meteorology(self.image,self.time_start);

        #AIR TEMPERATURE [C]
        self.T_air = col_meteorology.select('AirT_G');

        #WIND SPEED [M S-1]
        self.ux= col_meteorology.select('ux_G');

        #RELATIVE HUMIDITY [%]
        self.UR = col_meteorology.select('RH_G');

        #NET RADIATION 24H [W M-2]
        self.Rn24hobs = col_meteorology.select('Rn24h_G');

        #SRTM DATA ELEVATION
        SRTM_ELEVATION ='USGS/SRTMGL1_003'
        self.srtm = ee.Image(SRTM_ELEVATION).clip(self.geometryReducer);
        self.z_alt = self.srtm.select('elevation');

        #GET IMAGE
#         self.image=self.image.first()

        #SPECTRAL IMAGES (NDVI, EVI, SAVI, LAI, T_LST, e_0, e_NB, long, lat)
        self.image=fexp_spec_ind(self.image)

        #LAND SURFACE TEMPERATURE
        self.image=LST_DEM_correction(self.image, self.z_alt, self.T_air, self.UR,self.sun_elevation,self._hour,self._minuts)

        #COLD PIXEL
        self.d_cold_pixel=fexp_cold_pixel(self.image, self.geometryReducer, self.p_top_NDVI, self.p_coldest_Ts)

        #COLD PIXEL NUMBER
        self.n_Ts_cold = ee.Number(self.d_cold_pixel.get('temp').getInfo())

        #INSTANTANEOUS OUTGOING LONG-WAVE RADIATION [W M-2]
        self.image=fexp_radlong_up(self.image)

        #INSTANTANEOUS INCOMING SHORT-WAVE RADIATION [W M-2]
        self.image=fexp_radshort_down(self.image,self.z_alt,self.T_air,self.UR, self.sun_elevation)

        #INSTANTANEOUS INCOMING LONGWAVE RADIATION [W M-2]
        self.image=fexp_radlong_down(self.image, self.n_Ts_cold)

        #INSTANTANEOUS NET RADIATON BALANCE [W M-2]
        self.image=fexp_radbalance(self.image)

        #SOIL HEAT FLUX (G) [W M-2]
        self.image=fexp_soil_heat(self.image)

        #HOT PIXEL
        self.d_hot_pixel=fexp_hot_pixel(self.image, self.geometryReducer,self.p_lowest_NDVI, self.p_hottest_Ts)

        #SENSIBLE HEAT FLUX (H) [W M-2]
        self.image=fexp_sensible_heat_flux(self.image, self.ux, self.UR,self.Rn24hobs,self.n_Ts_cold,
                                           self.d_hot_pixel, self.date_string,self.geometryReducer)

        #DAILY EVAPOTRANSPIRATION (ET_24H) [MM DAY-1]
        self.image=fexp_et(self.image,self.Rn24hobs)

        self.NAME_FINAL=self.LANDSAT_ID[:5]+self.LANDSAT_ID[10:17]+self.LANDSAT_ID[17:25]
        self.image=self.image.addBands([self.image.select('ET_24h').rename(self.NAME_FINAL)])
