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

#CLOUD REMOVAL

#FUNCTION TO MASK CLOUDS IN LANDSAT 5 AND 7 FOR SURFACE REFLECTANCE

def f_cloudMaskL457_SR(image):
    quality = image.select('pixel_qa');
    c01 = quality.eq(66);#CLEAR, LOW CONFIDENCE CLOUD
    c02 = quality.eq(68);#WATER, LOW CONFIDENCE CLOUD
    mask = c01.Or(c02);
    return image.updateMask(mask);

#FUNCTION FO MASK CLOUD IN LANDSAT 8 FOR SURFACE REFELCTANCE
def f_cloudMaskL8_SR(image):
    quality = image.select('pixel_qa');
    c01 = quality.eq(21824); #CLEAR, LAND LOW CONFIDENCE CLOUD,BITS NOT SET
    c02 = quality.eq(21952); #CLEAR, WATER, LOW CONFIDENCE CLOUD,BITS NOT SET
    c03=quality.eq(21888) #CLEAR, WATER, LOW CONFIDENCE CLOUD,BITS  SET
    c04 = quality.eq(21756);#CLEAR, LAND, LOW CONFIDENCE CLOUD,BITS  SET
    mask = c01.Or(c02).Or(c03).Or(c04);
    return image.updateMask(mask);

#ALBEDO
#TASUMI ET AL(2008) FOR LANDSAT 5 AND 7
def f_albedoL5L7(image):

    alfa = image.expression(
      '(0.254*B1) + (0.149*B2) + (0.147*B3) + (0.311*B4) + (0.103*B5) + (0.036*B7)',{
        'B1' : image.select(['SR_B2']),
        'B2' : image.select(['SR_B3']),
        'B3' : image.select(['SR_B4']),
        'B4' : image.select(['SR_B5']),
        'B5' : image.select(['SR_B6']),
        'B7' : image.select(['SR_B7'])
      }).rename('ALFA');

    #ADD BANDS
    return image.addBands(alfa);

#ALBEDO
#USING TASUMI ET AL. (2008) METHOD FOR LANDSAT 8
#COEFFICIENTS FROM KE ET AL. (2016)
def f_albedoL8(image):
    alfa = image.expression(
      '(0.130*B1) + (0.115*B2) + (0.143*B3) + (0.180*B4) + (0.281*B5) + (0.108*B6) + (0.042*B7)',{  #// (Ke, Im  et al 2016)
        'B1' : image.select(['UB']),
        'B2' : image.select(['B']),
        'B3' : image.select(['GR']),
        'B4' : image.select(['R']),
        'B5' : image.select(['NIR']),
        'B6' : image.select(['SWIR_1']),
        'B7' : image.select(['SWIR_2'])
      }).rename('ALFA');

    #ADD BANDS
    return image.addBands(alfa);

if __name__ == "__main__":
    f_albedoL8()
    f_albedoL5L7()
    f_cloudMaskL8_SR()
    f_cloudMaskL457_SR()
