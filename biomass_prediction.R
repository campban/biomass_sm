#apply xgboost model to biomass prediction
library(raster)
library(mlr)
library(caret)
library(rgdal)
library(stringr)
library(stars)
library(dplyr)
library(ranger)
library(fasterRaster)
library(SpaDES)
#file location with tifs
rasters = list.files('C:/Work/biomass/ready',full.names=TRUE)
#change directory
setwd('C:/Work/biomass/ready')
# find only tifs
rasters = str_subset(rasters,'.tif')
#remove xml files
rasters = rasters[-grep(pattern='.xml', rasters)]
library(ggplot2)
library(tidyverse)
# loop through files predicting biomass
for(ras in rasters){
  raster.st=stack(ras)
  name1 =  strsplit(ras, "/")[[1]][6]
  #name1 =  strsplit(ras, "/")[[1]][5]
  name1 = substr(name1,1,nchar(name1)-4)
  raster.st=stack(ras)
  #conver to raster
  rpts2 = rasterToPoints(raster.st,spatial=TRUE,progress="text") 
  #change raster names to match features in xgb model
  names(rpts2) = c("b1","b2","b3","b4","b5","b7",'wdrvi5','savi','nd_r_g','nd_g_b','nd_swir2_r','nd_swir2_n',"B2mean","B3mean","B4mean","B8mean","ndvimean","VHmean","VVsample")
  #define x and y location data
  rpts2$centroid_y = rpts2@coords[,2]
  rpts2$centroid_x = rpts2@coords[,1]
  #predic biomass
  xgb_v2_p=predict(xgb_v2_re,newdata=rpts2@data[,c(20,21,8,1,2,3,4,5,6,7,9,10,11,12,18,19,17,13,14,15,16)],na.rm=TRUE)
  #add to new 
  rpts2$xgb_v3=xgb_v2_p$data$response
  name1 = paste(name1,i,sep="_")
  name1z = paste0('bio',name1,'xgb',sep="_")
  names(rpts2)
  raster_df = rasterize(rpts2,raster.st,field="xgb_v3")
  name1z2 = paste(name1z,'tif',sep=".")
  writeRaster(raster_df,name1z2)
}