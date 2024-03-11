####################################################################
#------------------------------------------------------------------#
#Code base orginal 
#Salt marsh probability and spatial uncertainty                             #
#Filename: "Saltmarsh_uncertainty.R"                          
#sections adapted from Evan R. DeLancey "Boreal_PeatlandProbability.R" code

#Load libraries
library(raster)
library(rgdal)
library(ggplot2)
library(dplyr)
library(caret)
library(snow)
library(rgeos)
library(RPyGeo)
library(dismo)
library(gbm)
library(RStoolbox)
library(ggthemes)
library(gmodels)
library(e1071)
library(mlr)
library(dplyr)
library(tidyr)
library(reshape2)
library(mlr)
library(devtools)
library(mlrMBO)
tt1 <- Sys.time()

#location of input rasters
location <- "C:/Work/biomass/bio_030801/prob"
list.files(location)


#set location of a temporary raster dump
#this can take up 100-300BG per run but is deleted after
rasterOptions(maxmemory = 1e+09,tmpdir = "C:/Work/temp")

?rasterOptions

#----------------------------------------------------------------
#----------------------------------------------------------------
#################################################################



#Name vars and get min and max for response curves
###################################################################
#------------------------------------------------------------------
#------------------------------------------------------------------
#set input varibles
list.files(location)
tifs <- c("B2-0000000000-0000000000.tif","B3-0000000000-0000000000.tif","B4-0000000000-0000000000.tif","B8-0000000000-0000000000.tif","NDVI-0000000000-0000000000.tif", "NDWI-0000000000-0000000000.tif", "NED-0000000000-0000000000.tif",'SRTM-0000000000-0000000000.tif', "VV-0000000000-0000000000.tif", "VH-0000000000-0000000000.tif")


fls <- tifs



#----------------------------------------------------------------


#build raster brick
setwd(location)
fls <- tifs
#build raster brick
r1 <- raster(fls[1])
#r1 <- crop(r, PUs)

r2 <- raster(fls[2])
#r2 <- crop(r, PUs)

r3 <- raster(fls[3])
#r3 <- crop(r, PUs)

r4 <- raster(fls[4])
# <- crop(r, PUs)

r5 <- raster(fls[5])
#r5 <- crop(r, PUs)

r6 <- raster(fls[6])
#r6 <- crop(r, PUs)

r7 <- raster(fls[7])
#r7 <- crop(r, PUs)

r8 <- raster(fls[8])
#r8 <- crop(r, PUs)

r9 <- raster(fls[9])
#r9 <- crop(r, PUs)

r10 <- raster(fls[10])
#r10 <- crop(r, PUs)

viirsbrick <- stack(r1,r2,r3,r4,r9,r10)







library(sf)


#list.files('C:/Work/biomass/bio_020301/prob/')
list.files('C:/Work/biomass/bio_030501/prob/')
#conver to poitns
fc=rasterToPoints(r1,digits=12,progress='text')
#points to dataframe
fc=as.data.frame(fc)
train = readOGR('C:/Work/biomass/bio_030801/prob/train_pts_data.shp')
train$binary_sm[which(is.na(train$binary_sm))]=0
train$sm = as.character(train$binary_sm)
train$binary_sm = NULL
train$CID = NULL
train$REIP = NULL
train[which(is.na(train$NDWI)),]
#rename to the names of the raster data
names(train) = c('VV','VH',"SRTM","NED",'NDWI','NDVI','B8','B4',"B3","B2","sm")
train = train[-which(is.na(train$NDWI)),]
train$sm=as.factor(train$sm)
train$SRTM[which(is.na(train$SRTM))]=-9999
train$NED[which(is.na(train$NED))]=-9999
#create classification task
classif_task = makeClassifTask(data=train@data,target="sm",positive ="1")
#create learners including support vector machines, rotational forest and xgboost
rotForest_lrn = makeLearner("classif.rotationForest",predict.type="prob")
xgb_learner = makeLearner("classif.xgboost",predict.type="prob",par.vals=list(objective="binary:logistic",eval_metric="error",nrounds=200))
svm_lrn = makeLearner("classif.svm",predict.type="prob")

#controls
ctrl = makeTuneControlRandom(maxit=100L)
rdesc = makeResampleDesc("CV")
#parameter set
xgb_params = makeParamSet(
  makeIntegerParam("nrounds",lower=100,upper=500),
  makeIntegerParam("max_depth",lower=1,upper=10),
  makeNumericParam("eta", lower=0.1,upper=0.5),
  makeNumericParam("lambda",lower=-1,upper=0,trafo=function(x) 10^x)
)
control = makeTuneControlRandom(maxit=50)
resample_desc = makeResampleDesc("CV",iters=4)
tuned_xgb = tuneParams(learner = xgb_learner, task=classif_task,resampling=resample_desc,par.set = xgb_params, control=control)
xgb_tuned_learner = setHyperPars(learner=xgb_learner,par.vals=tuned_xgb$x)


ps = makeParamSet(
  makeNumericParam("C", lower = -5, upper = 5, trafo = function(x) 2^x)
)
ctrl = makeTuneControlRandom(maxit = 100L)
lrn = makeHyperoptWrapper(classif_task)
res = tuneParams("classif.svm", task = classif_task, control = ctrl,
                 measures = list(acc, mmce), resampling = rdesc, par.set = ps, show.info = FALSE)
generateHyperParsEffectData(res, trafo = T, include.diagnostics = FALSE)


par.set = makeParamSet(
  makeNumericParam("cost", 0, 15, trafo = function(x) 2^x),
  makeNumericParam("gamma", 0, 15, trafo = function(x) 2^x)
)
#support vector machines classifier


#tuning
svm = makeSingleObjectiveFunction(name = "svm.tuning",
                                  fn = function(x) {
                                    lrn = makeLearner("classif.svm", par.vals = x)
                                    resample(lrn, classif_task, cv3, show.info = FALSE)$aggr
                                  },
                                  par.set = par.set,
                                  noisy = TRUE,
                                  has.simple.signature = FALSE,
                                  minimize = FALSE
)
ctrl = makeMBOControl()
ctrl = setMBOControlTermination(ctrl, iters = 5)
svm_lrn =setHyperPars(svm_lrn,cost = res$x$cost,gamma=res$x$gamma)

res = mbo(svm, control = ctrl, show.info = FALSE)
print(res)
op = as.data.frame(res$opt.path)
plot(cummin(op$y), type = "l", ylab = "mmce", xlab = "iteration")
mod_svm = train(svm_lrn,classif_task)
#train models with parameter tuning
xgb_model = train(xgb_tuned_learner,classif_task)
mod_svm = train(svm_lrn,classif_task)
mod_rot = train(rotForest_lrn,classif_task)

#internal cross validation with benchmark
benchmark(rotForest_lrn,classif_task,rdesc,acc)
benchmark(svm_lrn,classif_task,rdesc,acc)
benchmark(xgb_learner,classif_task,rdesc,acc)
#classify using the resulting trained models and compute a spatial confidence interval
#extract raster values with interpolation when necessary
NED=raster::extract(r7,fc[,-c(3)],method='simple')
NDVI=raster::extract(r5,fc[,-c(3)],method='simple')
NDWI=raster::extract(r6,fc[,-c(3)],method='simple')
all=raster::extract(viirsbrick,fc3[,-c(3)],method='simple')
srtm=raster::extract(r8,fc3[,-c(3)],method='simple')
#add data extracts to data frame
fc_class=cbind(NED,all)
fc_class=cbind(fc_class,srtm)
fc_class=cbind(fc_class,NDVI)
fc_class=cbind(fc_class,NDWI)
fc_class = as.data.frame(fc_class)
#define names
names(fc_class) = c("NED","B2","B3","B4","B8","VV","VH","SRTM","NDVI","NDWI")
#add nodata value
if(length(which(is.na(fc_class$NED)))!=0){
    fc_class$NED[which(is.na(fc_class$NED))]= -9999
}
#add nodata value
if(length(which(is.na(fc_class$SRTM)))!=0){
  fc_class$SRTM[which(is.na(fc_class$SRTM))]= -9999
}
#predict models
predrot = predict(mod_rot, newdata=fc_class[,c(6,7,8,1,10,9,5,4,3,2)],probability=TRUE)
predsvm = predict(mod_svm, newdata=fc_class[,c(6,7,8,1,10,9,5,4,3,2)],probability=TRUE)
predxgb = predict(xgb_model, newdata=fc_class[,c(6,7,8,1,10,9,5,4,3,2)],probability=TRUE)
name1= paste0("probs_",seqz,"svm",".csv")
#add prediction outputs to dataframe
fc$probROT = predrot$data$prob.1
fc$probSVM = predsvm$data$prob.1
fc$probxgb = predxgb$data$prob.1
#compute confidence interval
cis=apply(fc[,4:6],1,function(x) ci(x))
cis2 = dcast(melt(cis),Var2~Var1)
#combined data frame and CI
fc = cbind(fc,cis2)
fc$SMlse = 0
fc$SMlse[which((fc$Estimate-fc$`Std. Error`)>0.5)]=1
fc$SMuse = 0
fc$SMuse[which((fc$Estimate+fc$`Std. Error`)>0.5)]=1
fc$SM = 0
fc$SM[which((fc$Estimate)>0.5)]=1
write.csv(fc,name1)  