The datasets provided have been generated with 
[responses_models globalHI localHI] = health_indicators_estimation_cmapss(1, [7 8 9 10 12 14 16 17 20 25 26], 6, true);
data_train_modele_morceaux = globalHI for dataset 1 and 3
or localHI for 2 and 4

Repository SINGLE contains files .mat named "data_indicators_turbofanXX_YY_singleSubsetTable5.mat" where XX=1,2,3,4 and YY=a set of features corresponding to the ones in Table of the paper [1]. The same holds for directory MULTIPLE which contains 2 files per dataset which can be used for fusion or independently as described in [1].

It allows to load two cells:

- data_train_modele_morceaux{j}, j=1:N, N depends on the dataset (100, 248, 249) => contains the health indicator for training instance "j" in the corresponding dataset XX, from t=1 to the time of failure. Each indicator was computed by a model Mj described in Eq. 11 and 12 in [1]. Each model is assigned a Remaining Useful Life (RUL) available at NASA PCOE (see download_RULs_FromNasaPCOE.txt), i have kept the same order.

- responses_models{i}(1:t, j) => contains the i-th testing instance indicator from 1:t calculed by the model Mj, j=1:N. 

The goal with similarity approach is to find the best model (the best "j" or a combination) to predict the remaining lifetime of the testing instance by learning the decision rule that allows to decide which "j" (or the parameter of the combination) is the best. You can also used the training health indicators for other purposes. 


########################################################################
TURBOFAN DATASETS
########################################################################

--------------------------------------------------------------------------
USING SINGLE PARAMETERIZATION TABLES 5 AND 6
--------------------------------------------------------------------------

Dataset #1 (Table 5 of the paper)
Features : [7 8 9 10 12 14 16 17 20 25 26]
data_indicators_turbofan1_9_10_25_26_singleSubsetTable5.mat

Dataset #2 (Table 6 of the paper)
Features : [7 8 12 16 17 20 9 13 25]
Local computation of health indicators using operating conditions
data_indicators_turbofan2_9_13_25_singleSubsetTable6.mat

Dataset #3 (Table 5 of the paper)
Features : [7 8 12 16 17 20 9 13 14 22 26]
data_indicators_turbofan3_9_13_14_22_26_singleSubsetTable5.mat

Dataset #4 (Table 6 of the paper)
Features : [7 8 12 16 17 20 9 10 11 22]
data_indicators_turbofan4_9_10_11_22_singleSubsetTable6.mat

--------------------------------------------------------------------------
USING ENSEMBLES (MULTIPLE PARAMETERIZATIONS) TABLE 7
--------------------------------------------------------------------------

I have proposed in the paper to combine the RULs estimated by various subsets of features, results are much better.
Below i provide the two sets of indicators for each dataset used to build Table 7.

Dataset #1
Features : [7 8 12 16 17 20 10 11 14 22]
data_indicators_turbofan1_multipleParamCase_10_11_14_22.mat
Features : [7 8 12 16 17 20 13 18 19 22]
data_indicators_turbofan1_multipleParamCase_13_18_19_22.mat

Dataset #2 
Features : [7 8 12 16 17 20 9 10 13 26]
data_indicators_turbofan2_multipleParamCase_9_10_13_26.mat
Features : [7 8 12 16 17 20 9 13 14 26]
data_indicators_turbofan2_multipleParamCase_9_13_14_26.mat

Dataset #3
Features : [7 8 12 16 17 20 13 19 25 26]
data_indicators_turbofan3_multipleParamCase_13_19_25_26.mat
Features : [7 8 12 16 17 20 13 9 13 14 22 26]
data_indicators_turbofan3_multipleParamCase_9_13_14_22_26.mat

Dataset #4
Features : [7 8 12 16 17 20 9 10 11 25 26]
data_indicators_turbofan4_multipleParamCase_9_10_11_25_26.mat
Features : [7 8 12 16 17 20 13 9 13 22 25]
data_indicators_turbofan4_multipleParamCase_9_13_22_25.mat


