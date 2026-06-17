# Predicting the Past
 
Vasileios Dalampiras, Carla Mailin Friedrich, Aikaterini Karathanasi, Peter Resch

## Machine Learning Weather Prediction Project
in the Course Applied Machine Learning (Troels Petersen), 2026, University of Copenhagen (https://www.nbi.dk/~petersen/Teaching/AppliedMachineLearning2026.html)


## Aim
Building ML models to predict a forecast vector from a timeseries and find out which variables are important for precipitation.

## Data used
ERA5 (https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview)

## Outcome: Single Point Feature Importance
<img width="3000" height="1000" alt="XGBoost Feature Importance" src="topics/Feature Importance - Vasilis/XGboost_pred_vs_true_1h.png">
<img width="400" height="400" alt="XGBoost Confusion Matrix" src="topics/Feature Importance - Vasilis/XGBoost_confusion_matrix_1h.png">
<img width="400" height="400" alt="XGBoost Feature Importance" src="topics/Feature Importance - Vasilis/XGboost_feature_importance.png">

## Outcome: Small grid cross-predicting seasons
<img width="3000" height="1000" alt="losses_vs_epochs_recent" src="topics/training on wrong season - Peter/images/losses_vs_epochs_recent+LSTM-HPT.png">
<img width="3000" height="1000" alt="GRU_test_losses_combined_years" src="topics/training on wrong season - Peter/images/GRU_losses_combined_plots.png">

## Outcome: Small grid hindcasting with recent Years training
<img width="3000" height="1000" alt="Precipitation Forecast for 1980" src="topics/training on wrong years - Mailin/plots/precip_map_1980-01-01.png">
<img width="400" height="400" alt="ROC curve for model classifying rain or not" src="topics/training on wrong years - Mailin/plots/roc_curve.png">
