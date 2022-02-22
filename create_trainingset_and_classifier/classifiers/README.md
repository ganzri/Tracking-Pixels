Scripts to train the classifier and extract interesting misclassificatins to analyse.

Scripts:
Scripts used in earlier iterations of model training and evaluation.
* [analyse_misclassfied] checks which domains are most commonly found in necessary but predicted as analytics or advertising. This is the basis for the reclassification of samples in the training_data_output_offline model folder. It uses the result of one of the crossvalidations steps when training the model.
* [misclassified.py] used to extract misclassfied samples from one of the crossvalidation folds. Used in earlier analysis

Scripts used for thesis:
* [train_xgb.py] Train the classifier or conduct parameter search or cross validation. Script from Dino Bollinger.
* [utils.py] utility functions used by train_xgb.py], from Dino Bollinger
* [training_stats.py] calculate mean balanced accuracy, precision and recall per category over five folds of cross validation. The values have to be manually added.
* [trees_display.py] display a tree of a trained model, to visually inspect it
* [xgboost_stats.py] outputs statistics about feature importance etc of a trained model.
* [predict_and_analyse.py] Takes a trained model and validation data and predicts labels for the validation data. Then compares these to ground truth, creating a confusion matrix and saving a plot thereof. It also compares the ground truth and predictions to filter list decisions (to block or not to block), and saves the correspondinng figures. Finally outputs and saves random samples of each interesting category.
