import src.cleanup as cleanup
import src.trainmodel as trainmodel
import numpy as np
import pandas as pd
import joblib

# Get path variables
src_path, out_path = cleanup.build_path()
# Parse csv
csv = cleanup.get_csv(src_path)
# Clean up the csv
csv = cleanup.clean_csv(csv)
# Save the csv
cleanup.save_csv(csv, out_path)

# Prepare data, filter out columns, split dataset, scale dataset
X_train, X_test, y_train, y_test, y, scaler, encoder = trainmodel.prep_data(csv)

# Train the models
models = [trainmodel.train_LinearRegression, trainmodel.train_DecisionTreeRegressor, trainmodel.train_XGBRegressor, trainmodel.train_SGDRegressor, trainmodel.train_NeuralNetwork]
# Prep the score DataFrame
total_scores = pd.DataFrame(columns=['train_score', 'test_score', 'rmse', 'coef_determination'])

# Look through the different regression models
for model in models:
    regressor = model(X_train, y_train)
    score_train, score_test, rmse, coef_determination = trainmodel.score(regressor, X_train, X_test, y_train, y_test, y)
    score = {}
    score['train_score'] = score_train
    score['test_score'] = score_test
    score['rmse'] = rmse
    score['coef_determination'] = coef_determination
    total_scores = total_scores._append(score, ignore_index=True)

    if 'train_XGBRegressor' in str(model):
        regressor.save_model('models/xgbmodel.model')
        scaler_filename = "models/scaler.save"
        joblib.dump(scaler, scaler_filename) 
        encoder_filename = "models/encoder.save"
        joblib.dump(encoder, encoder_filename)

total_scores.index = [(str(model).rsplit("_")[1]).split(" ")[0] for model in models]
print(total_scores)