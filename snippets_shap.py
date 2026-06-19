# SHAP — which features the model relies on, and in which direction
# feature_importances_ tells you HOW MUCH a feature matters; SHAP also tells you
# whether it pushes the prediction toward "up" or "down".
# Works for tree models (Random Forest / XGBoost), not for the LSTM.
# Install once:  pip install shap

import shap

# Train on the last (largest) training fold
splits = list(tscv.split(X))
train_idx, test_idx = splits[-1]
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train = y.iloc[train_idx]

model_shap = RandomForestClassifier(n_estimators=100, random_state=42)
model_shap.fit(X_train, y_train)

# Compute SHAP values on the test fold
explainer = shap.TreeExplainer(model_shap)
shap_values = explainer.shap_values(X_test)

# For a binary Random Forest, shap_values may be a list [class_0, class_1] —
# keep class 1 ("up"). With newer versions it is already a single array.
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Summary plot: most influential features on top
shap.summary_plot(shap_values, X_test, feature_names=features)
