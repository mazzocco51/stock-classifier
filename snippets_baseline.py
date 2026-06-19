# Baseline — does the model actually beat "always predict up"?
# If a model can't beat the most-frequent-class guess, it has no real edge.
# Paste this cell after the XGBoost cell (it reuses tscv, accuracies, accuracies_xgb).

from sklearn.dummy import DummyClassifier

accuracies_baseline = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Always predict the most frequent class in the training set (usually "up")
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    acc = accuracy_score(y_test, baseline.predict(X_test))
    accuracies_baseline.append(acc)
    print(f"Fold {fold+1} — Baseline Accuracy: {acc:.2%}")

print(f"\nBaseline Mean accuracy: {np.mean(accuracies_baseline):.2%}")
print(f"Random Forest Mean accuracy: {np.mean(accuracies):.2%}")
print(f"XGBoost Mean accuracy: {np.mean(accuracies_xgb):.2%}")

# If the models are not clearly above the baseline, there is no real edge —
# which is an honest and valid conclusion to report.
