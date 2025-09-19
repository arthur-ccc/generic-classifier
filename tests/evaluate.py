from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    try:
        y_proba = model.predict_proba(X_test)[:,1]
        print("ROC AUC:", roc_auc_score(y_test, y_proba))
    except NotImplementedError:
        print("ROC AUC não disponível para este classificador")
