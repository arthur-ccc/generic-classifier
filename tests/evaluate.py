from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

def evaluate_model(model, X_test, y_test, le=None):
    """
    Avalia um modelo de classificação multiclasse.

    Parâmetros:
    - model: classificador treinado (GenericClassifier)
    - X_test: features de teste
    - y_test: labels de teste (numéricos)
    - le: LabelEncoder usado no pré-processamento (opcional)
    """

    # Fazer predição
    y_pred = model.predict(X_test)

    # Converter para strings se LabelEncoder foi passado
    if le:
        y_test_labels = le.inverse_transform(y_test)
        y_pred_labels = le.inverse_transform(y_pred)
    else:
        y_test_labels = y_test
        y_pred_labels = y_pred

    # Accuracy
    print("Accuracy:", accuracy_score(y_test_labels, y_pred_labels))

    # Classification report
    print("\nClassification Report:\n", classification_report(
        y_test_labels,
        y_pred_labels,
        target_names=le.classes_ if le else None
    ))

    # ROC AUC (tenta multiclasse)
    try:
        y_proba = model.predict_proba(X_test)
        if y_proba.shape[1] > 2:
            # Multiclasse: one-vs-rest
            y_test_bin = label_binarize(y_test, classes=range(y_proba.shape[1]))
            roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr")
        else:
            # Binário
            roc_auc = roc_auc_score(y_test, y_proba[:,1])
        print("ROC AUC:", roc_auc)
    except (NotImplementedError, ValueError):
        print("ROC AUC não disponível para este classificador")
