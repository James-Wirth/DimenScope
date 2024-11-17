import shap

def calculate_shap_values(model, data):
    explainer = shap.KernelExplainer(model.predict, data[:100])
    shap_values = explainer.shap_values(data)
    return shap_values
