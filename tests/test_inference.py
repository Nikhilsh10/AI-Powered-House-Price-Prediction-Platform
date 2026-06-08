import pytest
from src.inference.predict import predict_price

def test_predict_price_valid_input():
    input_data = {
        "location": "Whitefield",
        "size": 1200.0,
        "bhk": 2,
        "bath": 2,
    }
    result = predict_price(input_data)
    assert isinstance(result, dict)
    assert "predicted_price" in result
    assert "lower_bound" in result
    assert "upper_bound" in result
    
    assert isinstance(result["predicted_price"], float)
    assert result["predicted_price"] > 0
    assert result["lower_bound"] < result["predicted_price"]
    assert result["upper_bound"] > result["predicted_price"]

def test_predict_price_invalid_size():
    with pytest.raises(ValueError, match="Total Area must be a positive number"):
        predict_price({
            "location": "Whitefield",
            "size": -100,
            "bhk": 2,
            "bath": 2,
        })

def test_predict_price_missing_location():
    with pytest.raises(ValueError, match="Location cannot be empty"):
        predict_price({
            "location": "   ",
            "size": 1200,
            "bhk": 2,
            "bath": 2,
        })

def test_predict_price_invalid_bhk():
    with pytest.raises(ValueError, match="BHK must be at least 1"):
        predict_price({
            "location": "Whitefield",
            "size": 1200,
            "bhk": 0,
            "bath": 2,
        })

def test_shap_explainer_works():
    import shap
    import pandas as pd
    from src.inference.predict import load_model, load_preprocessor, preprocess_input
    
    model = load_model()
    input_data = {
        "location": "Whitefield",
        "size": 1200.0,
        "bhk": 2,
        "bath": 2,
    }
    
    X_processed = preprocess_input(input_data)
    
    assert hasattr(model, "get_booster") or hasattr(model, "feature_name_"), "Model must be a tree model for TreeExplainer"
    explainer = shap.TreeExplainer(model)
    
    shap_values = explainer.shap_values(X_processed)
    assert shap_values is not None
    assert len(shap_values) > 0
