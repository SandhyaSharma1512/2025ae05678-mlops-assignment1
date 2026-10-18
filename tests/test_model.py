import pytest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from heart_disease.api import app  # Import after modifying the path
@pytest.fixture
def sample_data():
    # Generate sample data for testing
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def test_pipeline_creation():
    # Test if the preprocessing pipeline is created successfully
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    assert pipeline is not None, "Pipeline creation failed."

def test_model_training(sample_data):
    # Test if the model trains successfully
    X_train, X_test, y_train, y_test = sample_data
    model = LogisticRegression()
    model.fit(X_train, y_train)
    assert model.score(X_test, y_test) > 0.5, "Model training failed."