from src.config import ARTIFACTS_DIR, DATA_DIR


def test_artifact_files_exist():
    required_artifacts = [
        ARTIFACTS_DIR / "model.pkl",
        ARTIFACTS_DIR / "preprocessor.pkl",
        ARTIFACTS_DIR / "metrics.json",
    ]
    for path in required_artifacts:
        assert path.exists(), f"Missing required artifact: {path}"


def test_dataset_exists():
    data_path = DATA_DIR / "processed" / "clean_data.csv"
    assert data_path.exists(), f"Dataset not found: {data_path}"
