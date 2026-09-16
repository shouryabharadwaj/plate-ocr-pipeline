import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.classifier import CharacterClassifier


def _make_toy_dataset(n_per_class=15, n_classes=4, dim=20, seed=0):
    rng = np.random.default_rng(seed)
    labels = [str(i) for i in range(n_classes)]
    X, y = [], []
    for i, label in enumerate(labels):
        center = rng.normal(loc=i * 5, scale=0.5, size=dim)
        samples = center + rng.normal(scale=0.3, size=(n_per_class, dim))
        X.append(samples)
        y.extend([label] * n_per_class)
    return np.vstack(X), np.array(y)


def test_classifier_trains_and_predicts_well_on_separable_data():
    X, y = _make_toy_dataset()
    clf = CharacterClassifier(n_neighbors=3, use_pca=False)
    report = clf.fit(X, y)
    assert report["test_accuracy"] > 0.8

    preds, confidences = clf.predict(X[:5])
    assert len(preds) == 5
    assert all(0.0 <= c <= 1.0 for c in confidences)


def test_classifier_save_and_load_roundtrip(tmp_path):
    X, y = _make_toy_dataset()
    clf = CharacterClassifier(n_neighbors=3, use_pca=True, pca_components=5)
    clf.fit(X, y)

    model_path = tmp_path / "model.pkl"
    clf.save(str(model_path))

    loaded = CharacterClassifier.load(str(model_path))
    preds, _ = loaded.predict(X[:5])
    assert len(preds) == 5


def test_classifier_predict_before_training_raises():
    clf = CharacterClassifier()
    with pytest.raises(Exception):
        clf.predict(np.zeros((1, 10)))


@pytest.mark.skipif(
    not os.path.isfile(config.MODEL_PATH),
    reason="Trained model not present; run scripts/train_classifier.py first.",
)
def test_full_pipeline_on_demo_image():
    from src.pipeline import OCRPipeline

    demo_image = os.path.join(config.SAMPLE_IMAGES_DIR, "plate_demo.png")
    if not os.path.isfile(demo_image):
        pytest.skip("Demo image not generated; run scripts/generate_sample_plate.py first.")

    pipeline = OCRPipeline()
    result = pipeline.process_image(demo_image)

    assert result.num_characters_detected > 0
    assert isinstance(result.recognized_text, str)
    assert 0.0 <= result.confidence <= 1.0
