import numpy as np
import pandas as pd
import pytest

import pert2state_model
from pert2state_model import Perturb2StateModel


def test_package_has_version():
    assert pert2state_model.__version__ is not None


@pytest.fixture
def toy_data():
    """Create toy dataset that mimics the structure of real input DE statistics.

    Returns
    -------
    tuple
        (X_perturbs, y_target) where:
        - X_perturbs: DataFrame of shape (n_genes, n_perturbations) with perturbation effect signatures
        - y_target: Series of length n_genes with target state signature
    """
    np.random.seed(243)

    n_genes = 50
    n_perturbations = 100

    # Create gene names
    genes = [f"GENE{i}" for i in range(n_genes)]

    # Create perturbation names
    perturbations = [f"PERT{i}" for i in range(n_perturbations)]

    # Create perturbation matrix (genes x perturbations)
    # Values are z-scores of gene expression changes upon perturbation
    X_perturbs = pd.DataFrame(np.random.randn(n_genes, n_perturbations), index=genes, columns=perturbations)

    # Create target signature (z-scores for each gene in target state)
    # Add some correlation with specific perturbations for realistic behavior
    y_target = pd.Series(np.random.randn(n_genes), index=genes, name="target_state")

    return X_perturbs, y_target


def test_model_initialization():
    """Test that Perturb2StateModel can be initialized with default parameters."""
    model = Perturb2StateModel()
    assert model is not None
    assert model.pca_transform is False
    assert model.n_splits == 5
    assert model.n_repeats == 10  # Default is 10


def test_model_initialization_with_params():
    """Test that Perturb2StateModel can be initialized with custom parameters."""
    model = Perturb2StateModel(pca_transform=True, n_splits=3, n_repeats=2, alpha=[0.1, 1.0])
    assert model.pca_transform is True
    assert model.n_splits == 3
    assert model.n_repeats == 2
    assert model.alpha == [0.1, 1.0]


def test_model_fit(toy_data):
    """Test that the model can be fitted on toy data."""
    X_perturbs, y_target = toy_data

    model = Perturb2StateModel(pca_transform=False, n_splits=2, n_repeats=1)
    model.fit(X_perturbs, y_target, model_id="test")

    # Check that model has been fitted
    assert hasattr(model, "models")
    assert len(model.models) > 0
    assert hasattr(model, "eval")  # Attribute is "eval" not "eval_"
    assert len(model.eval) > 0


def test_model_fit_with_pca(toy_data):
    """Test that the model can be fitted with PCA transformation."""
    X_perturbs, y_target = toy_data

    model = Perturb2StateModel(pca_transform=True, n_splits=2, n_repeats=1, n_components=10)
    model.fit(X_perturbs, y_target, model_id="test_pca")

    # Check that PCA was applied
    assert hasattr(model, "pcas")
    assert len(model.pcas) > 0


def test_get_coefs(toy_data):
    """Test that get_coefs returns coefficient DataFrame with correct structure."""
    X_perturbs, y_target = toy_data

    model = Perturb2StateModel(pca_transform=False, n_splits=2, n_repeats=1)
    model.fit(X_perturbs, y_target, model_id="test")

    coefs = model.get_coefs()

    # Check structure
    assert isinstance(coefs, pd.DataFrame)
    assert "coef_mean" in coefs.columns
    assert "coef_sem" in coefs.columns
    assert len(coefs) == X_perturbs.shape[1]  # One coefficient per perturbation
    assert coefs.index.equals(X_perturbs.columns)


def test_get_prediction(toy_data):
    """Test that get_prediction returns predictions with correct structure."""
    X_perturbs, y_target = toy_data

    model = Perturb2StateModel(pca_transform=False, n_splits=2, n_repeats=1)
    model.fit(X_perturbs, y_target, model_id="test")

    predictions = model.get_prediction(X_perturbs)

    # Check structure
    assert isinstance(predictions, pd.DataFrame)
    assert "pred_mean" in predictions.columns
    assert "pred_sem" in predictions.columns
    assert len(predictions) == X_perturbs.shape[0]  # One prediction per gene
    assert predictions.index.equals(X_perturbs.index)


def test_get_prediction_return_splits(toy_data):
    """Test that get_prediction can return individual split predictions."""
    X_perturbs, y_target = toy_data

    model = Perturb2StateModel(pca_transform=False, n_splits=2, n_repeats=1)
    model.fit(X_perturbs, y_target, model_id="test")

    predictions = model.get_prediction(X_perturbs, return_splits=True)

    # Should return predictions for each split
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions.columns) == 2  # One column per split
    assert len(predictions) == X_perturbs.shape[0]  # One prediction per gene


def test_summarize_eval(toy_data):
    """Test that summarize_eval returns evaluation metrics DataFrame."""
    X_perturbs, y_target = toy_data

    model = Perturb2StateModel(pca_transform=False, n_splits=2, n_repeats=1)
    model.fit(X_perturbs, y_target, model_id="test")

    eval_summary = model.summarize_eval()

    # Check structure
    assert isinstance(eval_summary, pd.DataFrame)
    assert "test_r2" in eval_summary.columns
    assert "test_pearson" in eval_summary.columns
    assert "test_spearman" in eval_summary.columns
    assert len(eval_summary) == 1  # One row for the fitted model
