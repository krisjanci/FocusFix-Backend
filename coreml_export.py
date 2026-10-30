"""Export fitted sklearn regressors using public Core ML model specifications.

Avoid sklearn's version-dependent converter. Linear scaling is folded into the
weights; trees retain their exact learned splits and weighted leaf values.
"""
from pathlib import Path
import json
import platform

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor


def export_model(model, features, target, destination, metadata, verification_rows):
    import coremltools as ct
    from coremltools.models import datatypes
    from coremltools.models.tree_ensemble import TreeEnsembleRegressor
    from coremltools.proto import Model_pb2

    inputs = [("features", datatypes.Array(len(features)))]
    spec = Model_pb2.Model()
    spec.specificationVersion = 1
    feature = spec.description.input.add()
    feature.name = "features"
    feature.type.multiArrayType.shape.append(len(features))
    feature.type.multiArrayType.dataType = 65600  # DOUBLE
    output = spec.description.output.add()
    output.name = "raw_rating"
    output.type.doubleType.SetInParent()
    spec.description.predictedFeatureName = "raw_rating"
    regressor = model.steps[-1][1] if isinstance(model, Pipeline) else model
    if isinstance(regressor, (LinearRegression, Ridge, DummyRegressor)):
        if isinstance(regressor, DummyRegressor):
            weights, bias = np.zeros(len(features)), float(regressor.constant_.ravel()[0])
        else:
            weights = np.asarray(regressor.coef_, dtype=float)
            bias = float(regressor.intercept_)
            if isinstance(model, Pipeline):
                scaler = model.steps[0][1]
                weights = weights / scaler.scale_
                bias -= float(np.dot(weights, scaler.mean_))
        spec.glmRegressor.weights.add().value.extend(weights.tolist())
        spec.glmRegressor.offset.append(bias)
    else:
        builder = TreeEnsembleRegressor(inputs, "raw_rating")
        if isinstance(regressor, RandomForestRegressor):
            trees = [(t, 1 / len(regressor.estimators_)) for t in regressor.estimators_]
            base = 0.0
        elif isinstance(regressor, GradientBoostingRegressor):
            trees = [(t, regressor.learning_rate) for t in regressor.estimators_.ravel()]
            base = float(regressor.init_.constant_.ravel()[0])
        elif isinstance(regressor, DecisionTreeRegressor):
            trees, base = [(regressor, 1.0)], 0.0
        else:
            raise ValueError(f"Unsupported Core ML export: {type(regressor).__name__}")
        builder.set_default_prediction_value(base)
        for tree_id, (estimator, weight) in enumerate(trees):
            tree = estimator.tree_
            for node in range(tree.node_count):
                if tree.children_left[node] == -1:
                    builder.add_leaf_node(tree_id, node, float(tree.value[node].ravel()[0]) * weight)
                else:
                    builder.add_branch_node(tree_id, node, int(tree.feature[node]),
                                            float(tree.threshold[node]), "BranchOnValueLessThanEqual",
                                            int(tree.children_left[node]), int(tree.children_right[node]))
        spec = builder.spec
    spec.description.metadata.shortDescription = f"FocusFix experimental self-reported {target}; not a medical measurement"
    spec.description.metadata.userDefined.update({
        "feature_order": json.dumps(features), "target": target,
        "input_contract": "Round inputs to float32 then store in Double MLMultiArray; clip raw_rating to [0,10] in app",
        **{key: str(value) for key, value in metadata.items()},
    })
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    exported = ct.models.MLModel(spec, skip_model_load=platform.system() != "Darwin")
    exported.save(str(destination))
    verification = {"runtime_verified": False, "max_absolute_error": None}
    if platform.system() == "Darwin":
        # Reload the saved file, testing the artifact the app will actually get.
        reloaded = ct.models.MLModel(str(destination), compute_units=ct.ComputeUnit.CPU_ONLY)
        values = np.asarray(verification_rows, dtype=np.float32)
        expected = model.predict(values)
        actual = np.array([reloaded.predict({"features": row.astype(np.float64)})["raw_rating"] for row in values])
        error = float(np.max(np.abs(expected - actual)))
        if not np.allclose(expected, actual, atol=1e-4, rtol=1e-5):
            destination.unlink(missing_ok=True)
            raise ValueError(f"Core ML prediction mismatch: {error}")
        verification = {"runtime_verified": True, "max_absolute_error": error}
    return verification
