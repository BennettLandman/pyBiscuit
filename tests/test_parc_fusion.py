"""Unit tests for parcellation multi-model fusion (B7.3)."""

import numpy as np

from pybiscuit.parc.fusion import fuse_folds, refine_fold


def test_fuse_folds_single_model():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)

    # 3 classes, 4 vertices
    # Vertex 0: class 0 high score
    # Vertex 1: class 1 high score
    # Vertex 2: class 0 high score
    # Vertex 3: class 2 high score
    scores = np.array(
        [
            [10.0, -10.0, 10.0, -10.0],
            [-10.0, 10.0, -10.0, -10.0],
            [-10.0, -10.0, -10.0, 10.0],
        ],
        dtype=np.float32,
    )

    pred, prob = refine_fold(vertices, faces, scores)
    assert pred.shape == (4,)
    assert prob.shape == (4,)
    assert np.all(prob > 0.99)

    labels = np.array([100, 200, 300], dtype=np.int64)
    final = fuse_folds(vertices, faces, [scores], labels=labels)
    assert np.array_equal(final, [100, 200, 100, 300])


def test_fuse_folds_multi_model():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    # 3 classes, 3 vertices, 3 models
    # Models 1 & 2 agree on class 1 for vertex 0 with high confidence; model 3 predicts class 2
    s1 = np.array([[0.0, 10.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 10.0]], dtype=np.float32)
    s2 = np.array([[0.0, 10.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 10.0]], dtype=np.float32)
    s3 = np.array([[0.0, 0.0, 10.0], [0.0, 10.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)

    labels = np.array([10, 20, 30], dtype=np.int64)
    final = fuse_folds(vertices, faces, [s1, s2, s3], labels=labels)
    # Vertex 0: class 1 (ID 20) receives 2 strong votes vs 1 for class 2
    assert final[0] == 20
