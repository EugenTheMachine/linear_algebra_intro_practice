import numpy as np

def negative_matrix(x: np.ndarray) -> np.ndarray:
    """
    Returns the negation of each element in the input vector or matrix.

    Args:
        x (np.ndarray): A vector (n*1) or matrix (n*n).

    Returns:
        np.ndarray: A matrix with each element negated.
    """
    return -x


def reverse_matrix(x: np.ndarray) -> np.ndarray:
    """
    Returns the input vector or matrix with the order of elements reversed.

    Args:
        x (np.ndarray): A vector (n*1) or matrix (n*n).

    Returns:
        np.ndarray: A matrix with the order of elements reversed.
    """
    return np.flip(x)


def affine_transform(
    x: np.ndarray, alpha_deg: float, scale: tuple[float, float], shear: tuple[float, float],
    translate: tuple[float, float],
) -> np.ndarray:
    """
    Compute affine transformation.

    Yevhen's NOTE: here we assume 2D points for simplicity,
    given that deg, shear, scale, and translate are defined for 2D space.

    Therefore, we treat the input as a collection of 2D points.
    E.g., vector [1, 2, 3, 4] represents points (1,2) and (3,4).

    If the input is a matrix, each row is treated as a 2D point.
    E.g., matrix [[1,2],[3,4]] represents points (1,2) and (3,4).

    All other cases (e.g., vector of uneven length, or matrices
    of other shapes than 2x2) will raise a ValueError.

    Hence, the size `n` below can be constrained as follows:
    - For vectors: n must be even.
    - For matrices: n must be 2 (i.e., 2x2 matrix).

    Args:
        x (np.ndarray): vector n*1 or matrix n*n.
        alpha_deg (float): rotation angle in deg.
        scale (tuple[float, float]): x, y scale factor.
        shear (tuple[float, float]): x, y shear factor.
        translate (tuple[float, float]): x, y translation factor.

    Returns:
        np.ndarray: transformed matrix.
    """
    alpha = np.deg2rad(alpha_deg)
    sx, sy = scale
    shx, shy = shear
    tx, ty = translate
    S = np.array([[sx, 0], [0, sy]])
    Sh = np.array([[1, shx], [shy, 1]])
    R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    A = R @ Sh @ S
    if x.ndim == 1:
        if x.size % 2 != 0:
            raise ValueError("Vector length must be even to interpret as 2D points.")
        x_reshaped = x.reshape(-1, 2)  # Assuming each pair represents a 2D point
    elif x.ndim == 2:
        if x.shape[0] != x.shape[1]:
            raise ValueError("Input matrix must be square (n*n).")
        if x.shape[0] != 2:
            raise ValueError("Only 2x2 matrices are supported for 2D transformations.")
        x_reshaped = x  # Treat rows as 2D points
    else:
        raise ValueError("Input must be a 1D vector (n*1) or 2D square matrix (n*n).")
    transformed = (A @ x_reshaped.T).T + np.array([tx, ty])
    if x.ndim == 1:
        return transformed.flatten()
    return transformed
