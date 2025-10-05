import numpy as np
from scipy.linalg import inv, qr


def get_matrix(n: int, m: int) -> np.ndarray:
    """Create random matrix n * m.

    Args:
        n (int): number of rows.
        m (int): number of columns.

    Returns:
        np.ndarray: matrix n*m.
    """
    return np.random.rand(n, m)


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matrix addition.

    Args:
        x (np.ndarray): 1st matrix.
        y (np.ndarray): 2nd matrix.

    Returns:
        np.ndarray: matrix sum.
    """
    return np.add(x, y)


def scalar_multiplication(x: np.ndarray, c: float) -> np.ndarray:
    """Matrix multiplication by scalar.

    Args:
        x (np.ndarray): matrix.
        c (float): scalar.

    Returns:
        np.ndarray: multiplied matrix.
    """
    return x * c


def dot_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matrices dot product.

    Args:
        x (np.ndarray): 1st matrix.
        y (np.ndarray): 2nd matrix or vector.

    Returns:
        np.ndarray: dot product.
    """
    return np.dot(x, y)


def identity_matrix(dim: int) -> np.ndarray:
    """Create identity matrix with dimension `dim`. 

    Args:
        dim (int): matrix dimension.

    Returns:
        np.ndarray: identity matrix.
    """
    return np.eye(dim)


def matrix_inverse(x: np.ndarray) -> np.ndarray:
    """Compute inverse matrix.

    Args:
        x (np.ndarray): matrix.

    Returns:
        np.ndarray: inverse matrix.
    """
    return inv(x)


def matrix_transpose(x: np.ndarray) -> np.ndarray:
    """Compute transpose matrix.

    Args:
        x (np.ndarray): matrix.

    Returns:
        np.ndarray: transosed matrix.
    """
    return x.T


def hadamard_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute hadamard product.

    Args:
        x (np.ndarray): 1th matrix.
        y (np.ndarray): 2nd matrix.

    Returns:
        np.ndarray: hadamard produc
    """
    return np.multiply(x, y)


def basis(x: np.ndarray) -> tuple[int]:
    """Compute matrix basis.

    Args:
        x (np.ndarray): matrix.

    Returns:
        tuple[int]: indexes of basis columns.
    """
    _, _, inds = qr(x, mode='economic', pivoting=True)
    rank = np.linalg.matrix_rank(x)
    return tuple(inds[:rank])


def norm(x: np.ndarray, order: int | float | str) -> float:
    """Matrix norm: Frobenius, Spectral or Max.

    Args:
        x (np.ndarray): vector
        order (int | float): norm's order: 'fro', 2 or inf.

    Returns:
        float: vector norm
    """
    return np.linalg.norm(x, ord=order)

if __name__ == "__main__":
    a = get_matrix(3, 3)
    b = get_matrix(3, 3)
    print("Matrix a:\n", a)
    print("Matrix b:\n", b)
    print("a + b:\n", add(a, b))
    print("2 * a:\n", scalar_multiplication(a, 2))
    print("a . b:\n", dot_product(a, b))
    print("Identity matrix 3x3:\n", identity_matrix(3))
    print("Inverse of a:\n", matrix_inverse(a))
    print("Transpose of a:\n", matrix_transpose(a))
    print("Hadamard product of a and b:\n", hadamard_product(a, b))
    print("Basis of a:", basis(a))
    print("Frobenius norm of a:", norm(a, 'fro'))
