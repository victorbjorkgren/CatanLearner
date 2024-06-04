from time import time
import torch

def test_sparse_mm(n, p, f):
    dense_b = (torch.rand((n, n)) < p).float() * f
    sparse_a = dense_b.to_sparse()

    start = time()
    for _ in range(1000):
        a = torch.sparse.mm(sparse_a, dense_b)
    print("mix ", time() - start)

    start = time()
    for _ in range(1000):
        a = torch.sparse.mm(sparse_a, sparse_a)
    print("sparse ", time() - start)

    start = time()
    for _ in range(1000):
        a = dense_b @ dense_b
    print("dense ", time() - start)
