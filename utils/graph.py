import torch


def normalize_adjacency(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Symmetric normalization D^{-1/2} A D^{-1/2}.
    A: (J, J)
    """
    deg = A.sum(-1) + eps
    D_inv_sqrt = torch.pow(deg, -0.5)
    D_inv_sqrt = torch.diag(D_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


def build_hand_body_adjacency(num_joints: int, hand_edges, body_edges) -> torch.Tensor:
    """
    Build a simple undirected adjacency for spatial edges only.
    """
    A = torch.zeros((num_joints, num_joints), dtype=torch.float32)
    for i, j in hand_edges + body_edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    # self connections
    A = A + torch.eye(num_joints)
    A = normalize_adjacency(A)
    return A


