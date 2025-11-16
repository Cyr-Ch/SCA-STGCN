import torch
import torch.nn as nn
import torch.nn.functional as F


def classification_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Focal Loss for addressing class imbalance.
    
    Focal loss focuses learning on hard examples by down-weighting easy examples.
    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        logits: (B, C) - raw logits from model
        targets: (B,) - integer class labels
        alpha: Balancing factor (default: 0.25). Can be a float or tensor of shape (C,)
        gamma: Focusing parameter (default: 2.0). Higher gamma = more focus on hard examples
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        loss: Scalar tensor (if reduction='mean' or 'sum') or (B,) tensor (if reduction='none')
    
    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    # Compute cross entropy loss (without reduction)
    ce_loss = F.cross_entropy(logits, targets, reduction='none')  # (B,)
    
    # Get probabilities for the true class
    probs = F.softmax(logits, dim=-1)  # (B, C)
    p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,) - probability of true class
    
    # Compute focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma  # (B,)
    
    # Apply alpha weighting
    if isinstance(alpha, (float, int)):
        # Uniform alpha for all classes
        alpha_t = alpha
    else:
        # Per-class alpha (tensor of shape (C,))
        alpha_t = alpha[targets]  # (B,)
    
    # Focal loss: -alpha * (1 - p_t)^gamma * log(p_t)
    # Note: ce_loss = -log(p_t), so we can write:
    focal_loss = alpha_t * focal_weight * ce_loss  # (B,)
    
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    elif reduction == 'none':
        return focal_loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean', 'sum', or 'none'.")


def signer_adversarial_loss(signer_logits: torch.Tensor, signer_ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(signer_logits, signer_ids)


def info_nce(
    z: torch.Tensor,
    pos_idx: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Simple InfoNCE between pairs (z_i, z_pos_i), all others as negatives.
    z: (B, D)
    pos_idx: (B,) index of positive partner for each i
    """
    z = F.normalize(z, dim=-1)
    sim = z @ z.t()  # (B,B)
    logits = sim / temperature
    loss = F.cross_entropy(logits, pos_idx)
    return loss


def attention_regularization(
    H_att: torch.Tensor, sparsity_weight: float = 0.0, smooth_weight: float = 0.0
) -> torch.Tensor:
    """
    Example reg on attended features:
    - sparsity: L1 on features to encourage focused channels
    - smoothness: temporal TV-loss
    H_att: (B, T, D)
    """
    loss = torch.tensor(0.0, device=H_att.device)
    if sparsity_weight > 0:
        loss = loss + sparsity_weight * H_att.abs().mean()
    if smooth_weight > 0:
        tv = (H_att[:, 1:] - H_att[:, :-1]).abs().mean()
        loss = loss + smooth_weight * tv
    return loss


class CLUBEstimator(nn.Module):
    """
    Contrastive Log-ratio Upper Bound (CLUB) for Mutual Information estimation.
    
    CLUB provides an upper bound on MI: I(X; Y) <= E[log p(y|x)] - E[log p(y|x')]
    where x' is a negative sample.
    
    This is more principled than adversarial training for minimizing MI.
    
    Reference: "Contrastive Log-ratio Upper Bound of Mutual Information" (Cheng et al., 2020)
    """
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 128):
        """
        Args:
            x_dim: Dimension of X (e.g., feature embedding)
            y_dim: Dimension of Y (e.g., signer embedding or one-hot signer ID)
            hidden_dim: Hidden dimension for the MI estimator network
        """
        super().__init__()
        # Network to estimate p(y|x)
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)  # Output: log p(y|x)
        )
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Estimate MI upper bound using CLUB.
        
        Args:
            x: (B, x_dim) - feature embeddings
            y: (B, y_dim) - signer embeddings or one-hot signer IDs
        
        Returns:
            mi_upper_bound: Scalar tensor, upper bound on I(X; Y)
        """
        B = x.size(0)
        
        # Positive pairs: (x_i, y_i)
        xy_pos = torch.cat([x, y], dim=-1)  # (B, x_dim + y_dim)
        log_p_pos = self.net(xy_pos)  # (B, 1)
        
        # Negative pairs: (x_i, y_j) where j != i
        # Shuffle y to create negative samples
        y_neg = y[torch.randperm(B, device=y.device)]  # (B, y_dim)
        xy_neg = torch.cat([x, y_neg], dim=-1)  # (B, x_dim + y_dim)
        log_p_neg = self.net(xy_neg)  # (B, 1)
        
        # CLUB upper bound: E[log p(y|x)] - E[log p(y'|x)]
        # where y' is a negative sample
        mi_upper_bound = (log_p_pos - log_p_neg).mean()
        
        return mi_upper_bound
    
    def mi_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute MI loss for minimization (returns positive value to minimize).
        
        Args:
            x: (B, x_dim) - feature embeddings
            y: (B, y_dim) - signer embeddings or one-hot signer IDs
        
        Returns:
            mi_loss: Scalar tensor, MI upper bound (to minimize)
        """
        return self.forward(x, y)


def mutual_information_loss(
    features: torch.Tensor,
    signer_embeddings: torch.Tensor,
    mi_estimator: CLUBEstimator
) -> torch.Tensor:
    """
    Compute mutual information loss between features and signer embeddings.
    
    This loss encourages signer-invariant features by minimizing I(Z; S),
    where Z are features and S are signer embeddings.
    
    Args:
        features: (B, D) - feature embeddings (e.g., from model)
        signer_embeddings: (B, d) - signer embeddings
        mi_estimator: CLUBEstimator instance
    
    Returns:
        mi_loss: Scalar tensor, MI upper bound to minimize
    """
    return mi_estimator.mi_loss(features, signer_embeddings)


def mutual_information_loss_onehot(
    features: torch.Tensor,
    signer_ids: torch.Tensor,
    num_signers: int,
    mi_estimator: CLUBEstimator
) -> torch.Tensor:
    """
    Compute mutual information loss between features and signer IDs (one-hot).
    
    Args:
        features: (B, D) - feature embeddings
        signer_ids: (B,) - signer IDs (integer labels)
        num_signers: Number of signers
        mi_estimator: CLUBEstimator instance (should have y_dim = num_signers)
    
    Returns:
        mi_loss: Scalar tensor, MI upper bound to minimize
    """
    # Convert signer IDs to one-hot
    signer_onehot = F.one_hot(signer_ids, num_classes=num_signers).float()  # (B, num_signers)
    return mi_estimator.mi_loss(features, signer_onehot)


