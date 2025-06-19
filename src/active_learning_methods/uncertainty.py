import torch
import torch.nn.functional as F

def calculate_uncertainty(outputs, select_uncertainty, number_of_classes):
    """
    Calculates uncertainty scores for model predictions using a specified strategy.
    Parameters
    ----------
    outputs :  The raw model outputs (logits) with shape (N, C), where N is the number of samples,
        and C is the number of classes.

    select_uncertainty : 
        The uncertainty sampling strategy to use. Options include:
        - 'none': No uncertainty, returns zero scores.
        - 'Least_confidence_sampling': Based on 1 - max probability.
        - 'Margin_sampling': Based on the difference between the top two class probabilities.
        - 'Entropy_sampling': Based on the entropy of the probability distribution.

    number_of_classes : int
        The number of classes in the classification task. Used for normalization.

    Returns
    -------
    uncertainty_score : np.ndarray
        Array of uncertainty scores (shape: [N]) for each sample.
    """
    if select_uncertainty == 'none':
        return torch.zeros_like(outputs[:, 0]).cpu().numpy()
    
    probs = F.softmax(outputs, dim=1)
    if select_uncertainty == 'Least_confidence_sampling':
        most_confident = torch.max(probs, dim=1)[0].cpu().numpy()
        score = (1 - most_confident) * (number_of_classes / (number_of_classes - 1))
        uncertainty_score = score
    elif select_uncertainty == 'Margin_sampling':
        sorted_probs, _ = torch.sort(probs, descending=True)
        difference = (sorted_probs[:,0] - sorted_probs[:,1]).cpu().numpy()
        margin_conf = 1 - difference
        uncertainty_score = margin_conf
    elif select_uncertainty == 'Entropy_sampling':
        entropy = -(probs * torch.log2(probs + 1e-10)).sum(dim=1)
        normalized_entropy = entropy / torch.log2(torch.tensor(number_of_classes).float())
        uncertainty_score = normalized_entropy.cpu().numpy()
        
    else:
        raise ValueError(f"Unknown uncertainty selection method: {select_uncertainty}")

    return uncertainty_score


