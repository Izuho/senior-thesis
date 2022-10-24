import torch

def amplitude_embedding(feat, n_qubits):
    # feat : torch.tensor of 2^n_qubits elements
    if feat.ndim == 1:
        feat = feat.reshape(-1,).to(torch.complex64)
        feat /= torch.norm(feat)
    elif feat.ndim == 2:
        feat = feat.reshape(-1, 2**n_qubits,).to(torch.complex64)
        feat = feat.transpose(0,1) / torch.norm(feat, dim=1)
        feat = feat.transpose(0,1)
    return feat