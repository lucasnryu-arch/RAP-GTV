"""Canonical three-view RAC-DMVC and the retained smoke fusion proxy."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

from rapgtv.clustering.kmeans import kmeans_fit
from rapgtv.representation.scaling import RobustScaler


REFERENCE_COMMIT = "441fafcae958a586f5d8d9727069be4104825035"


def _validate_views(views: tuple[np.ndarray, ...]) -> list[np.ndarray]:
    if len(views) != 3:
        raise ValueError("RAC-DMVC requires exactly deformation, terrain, and optical views")
    n = views[0].shape[0]
    arrays = [np.asarray(view, dtype=np.float64) for view in views]
    if any(view.ndim != 2 or view.shape[0] != n or not np.all(np.isfinite(view)) for view in arrays):
        raise ValueError("RAC-DMVC views must be finite aligned N by p arrays")
    return arrays


def _smoke(views: list[np.ndarray], k_star: int, params: dict, seed: int):
    scaled = [RobustScaler().fit(view).transform(view) for view in views]
    normalized = [view / max(float(np.sqrt(np.sum(view * view) / view.shape[0])), 1e-8) for view in scaled]
    result = kmeans_fit(np.concatenate(normalized, axis=1) / np.sqrt(3.0), k_star, seed=seed,
                        n_init=int(params.get("n_init", 4)))
    return result.labels, {"backend": "equal_view_direct_fusion_proxy+" + result.backend,
                           "converged": True, "canonical": False}


def _standardize(view: np.ndarray) -> np.ndarray:
    return (view - view.mean(axis=0)) / np.maximum(view.std(axis=0, ddof=0), 1e-8)


def _noisy_views(views: list[np.ndarray], ratio: float, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    result = [view.copy() for view in views]
    order = rng.permutation(views[0].shape[0])
    per_view = int(views[0].shape[0] * ratio) // len(views)
    for index, view in enumerate(result):
        rows = order[index * per_view:(index + 1) * per_view]
        view[rows] += rng.standard_normal((rows.size, view.shape[1]))
    return result


def _torch_modules(torch, nn, dimensions: list[int], latent: int, drop: float):
    class Encoder(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            sizes = [dim, *dimensions, latent]
            layers = []
            for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
                layers.append(nn.Linear(a, b, bias=False))
                layers.append(nn.BatchNorm1d(b, affine=i != len(sizes) - 2))
                if i != len(sizes) - 2:
                    layers.extend((nn.ReLU(), nn.Dropout(drop)))
            self.net = nn.Sequential(*layers)

        def forward(self, value):
            return self.net(value)

    class Decoder(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            sizes = [latent, *reversed(dimensions), dim]
            layers = []
            for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
                layers.append(nn.Linear(a, b, bias=i == len(sizes) - 2))
                if i != len(sizes) - 2:
                    layers.extend((nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(drop)))
            self.net = nn.Sequential(*layers)

        def forward(self, value):
            return self.net(value)

    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(latent, latent * 4), nn.ReLU(), nn.Linear(latent * 4, latent))

        def forward(self, value):
            return self.net(value)

    return Encoder, Decoder, Predictor


def _affinity(torch, functional, left, right, temperature: float, diagonal: bool):
    left, right = functional.normalize(left), functional.normalize(right)
    graph = torch.exp(-(2.0 - 2.0 * (left @ right.T)).clamp(min=0.0) / temperature)
    if diagonal and graph.shape[0] == graph.shape[1]:
        graph.fill_diagonal_(1.0)
    return graph / graph.sum(dim=1, keepdim=True).clamp_min(1e-7)


def _noisy_contrastive(torch, functional, query, key, positives, temperature: float):
    query, key = functional.normalize(query), functional.normalize(key)
    logits = (query @ key.T) / temperature
    negatives = 1.0 - positives
    positive_logit = (logits * positives).sum(dim=1, keepdim=True)
    denominator = (torch.exp(logits) * negatives).sum(dim=1, keepdim=True).clamp_min(1e-8)
    return (-positive_logit + torch.log(denominator)).mean()


def _canonical(views: list[np.ndarray], k_star: int, params: dict, seed: int, requested_device: str):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("canonical RAC-DMVC requires PyTorch") from exc

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(requested_device if requested_device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    clean = [_standardize(view).astype(np.float32) for view in views]
    noisy = _noisy_views(clean, float(params.get("noise_ratio", 0.5)), seed)
    clean_t = [torch.as_tensor(view, device=device) for view in clean]
    noisy_t = [torch.as_tensor(view, device=device) for view in noisy]
    hidden = [int(value) for value in params.get("hidden_dims", [1024, 1024, 1024])]
    latent = int(params.get("latent_dim", 128))
    drop = float(params.get("drop_rate", 0.2))
    Encoder, Decoder, Predictor = _torch_modules(torch, nn, hidden, latent, drop)
    online = nn.ModuleList([Encoder(view.shape[1]) for view in clean]).to(device)
    target = nn.ModuleList([Encoder(view.shape[1]) for view in clean]).to(device)
    target.load_state_dict(online.state_dict())
    for parameter in target.parameters():
        parameter.requires_grad = False
    decoders = nn.ModuleList([Decoder(view.shape[1]) for view in clean]).to(device)
    predictors = nn.ModuleList([Predictor() for _ in clean]).to(device)
    classifiers = nn.ParameterList([nn.Parameter(torch.randn(k_star, latent, device=device)) for _ in clean])
    parameters = list(online.parameters()) + list(decoders.parameters()) + list(predictors.parameters()) + list(classifiers)
    batch_size = min(int(params.get("batch_size", 1024)), clean[0].shape[0])
    if batch_size < 2:
        raise ValueError("RAC-DMVC needs at least two samples per training batch")
    base_lr = float(params.get("base_lr", 5e-4))
    lr = base_lr * batch_size / 256.0
    optimizer = torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.99), weight_decay=float(params.get("weight_decay", 0.0)))
    epochs = int(params.get("epochs", 100))
    warmup = int(params.get("warmup_epochs", 20))
    rectify = int(params.get("start_rectify_epoch", 20))
    momentum = float(params.get("momentum", 0.98))
    sigma = float(params.get("sigma", 0.07))
    con_temp = float(params.get("contrastive_temperature", 0.5))
    dist_temp = float(params.get("distill_temperature", 0.5))
    centers = None
    losses: list[float] = []

    for epoch in range(epochs):
        online.train(); decoders.train(); predictors.train()
        generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
        order = torch.randperm(clean_t[0].shape[0], generator=generator).cpu().numpy()
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            if ids.size < 2:
                continue
            xb, xn = [view[ids] for view in clean_t], [view[ids] for view in noisy_t]
            z = [encoder(view) for encoder, view in zip(online, xn)]
            p = [predictor(value) for predictor, value in zip(predictors, z)]
            with torch.no_grad():
                zt = [encoder(view) for encoder, view in zip(target, xn)]
            rec_terms = [functional.mse_loss(decoders[j](z[i]), xb[j])
                         for i in range(3) for j in range(3) if i != j]
            rec = torch.stack(rec_terms).mean()
            if epoch < rectify:
                identity = torch.eye(ids.size, device=device)
                intra_graphs = [identity] * 3
                cross_graphs = {(i, j): identity for i in range(3) for j in range(3) if i != j}
            else:
                with torch.no_grad():
                    intra_graphs = [_affinity(torch, functional, zt[i], zt[i], sigma, True) for i in range(3)]
                    cross_graphs = {(i, j): _affinity(torch, functional, p[i], zt[j], sigma, True)
                                    for i in range(3) for j in range(3) if i != j}
            intra = torch.stack([_noisy_contrastive(torch, functional, z[i], zt[i], intra_graphs[i], con_temp)
                                 for i in range(3)]).mean()
            inter = torch.stack([_noisy_contrastive(torch, functional, p[i], zt[j], cross_graphs[(i, j)], con_temp)
                                 for i in range(3) for j in range(3) if i != j]).mean()
            distill = torch.zeros((), device=device)
            if epoch >= rectify and centers is not None:
                fused = functional.normalize(torch.cat(zt, dim=1), dim=1)
                q = functional.softmax(fused @ functional.normalize(centers, dim=1).T / dist_temp, dim=1)
                dist_terms = []
                for index in range(3):
                    logits = functional.normalize(p[index], dim=1) @ functional.normalize(classifiers[index], dim=1).T
                    dist_terms.append(functional.kl_div(functional.log_softmax(logits / dist_temp, dim=1), q, reduction="batchmean"))
                distill = torch.stack(dist_terms).mean()
            total = rec + intra + inter + distill
            if not torch.isfinite(total):
                raise FloatingPointError("RAC-DMVC training produced NaN/Inf")
            optimizer.zero_grad(); total.backward(); optimizer.step()
            with torch.no_grad():
                for source, destination in zip(online.parameters(), target.parameters()):
                    destination.mul_(momentum).add_(source, alpha=1.0 - momentum)
            epoch_losses.append(float(total.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
        for group in optimizer.param_groups:
            group["lr"] = lr * min(1.0, (epoch + 1) / max(warmup, 1))
        target.eval()
        with torch.no_grad():
            features = [functional.normalize(encoder(view), dim=1).cpu().numpy() for encoder, view in zip(target, noisy_t)]
        fused_np = np.concatenate(features, axis=1)
        center_fit = kmeans_fit(fused_np, k_star, seed=42, n_init=10)
        centers_np = np.stack([fused_np[center_fit.labels == c].mean(axis=0) for c in range(k_star)])
        centers = torch.as_tensor(centers_np, dtype=torch.float32, device=device)

    target.eval()
    with torch.no_grad():
        features = [functional.normalize(encoder(view), dim=1).cpu().numpy() for encoder, view in zip(target, noisy_t)]
    fused = np.concatenate(features, axis=1)
    clustered = kmeans_fit(fused, k_star, seed=seed, n_init=int(params.get("kmeans_n_init", 10)))
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return clustered.labels, {
        "backend": "rac-dmvc-noisy-model-3view-symmetric+" + clustered.backend,
        "canonical": True, "converged": bool(np.all(np.isfinite(losses))),
        "reference_commit": REFERENCE_COMMIT, "reference_adaptation": "symmetric_all_ordered_pairs_3view",
        "epochs": epochs, "final_loss": losses[-1], "loss_history": losses,
        "device": str(device), "peak_gpu_memory_bytes": peak, "complete_views": True,
        "uses_rap_objects": False,
    }


def run_rac_dmvc(views: tuple[np.ndarray, np.ndarray, np.ndarray], k_star: int, params: dict, seed: int,
                 device: str = "cpu"):
    arrays = _validate_views(views)
    definition = params.get("definition")
    if definition == "smoke_equal_view_fusion_proxy":
        return _smoke(arrays, k_star, params, seed)
    if definition != "rac_dmvc_noisy_model_3view_symmetric":
        raise ValueError("unknown RAC-DMVC definition")
    return _canonical(arrays, k_star, params, seed, device)
