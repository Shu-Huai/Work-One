import tqdm
import torch
from typing import Dict
@torch.no_grad()
def retrieval_metrics_multi(
    text_embs_cpu: torch.Tensor,    # [N,D]
    imgset_embs_cpu: torch.Tensor,  # [N,D]
    logit_scale: torch.Tensor,
    device: torch.device,
    text_chunk: int,
    cand_chunk: int,
) -> Dict[str, float]:
    """
    text->imgset 检索，GT 为同 index
    """
    N, _ = text_embs_cpu.shape
    ranks = torch.empty(N, dtype=torch.long)
    scale = logit_scale.exp().detach().float().to(device)

    img_all = imgset_embs_cpu.to(device)  # [N,D]

    for t0 in tqdm(range(0, N, text_chunk), desc="Scoring (texts)"):
        t1 = min(t0 + text_chunk, N)
        t = text_embs_cpu[t0:t1].to(device)  # [C,D]
        C = t.shape[0]

        scores = torch.empty((C, N), dtype=torch.float32)  # CPU

        for c0 in range(0, N, cand_chunk):
            c1 = min(c0 + cand_chunk, N)
            cand = img_all[c0:c1]  # [M,D]
            sim = (t @ cand.t()) * scale
            scores[:, c0:c1] = sim.detach().cpu()

        gt_idx = torch.arange(t0, t1)
        row = torch.arange(0, C)
        gt_score = scores[row, gt_idx]
        rank = (scores > gt_score.unsqueeze(1)).sum(dim=1) + 1
        ranks[t0:t1] = rank

    return {
        "N": float(N),
        "R@1": float((ranks <= 1).float().mean().item()),
        "R@5": float((ranks <= 5).float().mean().item()),
        "R@10": float((ranks <= 10).float().mean().item()),
        "MedianRank": float(ranks.median().item()),
    }
