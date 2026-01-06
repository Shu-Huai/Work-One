import math
def cosine_with_warmup_lr(step: int, total_steps: int, warmup_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    warmup_steps = min(warmup_steps, total_steps)
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def cast_trainable_params_to_fp32(model):
    """
    避免 GradScaler 在 fp16 参数上 unscale 报错：把 requires_grad 的参数转 fp32
    """
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()