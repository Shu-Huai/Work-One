import torch
def extend_clip_context_length(model, new_len: int):
    """
    77 -> 256：插值 positional_embedding + 重建 attn_mask，并同步到每个 block
    """
    import torch.nn.functional as F

    if new_len == model.context_length:
        return
    old_len = model.context_length
    if new_len < old_len:
        raise ValueError(f"new_len({new_len}) must be >= old_len({old_len})")

    device = model.positional_embedding.device
    dtype = model.positional_embedding.dtype

    with torch.no_grad():
        old_pe = model.positional_embedding.detach()  # [old_len, width]
        pe = old_pe.T.unsqueeze(0)                    # [1, width, old_len]
        pe_new = F.interpolate(pe, size=new_len, mode="linear", align_corners=False)
        pe_new = pe_new.squeeze(0).T.contiguous()     # [new_len, width]

    model.context_length = new_len
    model.positional_embedding = torch.nn.Parameter(pe_new.to(device=device, dtype=dtype))

    attn_mask = model.build_attention_mask().to(device=device)
    model.attn_mask = attn_mask
    model._buffers["attn_mask"] = attn_mask
    for blk in model.transformer.resblocks:
        blk.attn_mask = attn_mask


def clip_encode_text_safe(clip_model, tokens: torch.Tensor) -> torch.Tensor:
    """
    等价于 clip_model.encode_text，但在 x @ text_projection 前对齐 dtype
    """
    x = clip_model.token_embedding(tokens).type(clip_model.dtype)
    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)
    x = clip_model.ln_final(x).type(clip_model.dtype)

    eot = tokens.argmax(dim=-1)
    x = x[torch.arange(x.shape[0], device=x.device), eot]  # [B, width]

    proj = clip_model.text_projection
    if proj is not None and proj.dtype != x.dtype:
        proj = proj.to(x.dtype)
    x = x @ proj
    return x

def clip_encode_image_safe(clip_model, images: torch.Tensor) -> torch.Tensor:
    """
    ViT 视觉塔：在 x @ visual.proj 前对齐 dtype
    """
    visual = clip_model.visual
    if visual.__class__.__name__ == "VisionTransformer":
        x = images.type(visual.conv1.weight.dtype)
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        cls = visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype)
        x = torch.cat([cls, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)

        x = visual.ln_post(x[:, 0, :])

        proj = visual.proj
        if proj is not None and proj.dtype != x.dtype:
            proj = proj.to(x.dtype)
        if proj is not None:
            x = x @ proj
        return x

    return clip_model.encode_image(images)


def freeze_to_projection_only(model):
    """
    贴论文：只 finetune projection layers（+ logit_scale）
    """
    for p in model.parameters():
        p.requires_grad = False

    if getattr(model, "text_projection", None) is not None:
        model.text_projection.requires_grad = True
    if getattr(model, "logit_scale", None) is not None:
        model.logit_scale.requires_grad = True
    if hasattr(model, "visual") and getattr(model.visual, "proj", None) is not None:
        model.visual.proj.requires_grad = True