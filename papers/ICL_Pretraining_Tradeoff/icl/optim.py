import jax.tree_util as tu
import optax


def get_optimizer_and_lr_schedule(
        optimizer: str, schedule: str, clip_max_norm: float = 1.0, **kwargs
) -> tuple[optax.GradientTransformation, optax.Schedule]:
    # Lr Schedule
    match schedule:
        case "warmup_cosine_decay":
            lr = optax.warmup_cosine_decay_schedule(0.0, kwargs["lr"], kwargs["warmup_steps"], kwargs["total_steps"])
        case "triangle":
            lr = optax.join_schedules(
                [
                    optax.linear_schedule(0.0, kwargs["lr"], kwargs["warmup_steps"]),
                    optax.linear_schedule(kwargs["lr"], 0.0, kwargs["total_steps"] - kwargs["warmup_steps"]),
                ],
                [kwargs["warmup_steps"]],
            )
        case "constant":
            lr = optax.constant_schedule(kwargs["lr"])
        case _:
            raise NotImplementedError(f"Unknown learning rate schedule: {schedule}")
    # Optimizer
    if optimizer == "adam":
        tx = optax.adam(lr)
    # Weight decay mask based on nanoGPT (https://github.com/karpathy/nanoGPT)
    elif optimizer == "adamw":
        tx = optax.adamw(
            lr,
            weight_decay=kwargs["weight_decay"],
            mask=tu.tree_map_with_path(lambda kp, _: kp[0].key == "_h" and kp[-1].key == "kernel", kwargs["params"]),
        )
    elif optimizer == "adamw_attn":
        def wd_mask_attn_only(kp, _):
            names = [k.key for k in kp]
            is_kernel = names[-1] == "kernel"
            in_attn = "attn" in names
            is_attn_weight = any(n in {"c_attn", "c_proj"} for n in names)
            return is_kernel and in_attn and is_attn_weight

        mask = tu.tree_map_with_path(wd_mask_attn_only, kwargs["params"])
        tx = optax.adamw(lr, weight_decay=kwargs["weight_decay"], mask=mask)
    else:
        raise NotImplementedError
    
    # Add gradient clipping
    print(f"Using optimizer: {optimizer}, schedule: {schedule}, clip_max_norm: {clip_max_norm}")
    tx = optax.chain(
        optax.clip_by_global_norm(clip_max_norm),
        tx
    )
    
    return tx, lr
