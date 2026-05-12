import json
import os
import warnings

import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from .data import build_dataloaders
from .losses import (
    estimate_inplane_irreps_huber_delta_from_trainset,
    estimate_model_component_huber_delta_from_trainset,
    evaluate_masked_tensor_metrics,
    inplane_irreps_huber_frobenius_loss,
    tensor_basis_huber_frobenius_loss,
)
from .metrics import (
    compute_diag_score,
    compute_model_basis_diag_mae,
    compute_original_tensor_metrics,
    estimate_model_basis_diag_normalizers,
)
from .model import build_network
from .target_transform import model_tensor_to_raw_tensor


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    import numpy as np

    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config):
    requested = config.get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, metrics, config):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def save_training_config(config):
    with open(config["resolved_config_output_path"], "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    original_config_path = config.get("config_path")
    if original_config_path and os.path.exists(original_config_path):
        with open(original_config_path, "r", encoding="utf-8") as src:
            original_config = json.load(src)
        with open(config["original_config_output_path"], "w", encoding="utf-8") as dst:
            json.dump(original_config, dst, ensure_ascii=False, indent=2)


def _loss_uses_irreps(loss_type):
    return loss_type in {"inplane_irreps_huber", "inplane_irreps_huber_frobenius"}


def _loss_component_weights(checkpoint_config):
    if _loss_uses_irreps(checkpoint_config["loss_type"]):
        return checkpoint_config.get("loss_irrep_weights") or checkpoint_config["loss_component_weights"]
    return checkpoint_config["loss_component_weights"]


def compute_training_loss(output, batch, checkpoint_config):
    kwargs = {
        "component_weights": _loss_component_weights(checkpoint_config),
        "delta": checkpoint_config["huber_delta"],
        "lambda_tensor": checkpoint_config["lambda_tensor"],
    }
    loss_type = checkpoint_config["loss_type"]
    if loss_type == "tensor_basis_huber_frobenius":
        return tensor_basis_huber_frobenius_loss(output, batch.energy, batch.energy_mask, **kwargs)
    if _loss_uses_irreps(loss_type):
        return inplane_irreps_huber_frobenius_loss(output, batch.energy, batch.energy_mask, **kwargs)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def build_checkpoint_config(config):
    return {
        "structure_path": config["structure_path"],
        "label_path": config["label_path"],
        "target_key": config["target_key"],
        "global_descriptor_path": config.get("global_descriptor_path"),
        "global_descriptor_columns": config.get("global_descriptor_columns"),
        "use_global_descriptor_in_attention": config.get("use_global_descriptor_in_attention", False),
        "use_global_descriptor_gating": config.get("use_global_descriptor_gating", True),
        "formula": config["formula"],
        "supervision": "in_plane_xx_yy_xy",
        "loss_type": config["loss_type"],
        "loss_component_weights": config["loss_component_weights"],
        "loss_irrep_weights": config.get("loss_irrep_weights"),
        "lambda_tensor": config["lambda_tensor"],
        "huber_delta": config.get("huber_delta"),
        "best_model_metric": config["best_model_metric"],
        "diag_score_weights": config["diag_score_weights"],
        "diag_score_normalization": config["diag_score_normalization"],
        "batch_size": config["batch_size"],
        "train_ratio": config["train_ratio"],
        "valid_ratio": config["valid_ratio"],
        "test_ratio": config["test_ratio"],
        "epochs": config["epochs"],
        "early_stopping_patience": config["early_stopping_patience"],
        "lr_scheduler": "ReduceLROnPlateau",
        "lr_scheduler_mode": config["lr_scheduler_mode"],
        "lr_scheduler_factor": config["lr_scheduler_factor"],
        "lr_scheduler_patience": config["lr_scheduler_patience"],
        "lr_scheduler_min_lr": config["lr_scheduler_min_lr"],
        "lr": config["lr"],
        "weight_decay": config.get("weight_decay", 0.01),
        "gradient_clip_max_norm": config.get("gradient_clip_max_norm", 1.0),
        "uvu_dropout_p": config.get("uvu_dropout_p", 0.2),
        "max_radius": config["max_radius"],
        "radial_cutoff": config["radial_cutoff"],
        "use_attention": config.get("use_attention", True),
        "heads": config["heads"],
        "lmax": config["lmax"],
        "seed": config["seed"],
    }


def run_training(config):
    default_dtype = torch.float32
    torch.set_default_dtype(default_dtype)
    warnings.filterwarnings("ignore")

    device = get_device(config)
    set_seed(config["seed"])
    os.makedirs(config["save_dir"], exist_ok=True)
    save_training_config(config)

    (
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        feature_dim,
        num_nodes,
        target_transform,
        global_descriptor_dim,
    ) = build_dataloaders(
        config, default_dtype
    )

    net = build_network(feature_dim, num_nodes, config, global_descriptor_dim=global_descriptor_dim).to(device)
    optim = torch.optim.AdamW(
        net.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.01),
    )
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode=config["lr_scheduler_mode"],
        factor=config["lr_scheduler_factor"],
        patience=config["lr_scheduler_patience"],
        min_lr=config["lr_scheduler_min_lr"],
    )

    checkpoint_config = build_checkpoint_config(config)
    if target_transform is not None:
        checkpoint_config["target_transform"] = target_transform.state_dict()
    checkpoint_config["global_descriptor_dim"] = global_descriptor_dim
    if checkpoint_config["huber_delta"] is None:
        if _loss_uses_irreps(checkpoint_config["loss_type"]):
            checkpoint_config["huber_delta"] = estimate_inplane_irreps_huber_delta_from_trainset(
                train_dataloader.dataset
            )
        else:
            checkpoint_config["huber_delta"] = estimate_model_component_huber_delta_from_trainset(
                train_dataloader.dataset
            )
    checkpoint_config["diag_score_normalizers"] = estimate_model_basis_diag_normalizers(train_dataloader.dataset)

    print(f"Loss type: {checkpoint_config['loss_type']}")
    print(f"Loss component weights: {_loss_component_weights(checkpoint_config)}")
    print(f"Tensor Frobenius lambda: {checkpoint_config['lambda_tensor']}")
    print(f"Huber delta: {checkpoint_config['huber_delta']}")
    print(f"In-plane score normalizers: {checkpoint_config['diag_score_normalizers']}")

    sample_target_shape = tuple(train_dataloader.dataset[0].energy.shape)
    print(f"Device: {device}")
    print(
        f"Train samples: {len(train_dataloader.dataset)} | "
        f"Valid samples: {len(valid_dataloader.dataset)} | "
        f"Test samples: {len(test_dataloader.dataset)}"
    )
    print(f"Tensor label shape per sample: {sample_target_shape}")
    print(f"Global descriptor dim: {global_descriptor_dim}")
    print(f"Trainable parameters: {count_parameters(net):,}")
    print(f"Best checkpoint will be saved to: {config['best_model_path']}")
    print(f"Best model selection metric: {checkpoint_config['best_model_metric']}")
    print(f"Early stopping patience: {config['early_stopping_patience']}")

    best_model_score = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(config["epochs"]):
        train_loss_sum = 0.0
        train_mae_sum = 0.0
        train_rmse_sum = 0.0

        valid_loss_sum = 0.0
        valid_mae_sum = 0.0
        valid_rmse_sum = 0.0
        valid_max_abs_sum = 0.0
        valid_mae_raw_sum = 0.0
        valid_rmse_raw_sum = 0.0
        valid_max_abs_raw_sum = 0.0
        valid_pred_raw_chunks = []
        valid_target_raw_chunks = []
        valid_mask_raw_chunks = []

        net.train()
        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config['epochs']} [train]")
        for batch in train_bar:
            batch = batch.to(device)
            optim.zero_grad()

            with autocast():
                output = net(batch)
                loss = compute_training_loss(output, batch, checkpoint_config)

            if torch.isnan(output).any():
                print("output is nan")

            if torch.isnan(loss):
                print("NaN detected in loss before backward")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            grad_clip_max_norm = config.get("gradient_clip_max_norm", 1.0)
            if grad_clip_max_norm is not None and float(grad_clip_max_norm) > 0:
                grad_norm = clip_grad_norm_(net.parameters(), max_norm=float(grad_clip_max_norm))
            else:
                grad_norm = torch.tensor(0.0, device=device)
            scaler.step(optim)
            scaler.update()

            batch_mae, batch_rmse, _ = evaluate_masked_tensor_metrics(
                output.detach(), batch.energy.detach(), batch.energy_mask.detach()
            )
            train_loss_sum += loss.item()
            train_mae_sum += batch_mae
            train_rmse_sum += batch_rmse
            train_bar.set_postfix(
                loss=f"{loss.item():.4e}", mae=f"{batch_mae:.4e}", grad=f"{float(grad_norm):.3f}"
            )

        train_loss = train_loss_sum / max(len(train_dataloader), 1)
        train_mae = train_mae_sum / max(len(train_dataloader), 1)
        train_rmse = train_rmse_sum / max(len(train_dataloader), 1)

        net.eval()
        valid_bar = tqdm(valid_dataloader, desc=f"Epoch {epoch + 1}/{config['epochs']} [valid]")
        with torch.no_grad():
            for batch in valid_bar:
                batch = batch.to(device)
                with autocast():
                    output = net(batch)
                    loss = compute_training_loss(output, batch, checkpoint_config)

                    batch_mae, batch_rmse, batch_max_abs = evaluate_masked_tensor_metrics(
                        output, batch.energy, batch.energy_mask
                    )
                    output_model_unscaled = (
                        target_transform.inverse_tensor(output)
                        if target_transform is not None and hasattr(batch, "energy_model_raw")
                        else output
                    )
                    output_raw = model_tensor_to_raw_tensor(output_model_unscaled)
                    target_raw = model_tensor_to_raw_tensor(
                        batch.energy_raw if hasattr(batch, "energy_raw") else batch.energy
                    )
                    mask_raw = model_tensor_to_raw_tensor(
                        batch.energy_raw_mask if hasattr(batch, "energy_raw_mask") else batch.energy_mask
                    )
                    batch_mae_raw, batch_rmse_raw, batch_max_abs_raw = evaluate_masked_tensor_metrics(
                        output_raw, target_raw, mask_raw
                    )
                    valid_pred_raw_chunks.append(output_raw.detach().cpu())
                    valid_target_raw_chunks.append(target_raw.detach().cpu())
                    valid_mask_raw_chunks.append(mask_raw.detach().cpu())
                valid_loss_sum += loss.item()
                valid_mae_sum += batch_mae
                valid_rmse_sum += batch_rmse
                valid_max_abs_sum += batch_max_abs
                valid_mae_raw_sum += batch_mae_raw
                valid_rmse_raw_sum += batch_rmse_raw
                valid_max_abs_raw_sum += batch_max_abs_raw
                valid_bar.set_postfix(loss=f"{loss.item():.4e}", mae=f"{batch_mae:.4e}")

        valid_loss = valid_loss_sum / max(len(valid_dataloader), 1)
        valid_mae = valid_mae_sum / max(len(valid_dataloader), 1)
        valid_rmse = valid_rmse_sum / max(len(valid_dataloader), 1)
        valid_max_abs = valid_max_abs_sum / max(len(valid_dataloader), 1)
        valid_mae_raw = valid_mae_raw_sum / max(len(valid_dataloader), 1)
        valid_rmse_raw = valid_rmse_raw_sum / max(len(valid_dataloader), 1)
        valid_max_abs_raw = valid_max_abs_raw_sum / max(len(valid_dataloader), 1)
        valid_diag_mae = compute_model_basis_diag_mae(
            torch.cat(valid_pred_raw_chunks, dim=0),
            torch.cat(valid_target_raw_chunks, dim=0),
            torch.cat(valid_mask_raw_chunks, dim=0),
        )
        valid_diag_score = compute_diag_score(
            valid_diag_mae,
            checkpoint_config["diag_score_normalizers"],
            checkpoint_config["diag_score_weights"],
        )
        model_score = valid_diag_score if checkpoint_config["best_model_metric"] == "diag_score" else valid_mae
        scheduler.step(model_score)
        current_lr = optim.param_groups[0]["lr"]

        epoch_metrics = {
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "valid_loss": valid_loss,
            "valid_mae": valid_mae,
            "valid_rmse": valid_rmse,
            "valid_max_abs": valid_max_abs,
            "valid_mae_raw": valid_mae_raw,
            "valid_rmse_raw": valid_rmse_raw,
            "valid_max_abs_raw": valid_max_abs_raw,
            "valid_xx_mae_raw": valid_diag_mae["xx"],
            "valid_yy_mae_raw": valid_diag_mae["yy"],
            "valid_xy_mae_raw": valid_diag_mae["xy"],
            "valid_diag_score": valid_diag_score,
        }
        history.append(epoch_metrics)
        pd.DataFrame(history).to_csv(config["log_csv_path"], index=False)

        improved = model_score < best_model_score
        if improved:
            best_model_score = model_score
            epochs_without_improvement = 0
            save_checkpoint(
                config["best_model_path"],
                epoch + 1,
                net,
                optim,
                scheduler,
                scaler,
                epoch_metrics,
                checkpoint_config,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            config["last_model_path"],
            epoch + 1,
            net,
            optim,
            scheduler,
            scaler,
            epoch_metrics,
            checkpoint_config,
        )

        print(
            f"[Epoch {epoch + 1:03d}/{config['epochs']:03d}] "
            f"lr={current_lr:.3e} | "
            f"train_loss={train_loss:.6e} | train_mae={train_mae:.6e} | train_rmse={train_rmse:.6e} | "
            f"valid_loss={valid_loss:.6e} | valid_mae={valid_mae:.6e} | "
            f"valid_rmse={valid_rmse:.6e} | valid_max_abs={valid_max_abs:.6e} | "
            f"valid_mae_raw={valid_mae_raw:.6e} | valid_rmse_raw={valid_rmse_raw:.6e} | "
            f"valid_diag_score={valid_diag_score:.6e}"
        )
        if improved:
            print(f"  -> best model updated: {config['best_model_path']}")
        if epochs_without_improvement >= config["early_stopping_patience"]:
            print(
                f"Early stopping triggered at epoch {epoch + 1} "
                f"(no valid_mae improvement for {config['early_stopping_patience']} epochs)."
            )
            break

    print(f"Training finished. Best model score = {best_model_score:.6e}")
    print(f"Best checkpoint: {config['best_model_path']}")
    print(f"Last checkpoint: {config['last_model_path']}")
    print(f"Training log CSV: {config['log_csv_path']}")

    evaluate_best_checkpoint(net, test_dataloader, device, scaler, checkpoint_config, config, target_transform)


def evaluate_best_checkpoint(net, test_dataloader, device, scaler, checkpoint_config, config, target_transform=None):
    if os.path.exists(config["best_model_path"]):
        best_ckpt = torch.load(config["best_model_path"], map_location=device)
        net.load_state_dict(best_ckpt["model_state_dict"])
        net.eval()

        test_loss_sum = 0.0
        test_mae_sum = 0.0
        test_rmse_sum = 0.0
        test_max_abs_sum = 0.0
        test_mae_raw_sum = 0.0
        test_rmse_raw_sum = 0.0
        test_max_abs_raw_sum = 0.0
        test_records = []
        pred_tensors_for_metrics = []
        target_tensors_for_metrics = []
        mask_tensors_for_metrics = []
        test_record_idx = 0

        with torch.no_grad():
            for batch in test_dataloader:
                batch = batch.to(device)
                with autocast():
                    output = net(batch)
                    loss = compute_training_loss(output, batch, checkpoint_config)

                batch_mae, batch_rmse, batch_max_abs = evaluate_masked_tensor_metrics(
                    output, batch.energy, batch.energy_mask
                )
                output_model_unscaled = (
                    target_transform.inverse_tensor(output)
                    if target_transform is not None and hasattr(batch, "energy_model_raw")
                    else output
                )
                output_for_export = model_tensor_to_raw_tensor(output_model_unscaled)
                target_for_export = model_tensor_to_raw_tensor(
                    batch.energy_raw if hasattr(batch, "energy_raw") else batch.energy
                )
                mask_for_export = model_tensor_to_raw_tensor(
                    batch.energy_raw_mask if hasattr(batch, "energy_raw_mask") else batch.energy_mask
                )
                batch_mae_raw, batch_rmse_raw, batch_max_abs_raw = evaluate_masked_tensor_metrics(
                    output_for_export, target_for_export, mask_for_export
                )
                test_loss_sum += loss.item()
                test_mae_sum += batch_mae
                test_rmse_sum += batch_rmse
                test_max_abs_sum += batch_max_abs
                test_mae_raw_sum += batch_mae_raw
                test_rmse_raw_sum += batch_rmse_raw
                test_max_abs_raw_sum += batch_max_abs_raw

                output_cpu = output_for_export.detach().cpu()
                target_cpu = target_for_export.detach().cpu()
                mask_cpu = mask_for_export.detach().cpu()
                pred_tensors_for_metrics.append(output_cpu)
                target_tensors_for_metrics.append(target_cpu)
                mask_tensors_for_metrics.append(mask_cpu)
                for i in range(output_cpu.shape[0]):
                    test_records.append(
                        {
                            "test_order_index": test_record_idx,
                            "target_tensor": target_cpu[i].tolist(),
                            "pred_tensor": output_cpu[i].tolist(),
                            "mask_tensor": mask_cpu[i].tolist(),
                        }
                    )
                    test_record_idx += 1

        test_loss = test_loss_sum / max(len(test_dataloader), 1)
        test_mae = test_mae_sum / max(len(test_dataloader), 1)
        test_rmse = test_rmse_sum / max(len(test_dataloader), 1)
        test_max_abs = test_max_abs_sum / max(len(test_dataloader), 1)
        test_mae_raw = test_mae_raw_sum / max(len(test_dataloader), 1)
        test_rmse_raw = test_rmse_raw_sum / max(len(test_dataloader), 1)
        test_max_abs_raw = test_max_abs_raw_sum / max(len(test_dataloader), 1)

        with open(config["test_predictions_path"], "w", encoding="utf-8") as f:
            json.dump(test_records, f, ensure_ascii=False)

        test_metrics = compute_original_tensor_metrics(
            torch.cat(pred_tensors_for_metrics, dim=0),
            torch.cat(target_tensors_for_metrics, dim=0),
            torch.cat(mask_tensors_for_metrics, dim=0),
        )
        with open(config["test_metrics_path"], "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)

        print(
            f"[Test @ Best] "
            f"loss={test_loss:.6e} | mae={test_mae:.6e} | "
            f"rmse={test_rmse:.6e} | max_abs={test_max_abs:.6e} | "
            f"mae_raw={test_mae_raw:.6e} | rmse_raw={test_rmse_raw:.6e} | max_abs_raw={test_max_abs_raw:.6e}"
        )
        print(f"Test predictions JSON: {config['test_predictions_path']}")
        print(f"Test metrics JSON: {config['test_metrics_path']}")
    else:
        print(f"Best checkpoint not found at {config['best_model_path']}, skip final test evaluation.")
