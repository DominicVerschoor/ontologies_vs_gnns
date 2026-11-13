import os
import sys
import random
import statistics

import re
import gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    average_precision_score
)
from torch_geometric.utils import degree

def cleanup():
    """Release GPU memory after each config, success or failure -- prevents
    fragmentation-related OOMs that look random across a long sweep."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# VRAM left outside the allocator cap for what CUDA allocates past PyTorch's caching allocator
# (cuBLAS/cuSPARSE workspaces, lazily loaded kernels): measured at 0.04 GiB after training the
# largest config (Pure SULO R-GCN 512-d x 3 layers, 20 bases), so 0.5 GiB is >10x that.
CUDA_HEADROOM_BYTES = 2**29

def limit_cuda_memory_to_free_vram():
    """Caps this process's CUDA allocator at the VRAM that is free now minus
    CUDA_HEADROOM_BYTES, and returns the cap in bytes. Under WSL2 the Windows driver lets
    CUDA overcommit VRAM into system RAM, so an over-large model does not raise an OOM: it
    fails later with a driver error (e.g. 'CUDA driver error: device not ready') that no OOM
    handler recognises. With the cap, PyTorch raises torch.cuda.OutOfMemoryError before the
    driver is asked for more. Call once, before the first CUDA allocation (memory this
    process already holds is not counted as free, so a second call lowers the cap)."""
    free, total = torch.cuda.mem_get_info()
    cap = free - CUDA_HEADROOM_BYTES
    torch.cuda.set_per_process_memory_fraction(cap / total)
    return cap

def set_seed(seed):
    """Seeds python, numpy and torch (CPU + all CUDA devices): model init and
    dropout. GPU scatter ops (e.g. inside RGCNConv) can still differ in the last
    float bits."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_relation_dict(ont):
    return getattr(ont, 'enum_relations', {})

def compute_class_specific_metrics(targets, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    precisions = precision_score(targets, preds, average=None, labels=[0, 1], zero_division=0)
    recalls = recall_score(targets, preds, average=None, labels=[0, 1], zero_division=0)
    f1s = f1_score(targets, preds, average=None, labels=[0, 1], zero_division=0)
    return {
        'class0_precision': precisions[0], 'class0_recall': recalls[0], 'class0_f1': f1s[0],
        'class1_precision': precisions[1], 'class1_recall': recalls[1], 'class1_f1': f1s[1],
    }

def get_sulo_node_class_mask(ont, class_name):
    node_names = list(ont.enum_nodes.keys())
    mask = torch.zeros(ont.num_nodes, dtype=torch.bool)
    for name, idx in ont.enum_nodes.items():
        if class_name.lower() in name.lower():
            mask[idx] = True
    return mask

def _fmt(metric):
    """4-decimal metric, or a marker for runs that produced none (OOM/failure)."""
    return f"{metric:.4f}" if metric is not None else "OOM/FAIL"

def mask_graph_topology(graph_data, ontology_obj, target_relation=None, target_class=None):
    """
    Returns a copy of graph_data with targeted connections removed.
    - target_relation: mask out specific edge_type IDs (for direct Base relations).
    - target_class: mask out edges connected to specific node class names (for abstract SULO).
    """
    edge_index = graph_data.edge_index
    edge_type = graph_data.edge_type

    relation_dict = get_relation_dict(ontology_obj)
    
    if target_relation is not None:
        raw_rel_ids = [r_id for r_uri, r_id in relation_dict.items() if target_relation.lower() in r_uri.lower()]
        rel_ids = [rid * 2 + d for rid in raw_rel_ids for d in (0, 1)]  # expand to forward + inverse edge_type ids
        if not rel_ids:
            return None, 0.0
        mask = ~torch.isin(edge_type, torch.tensor(rel_ids, device=edge_type.device))
        
    elif target_class is not None:
        node_class_mask = get_sulo_node_class_mask(ontology_obj, target_class)
        if node_class_mask is None: return None, 0.0
        mask = ~(node_class_mask[edge_index[0]] | node_class_mask[edge_index[1]])
        
    else:
        return graph_data, 0.0
        
    masked_data = graph_data.clone()
    masked_data.edge_index = edge_index[:, mask]
    masked_data.edge_type = edge_type[mask]
    
    baseline_edges = edge_index.size(1)
    masked_edges = masked_data.edge_index.size(1)
    pct_removed = ((baseline_edges - masked_edges) / baseline_edges) * 100.0
    
    return masked_data, pct_removed

def get_graph_mask(graph_data, ontology_obj, schema_label, concept_name):
    """Boolean node mask for a target concept, matched by URI segment (SULO/Base
    class instances) or by relation name (Base datatype literal leaves)."""
    mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool)
    relation_dict = get_relation_dict(ontology_obj)

    def _normalize(s):
        return re.sub(r'[^a-z0-9]', '', s.lower())

    stripped_concept = concept_name[3:] if concept_name.lower().startswith('has') else concept_name
    target_clean = _normalize(concept_name)
    target_stripped_clean = _normalize(stripped_concept)

    # ── PATH 1: RESOLVE PHYSICAL CLASS INSTANCES + SULO PER-INSTANCE SUFFIXES ──
    # Literals (always quoted in raw N-Triples form, e.g. '"text"^^<type>')
    # are skipped -- URIs never are. Free-text literals can otherwise
    # spuriously match a concept name their text happens to end with (e.g.
    # a CaseDescription literal ending in "...vehicle").
    for name, idx in ontology_obj.enum_nodes.items():
        if name.startswith('"'):
            continue
        raw_segments = re.split(r'[/|#|:]', name)
        norm_segments = [_normalize(seg) for seg in raw_segments]
        
        matched = False
        for seg in norm_segments:
            seg_stripped = seg[3:] if seg.startswith('has') else seg
            if (seg == target_clean or seg == target_stripped_clean or 
                seg_stripped == target_clean or seg_stripped == target_stripped_clean or
                (len(target_stripped_clean) > 3 and seg.endswith(target_stripped_clean))):
                matched = True
                break
        if matched:
            mask[idx] = True

    # ── PATH 2: RESOLVE LITERAL LEAF TARGETS (BASE DATATYPE RELATION PATHS) ───
    matching_rel_ids = []
    for rel_uri, rel_id in relation_dict.items():
        rel_segments = re.split(r'[/|#|:]', rel_uri)
        norm_rel_segments = [_normalize(seg) for seg in rel_segments]
        
        for seg in norm_rel_segments:
            seg_stripped = seg[3:] if seg.startswith('has') else seg
            if (seg == target_clean or seg == target_stripped_clean or 
                seg_stripped == target_clean or seg_stripped == target_stripped_clean):
                matching_rel_ids.append(rel_id * 2)  # forward direction only: subject -> literal object
                break

    if matching_rel_ids:
        edge_mask = torch.isin(graph_data.edge_type, torch.tensor(matching_rel_ids, device=graph_data.edge_type.device))
        target_nodes = graph_data.edge_index[1, edge_mask]
        mask[target_nodes] = True

    return mask

# Training and evaluation are full-batch: every layer aggregates each node's complete
# neighbourhood, as in Schlichtkrull et al. 2018 (R-GCN entity classification) and Kipf &
# Welling 2017 (GCN). Neighbour sampling hid hub identities: a shared Position/<n> node has
# ~1,800 neighbours and a single edge towards its rank, which a fanout of 10 almost never drew.
#   https://arxiv.org/abs/1703.06103
#   https://arxiv.org/abs/1609.02907
# No gradient clipping: neither Hamilton et al. 2017 nor Schlichtkrull et al. 2018 (R-GCN) clip.
# The pre-step gradient norm is logged instead (history['grad_norm'], CSV column max_grad_norm).
#   https://arxiv.org/abs/1706.02216

def _grad_norm(model):
    """L2 norm of all parameter gradients together (what clip_grad_norm_ would clip), unmodified."""
    return torch.nn.utils.get_total_norm([p.grad for p in model.parameters() if p.grad is not None]).item()

def _train_step(model, graph, indices, optimizer, loss_fn):
    """One full-batch gradient step on the labels of `indices`; returns (loss, pre-step gradient norm)."""
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(graph)[indices].squeeze(1), graph.y[indices].float())
    loss.backward()
    grad_norm = _grad_norm(model)
    optimizer.step()
    return loss.item(), grad_norm

def _predict(model, graph, indices):
    """Logits of `indices` from one eval-mode forward pass over the whole graph."""
    model.eval()
    with torch.no_grad():
        return model(graph)[indices].squeeze(1)

def _binary_metrics(targets, probs):
    """Threshold-free ROC-/PR-AUC plus accuracy/precision/recall/F1 at the fixed 0.5 threshold."""
    preds = (probs >= 0.5).astype(int)
    return {
        'accuracy': accuracy_score(targets, preds),
        'precision': precision_score(targets, preds, zero_division=0),
        'recall': recall_score(targets, preds, zero_division=0),
        'f1': f1_score(targets, preds, zero_division=0),
        'roc_auc': roc_auc_score(targets, probs),
        'pr_auc': average_precision_score(targets, probs),
    }

def train_final_model(model, graph_data, combined_indices, epochs, lr=0.001, weight_decay=1e-4, save_path=None):
    """
    Trains the final selected model full-batch on the train+validation combined
    indices for a fixed number of epochs (no validation, so no early stopping).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    graph = graph_data.clone().to(device)   # Data.to moves in place; the caller's graph stays put
    indices = combined_indices.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    try:
        for epoch in range(epochs):
            loss, grad_norm = _train_step(model, graph, indices, optimizer, loss_fn)
            print(f"Final Train Epoch {epoch+0:03d} | Train Loss: {loss:.4f} | Grad Norm: {grad_norm:.4f}")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
            }, save_path)
            print(f"Final trained model saved to {save_path}")

        return model
    finally:
        model.cpu()
        del graph, indices, optimizer
        cleanup()

def evaluate_test_set(model, graph_data, test_indices):
    """
    Evaluates the finalized model on the unseen test split, full-batch, using a fixed 0.5 threshold.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    graph = graph_data.clone().to(device)

    try:
        probs = torch.sigmoid(_predict(model, graph, test_indices.to(device))).cpu().numpy()
        targets = graph_data.y[test_indices].numpy()
        metrics = _binary_metrics(targets, probs)
        metrics['num_parameters'] = int(sum(p.numel() for p in model.parameters()))
        metrics.update(compute_class_specific_metrics(targets, probs))
        return metrics
    finally:
        model.cpu()
        del graph
        cleanup()

def compute_entity_reuse_stats(graph_data, ontology_obj, schema_label, concept_names, max_hops=3):
    """
    For each concept: instance count, degree distribution, and how many
    distinct Case nodes it reaches (expanding outward hop-by-hop, since some
    concepts attach directly to Case while others sit further out via an
    intermediate entity).

    Reuse(x) = unique instances of x / distinct cases connected to x. Near 1
    means each case gets its own dedicated instance ('dead end'); << 1 means
    many cases share a small pool of instances ('hub').
    """
    src, dst = graph_data.edge_index
    case_mask = get_graph_mask(graph_data, ontology_obj, schema_label, 'Case')
    num_cases = int(case_mask.sum().item())
    node_degrees = degree(dst, num_nodes=graph_data.num_nodes)

    rows = []
    for concept in concept_names:
        concept_mask = get_graph_mask(graph_data, ontology_obj, schema_label, concept)
        num_instances = int(concept_mask.sum().item())

        if num_instances == 0:
            rows.append({
                'entity': concept, 'num_unique_instances': 0, 'num_cases_connected': 0,
                'reuse_ratio': float('nan'), 'mean_degree': float('nan'), 'median_degree': float('nan'),
                'p95_degree': float('nan'), 'max_degree': float('nan'), 'cases_per_entity': float('nan'),
            })
            continue

        concept_degrees = node_degrees[concept_mask].float()

        # Multi-source BFS from every instance of this concept, hop by hop,
        # tracking which Case nodes get reached along the way.
        visited = concept_mask.clone()
        frontier = concept_mask.clone()
        reached_cases = frontier & case_mask
        for _ in range(max_hops):
            if not frontier.any():
                break
            active_src = frontier[src]
            active_dst = frontier[dst]
            next_frontier = torch.zeros(graph_data.num_nodes, dtype=torch.bool)
            next_frontier[dst[active_src]] = True
            next_frontier[src[active_dst]] = True
            next_frontier &= ~visited
            visited |= next_frontier
            reached_cases |= next_frontier & case_mask
            frontier = next_frontier

        num_cases_connected = int(reached_cases.sum().item())
        reuse_ratio = num_instances / num_cases_connected if num_cases_connected > 0 else float('nan')
        cases_per_entity = num_cases_connected / num_instances

        rows.append({
            'entity': concept,
            'num_unique_instances': num_instances,
            'num_cases_connected': num_cases_connected,
            'reuse_ratio': reuse_ratio,
            'mean_degree': float(concept_degrees.mean().item()),
            'median_degree': float(concept_degrees.median().item()),
            'p95_degree': float(torch.quantile(concept_degrees, 0.95).item()),
            'max_degree': float(concept_degrees.max().item()),
            'cases_per_entity': cases_per_entity,
        })

    return pd.DataFrame(rows), num_cases, case_mask


def train_model(model, graph_data, train_indices, val_indices,
                 epochs=100, lr=0.001, weight_decay=5e-4, eval_frequency=1,
                 save_path=None, resume=False, patience=10, monitor='roc_auc', start_from_epoch=0):
    """
    Model-agnostic full-batch training loop: one gradient step per epoch on the
    train labels, scored on the val labels every eval_frequency epochs. `model`
    must accept forward(data) on the whole graph and read whatever node features
    and edge attributes it needs from it (e.g. GraphSAGE simply ignores
    data.edge_type), so this loop never needs to branch on model type.

    start_from_epoch: warm-up. Epochs before it are trained and logged but neither
    checkpointed nor counted towards patience, as Keras' EarlyStopping(start_from_epoch).
    Without it, a val spike in the unstable first epochs can become the score to beat
    and stop the run before it gets there for real.
    """
    if start_from_epoch >= epochs:
        raise ValueError(f"start_from_epoch {start_from_epoch} leaves no epoch to select from (epochs {epochs})")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    graph = graph_data.clone().to(device)   # Data.to moves in place; the caller's graph stays put
    train_idx, val_idx = train_indices.to(device), val_indices.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    start_epoch = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'val_accuracy': [], 'val_precision': [], 'val_recall': [],
        'val_f1': [], 'val_roc_auc': [], 'val_pr_auc': [],
        'grad_norm': [],   # pre-step L2 norm over all parameters
    }

    best_metric_val = -float('inf')
    epochs_no_improve = 0
    best_model_state = None
    best_epoch = 0

    # Ensure tuning runs never silently resume old checkpoints
    if resume and save_path and os.path.exists(save_path):
        print(f"Found checkpoint at {save_path}. Resuming...")
        checkpoint = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        loaded_history = checkpoint.get('history', {})
        for k, v in loaded_history.items():
            if k in history:
                history[k] = v
        print(f"Resuming from epoch {start_epoch}.")

    try:
        for epoch in range(start_epoch, epochs):
            avg_train_loss, grad_norm = _train_step(model, graph, train_idx, optimizer, loss_fn)
            history['train_loss'].append(avg_train_loss)
            history['grad_norm'].append(grad_norm)

            if (epoch + 1) % eval_frequency == 0 or epoch == epochs - 1:
                logits, targets = _predict(model, graph, val_idx), graph.y[val_idx].float()
                val_metrics = {
                    'loss': loss_fn(logits, targets).item(),
                    **_binary_metrics(targets.cpu().numpy(), torch.sigmoid(logits).cpu().numpy()),
                }
                history['val_loss'].append(val_metrics['loss'])
                history['val_accuracy'].append(val_metrics['accuracy'])
                history['val_precision'].append(val_metrics['precision'])
                history['val_recall'].append(val_metrics['recall'])
                history['val_f1'].append(val_metrics['f1'])
                history['val_roc_auc'].append(val_metrics['roc_auc'])
                history['val_pr_auc'].append(val_metrics['pr_auc'])

                print(f"Epoch {epoch+0:03d} | Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
                      f"Val F1: {val_metrics['f1']:.4f} | Val ROC-AUC: {val_metrics['roc_auc']:.4f} | "
                      f"Grad Norm: {grad_norm:.4f}")

                current_metric = val_metrics[monitor]

                if epoch < start_from_epoch:
                    continue

                if np.isnan(current_metric):
                    epochs_no_improve += eval_frequency
                    continue

                if current_metric > best_metric_val:
                    best_metric_val = current_metric
                    best_epoch = epoch
                    epochs_no_improve = 0

                    history['best_epoch'] = best_epoch
                    if save_path:
                        # Write best weights directly to disk; free memory from RAM copy
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'history': history,
                        }, save_path)
                        best_model_state = None
                        print(f"  -> Best model saved to disk! ({monitor}: {best_metric_val:.4f})")
                    else:
                        # copy=True: on CPU, .cpu() would alias the live weights the next steps overwrite
                        best_model_state = {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()}
                        print(f"  -> Best model saved to RAM! ({monitor}: {best_metric_val:.4f})")
                else:
                    epochs_no_improve += eval_frequency
                    if epochs_no_improve >= patience:
                        print(f"\nEarly stopping triggered! No improvement in {monitor} for {epochs_no_improve} epochs.")
                        break
            else:
                history['val_loss'].append(np.nan)
                history['val_accuracy'].append(np.nan)
                history['val_precision'].append(np.nan)
                history['val_recall'].append(np.nan)
                history['val_f1'].append(np.nan)
                history['val_roc_auc'].append(np.nan)
                history['val_pr_auc'].append(np.nan)
                print(f"Epoch {epoch+0:03d} | Train Loss: {avg_train_loss:.4f}")

        if save_path and os.path.exists(save_path):
            checkpoint = torch.load(save_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\nRestored best model weights from disk (Epoch {best_epoch} with {monitor} = {best_metric_val:.4f})")
        elif best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"\nRestored best model weights from RAM (Epoch {best_epoch} with {monitor} = {best_metric_val:.4f})")

        return model, history

    finally:
        model.cpu()
        del graph, train_idx, val_idx, optimizer
        cleanup()


def run_one_config(model_factory, graph_data, train_indices, val_indices,
                    hidden_list, bases, lr, wd, epochs, patience, save_path, seed=0, start_from_epoch=0):
    """
    Memory-safe single hyperparameter-configuration executor, returning
    val_roc_auc/best_epoch/status for use by a sweep. val_roc_auc is the peak from
    start_from_epoch on (see train_model), i.e. the restored checkpoint's.

    model_factory(hidden_list, bases) -> nn.Module: builds a fresh model for
        this configuration. All other architecture-specific construction
        (feature vocabulary sizes, num_relations, ...) must already be bound
        via closure by the caller. Pass bases=None for architectures with no
        relation-basis concept (e.g. GraphSAGE) -- the factory just ignores it.
    seed: seeds model initialisation and the whole training run.
    """
    start_time = time.time()

    set_seed(seed)
    model = model_factory(hidden_list, bases)

    history = None
    peak_auc = None
    best_epoch = None
    max_grad_norm = None
    status = 'ok'

    try:
        gpu_model, history = train_model(
            model=model, graph_data=graph_data,
            train_indices=train_indices, val_indices=val_indices,
            epochs=epochs, lr=lr, weight_decay=wd,
            save_path=save_path, resume=False, patience=patience, monitor='roc_auc',
            start_from_epoch=start_from_epoch,
        )
        auc_values = [
            x for x in history.get("val_roc_auc", [])[start_from_epoch:]
            if x is not None and not np.isnan(x)
        ]
        peak_auc = max(auc_values) if auc_values else None
        if peak_auc is None:
            status = 'no_valid_metric'

        best_epoch = history.get('best_epoch', None)
        max_grad_norm = max(history['grad_norm'])
        elapsed = time.time() - start_time
        print(f"  -> Finished in {elapsed:.1f}s | Peak ROC-AUC: {_fmt(peak_auc)} (Best Epoch: {best_epoch})")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("  -> CUDA OOM: Skipping this configuration.")
            elapsed = time.time() - start_time
            status = 'oom'
            for attr in ('last_traceback', 'last_type', 'last_value'):
                if hasattr(sys, attr):
                    delattr(sys, attr)
        else:
            raise e
    finally:
        if 'model' in locals():
            model.cpu()
            del model
        if 'gpu_model' in locals():
            gpu_model.cpu()
            del gpu_model
        if 'history' in locals():
            del history
        if 'e' in locals():
            del e

        cleanup()

    return {
        'num_bases': bases, 'lr': lr, 'weight_decay': wd,
        'val_roc_auc': peak_auc,
        'best_epoch': best_epoch, 'max_grad_norm': max_grad_norm,
        'runtime_sec': elapsed, 'status': status,
    }


def run_config_over_seeds(model_factory, graph_data, train_indices, val_indices,
                          hidden_list, bases, lr, wd, epochs, patience, save_path, seeds, start_from_epoch=0):
    """
    run_one_config once per seed (checkpoint '<save_path>_seed<s>.pt'), with an early-stopping
    warm-up of start_from_epoch epochs (see train_model; the main tuning uses none). The config is
    scored by its mean val ROC-AUC over the seeds, so a seed that collapses early lowers
    the score instead of deciding it. Its best epoch, used for the final retrain, is the
    median per-seed best epoch (median_low: an epoch one of the seeds actually reached).
    A seed that fails (e.g. OOM) fails the config; the remaining seeds are not run, since
    they need the same memory.
    """
    runs = []
    for s in seeds:
        runs.append(run_one_config(
            model_factory, graph_data, train_indices, val_indices, hidden_list, bases, lr, wd,
            epochs=epochs, patience=patience, save_path=f"{save_path.removesuffix('.pt')}_seed{s}.pt", seed=s,
            start_from_epoch=start_from_epoch,
        ))
        if runs[-1]['status'] != 'ok':
            break

    aucs = [r['val_roc_auc'] for r in runs]
    best_epochs = [r['best_epoch'] for r in runs]
    grad_norms = [r['max_grad_norm'] for r in runs if r['max_grad_norm'] is not None]
    ok = runs[-1]['status'] == 'ok'
    return {
        'num_bases': bases, 'lr': lr, 'weight_decay': wd,
        'val_roc_auc': float(np.mean(aucs)) if ok else None,
        'val_roc_auc_std': float(np.std(aucs, ddof=1)) if ok and len(aucs) > 1 else None,
        'val_roc_auc_seeds': ';'.join(_fmt(a) for a in aucs),
        'best_epoch': statistics.median_low(best_epochs) if ok else None,
        'best_epoch_seeds': ';'.join(str(e) for e in best_epochs),
        'max_grad_norm': max(grad_norms) if grad_norms else None,
        'runtime_sec': sum(r['runtime_sec'] for r in runs),
        'status': runs[-1]['status'],
    }


def _ended_by_epoch_cap(best_epoch_seeds, epochs, patience):
    """
    Whether any seed of a run ('<e0>;<e1>;...' best epochs) was cut off by the epoch cap
    rather than by early stopping. A seed stops early at epoch best_epoch + patience, which
    only happens before the cap when best_epoch < epochs - patience; otherwise it ends at
    epoch epochs - 1 while possibly still improving.
    """
    return any(int(e) >= epochs - patience for e in str(best_epoch_seeds).split(';'))


def runs_ended_by_epoch_cap(results, epochs, patience):
    """The trained tuning runs (phases 1-3) with a seed cut off by the epoch cap (_ended_by_epoch_cap)."""
    trained = results[(results['runtime_sec'] > 0) & (results['status'] == 'ok')
                      & results['best_epoch_seeds'].notna()]
    return trained[trained['best_epoch_seeds'].map(lambda e: _ended_by_epoch_cap(e, epochs, patience))]


def finished_tuning_results(csv_path, grid_dim, grid_depth, tuning_seeds, final_seeds, metric='val_roc_auc'):
    """
    The results CSV of a KG whose search already finished under this protocol, or None if it
    has to be (re-)run: the file is missing, from the single-seed protocol (no per-seed
    columns), from another dim x depth grid or seed count, or lacks a phase-4 row for any
    final seed (the run was interrupted). Other constants are not checked, so move the CSVs
    away after changing one. Lets a crashed run resume at the first unfinished KG.
    metric: the selection score's column ('val_mrr' for link prediction).
    """
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if f'{metric}_seeds' not in df.columns:
        return None
    phase1 = df[df['phase'] == '1_Architecture']
    grid = {(l, d) for l in grid_depth for d in (grid_dim if l > 1 else [None])}
    logged = {(int(l), None if pd.isna(d) else int(d)) for l, d in zip(phase1['layers'], phase1['hidden_dim'])}
    trained = df[(df['runtime_sec'] > 0) & (df['status'] == 'ok') & df['phase'].str.match(r'[123]_')]
    seed_counts = {len(str(v).split(';')) for v in trained[f'{metric}_seeds']}
    final = df.loc[df['phase'] == '4_Final_Test_Evaluation', 'seed']
    if logged != grid or seed_counts - {len(tuning_seeds)} or sorted(final.astype(int)) != sorted(final_seeds):
        return None
    return df


def run_sequential_optimization_with_verification(
    model_factory, graph_data, train_indices, val_indices, test_indices,
    schema_label, save_dir, csv_path,
    grid_dim, grid_depth, grid_bases, grid_lr, grid_wd,
    default_bases, default_lr, default_wd,
    epochs, patience, tuning_seeds=(0,), final_seeds=(0,), start_from_epoch=0,
):
    """
    Hyperparameter search: a joint dim x depth grid, then greedy bases -> lr -> wd at
    the winning architecture, then a 2x2 verification grid over the top-2 dims (at the
    winning depth) x top-2 (bases, lr, wd) combos from all of phase 2, then a final
    held-out test evaluation of the overall winner, once per seed in final_seeds.
    Returns the full results log as a DataFrame.

    Phase 1 is joint because the depth a KG needs decides which dims can be compared:
    in Pure/Hybrid SULO the grid rank is 3 hops from a RaceEntry, so dims ranked at a
    fixed depth of 2 were ranked without that signal. A 1-layer model has no hidden
    layer, so it is trained once (hidden_dim None).

    tuning_seeds: every configuration is trained once per seed and scored by its mean
        val ROC-AUC (see run_config_over_seeds), so all configurations (and all KGs)
        see the same seeds.
    start_from_epoch: early-stopping warm-up of every tuning run (see train_model): without
        it, a seed that kept a val spike from the first epochs as its best scored that
        spike, so the search compared stopping artefacts instead of configurations. The
        final retrain trains a fixed best_epoch + 1 epochs and has no early stopping.

    model_factory(hidden_list, bases) -> nn.Module: builds a fresh model.
    All other architecture-specific construction (feature vocabulary sizes,
    num_relations, ...) must already be bound via closure by the caller.
    For architectures with no relation-basis concept (e.g.
    GraphSAGE), pass grid_bases=[None] and default_bases=None -- the
    factory just ignores the bases argument it's given.

    save_dir: root directory for this model type's checkpoints (e.g.
        f"{volume}rgcn/") -- per-phase checkpoints go in
        f"{save_dir}robust_tuning_logs/", the final unified-eval checkpoint
        goes directly under save_dir.
    """
    cleanup()

    results = []
    phase_2_results = {}  # (bases, lr, wd) -> val ROC-AUC, from every phase-2 sweep

    def run(hidden_list, bases, lr, wd, name):
        return run_config_over_seeds(
            model_factory, graph_data, train_indices, val_indices, hidden_list, bases, lr, wd,
            epochs=epochs, patience=patience,
            save_path=f"{save_dir}robust_tuning_logs/{schema_label}_{name}.pt", seeds=tuning_seeds,
            start_from_epoch=start_from_epoch,
        )

    def log_and_save(phase, parameter, row, d, l, b, lr, wd):
        row_dict = {
            'phase': phase, 'parameter_tested': parameter,
            'val_roc_auc': row['val_roc_auc'], 'val_roc_auc_std': row.get('val_roc_auc_std'),
            'val_roc_auc_seeds': row.get('val_roc_auc_seeds'),
            'runtime_sec': row['runtime_sec'], 'status': row['status'], 'hidden_dim': d, 'layers': l,
            'num_bases': b, 'lr': lr, 'weight_decay': wd,
            'best_epoch': row['best_epoch'], 'best_epoch_seeds': row.get('best_epoch_seeds'),
            'max_grad_norm': row.get('max_grad_norm'),
        }
        results.append(row_dict)
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"  -> [{phase}] Mean Metric: {_fmt(row['val_roc_auc'])} (seeds: {row.get('val_roc_auc_seeds')}) "
              f"| Best Epoch: {row['best_epoch']} | Status: {row['status']}")
        return row_dict

    def scored(row):
        return row['status'] == 'ok' and row['val_roc_auc'] is not None

    def reused(row):
        """A row already trained in an earlier phase, logged again without its runtime."""
        return dict(row, runtime_sec=0.0)

    print(f"\n=======================================================")
    print(f" STARTING SEQUENTIAL OPTIMIZATION: {schema_label.upper()} (seeds {list(tuning_seeds)})")
    print(f"=======================================================")

    # ── PHASE 1: JOINT HIDDEN DIMENSION x DEPTH GRID ─────────────────────────
    print(f"\n>>> Phase 1: Tuning Hidden Dimension x Depth")
    for layers in grid_depth:
        for dim in (grid_dim if layers > 1 else [None]):
            print(f" [RUNNING] Depth: {layers} | Hidden Dim: {dim}")
            row = run([dim] * (layers - 1), default_bases, default_lr, default_wd, f"1_L{layers}_dim{dim}")
            log_and_save("1_Architecture", f"layers_{layers}_dim_{dim}", row,
                         dim, layers, default_bases, default_lr, default_wd)

    phase_1 = [r for r in results if scored(r)]
    if not phase_1:
        raise RuntimeError(f"every phase-1 configuration failed for {schema_label}; see {csv_path}")
    best_arch_row = max(phase_1, key=lambda r: r['val_roc_auc'])
    best_dim, best_depth = best_arch_row['hidden_dim'], best_arch_row['layers']
    top_dims = [r['hidden_dim'] for r in sorted((r for r in phase_1 if r['layers'] == best_depth),
                                                key=lambda r: r['val_roc_auc'], reverse=True)[:2]]

    print(f"\n>>> Phase 1 winner: Depth={best_depth}, Dim={best_dim}; top dims at that depth for 2x2 validation: {top_dims}")
    cleanup()

    # ── PHASE 2A: RELATIONAL REGULARIZATION (BASES) SWEEP ────────────────────
    print(f"\n>>> Phase 2a: Tuning Bases (Architecture locked to L={best_depth}, H={best_dim})")
    hidden_list = [best_dim] * (best_depth - 1)
    best_bases_row = None

    for bases in grid_bases:
        if bases == default_bases:
            print(f" [SKIPPING] Reusing default Base state from Phase 1.")
            row = reused(best_arch_row)
        else:
            print(f" [RUNNING] Bases: {bases}")
            row = run(hidden_list, bases, default_lr, default_wd, f"2a_B{bases}")
        logged = log_and_save("2a_Bases", f"bases_{bases}", row,
                              best_dim, best_depth, bases, default_lr, default_wd)
        if scored(logged):
            phase_2_results[(bases, default_lr, default_wd)] = logged['val_roc_auc']
            if best_bases_row is None or logged['val_roc_auc'] > best_bases_row['val_roc_auc']:
                best_bases_row = logged
    best_bases = best_bases_row['num_bases']

    cleanup()

    # ── PHASE 2B: LEARNING RATE SWEEP ────────────────────────────────────────
    print(f"\n>>> Phase 2b: Tuning LR (Bases locked at {best_bases})")
    best_lr_row = None

    for lr in grid_lr:
        if lr == default_lr:
            print(f" [SKIPPING] Reusing current runner from Phase 2a.")
            row = reused(best_bases_row)
        else:
            print(f" [RUNNING] Learning Rate: {lr}")
            row = run(hidden_list, best_bases, lr, default_wd, f"2b_lr{lr}")
        logged = log_and_save("2b_LR", f"lr_{lr}", row, best_dim, best_depth, best_bases, lr, default_wd)
        if scored(logged):
            phase_2_results[(best_bases, lr, default_wd)] = logged['val_roc_auc']
            if best_lr_row is None or logged['val_roc_auc'] > best_lr_row['val_roc_auc']:
                best_lr_row = logged
    best_lr = best_lr_row['lr']

    cleanup()

    # ── PHASE 2C: WEIGHT DECAY SWEEP ─────────────────────────────────────────
    print(f"\n>>> Phase 2c: Tuning Weight Decay (Bases locked to {best_bases}, LR to {best_lr})")

    for wd in grid_wd:
        if wd == default_wd:
            print(f" [SKIPPING] Reusing current runner from Phase 2b.")
            row = reused(best_lr_row)
        else:
            print(f" [RUNNING] Weight Decay: {wd}")
            row = run(hidden_list, best_bases, best_lr, wd, f"2c_wd{wd}")
        logged = log_and_save("2c_WD", f"wd_{wd}", row, best_dim, best_depth, best_bases, best_lr, wd)
        if scored(logged):
            phase_2_results[(best_bases, best_lr, wd)] = logged['val_roc_auc']

    sorted_reg = sorted(phase_2_results.items(), key=lambda x: x[1], reverse=True)
    top_reg = [params for params, auc in sorted_reg[:2]]
    print(f"\n>>> Phase 2 Regularization coordinates isolated for 2x2 validation: {top_reg}")

    cleanup()

    # ── PHASE 3: 2X2 VERIFICATION GRID ───────────────────────────────────────
    print(f"\n=======================================================")
    print(f" STARTING PHASE 3 (2X2 INTERACTION VERIFICATION): {schema_label.upper()}")
    print(f"=======================================================")

    # A 1-layer model has only one "dim" (None), so this is then one row per regularisation combo.
    for verify_dim in top_dims:
        for verify_bases, verify_lr, verify_wd in top_reg:
            cached = next((r for r in results if scored(r) and
                           r['hidden_dim'] == verify_dim and r['layers'] == best_depth and
                           r['num_bases'] == verify_bases and r['lr'] == verify_lr and
                           r['weight_decay'] == verify_wd), None)
            if cached is not None:
                print(f" [SKIPPING] Caching verified cross-state: Dim={verify_dim} | Bases={verify_bases} | LR={verify_lr} | WD={verify_wd}")
                log_and_save("3_Verification_Cached", f"dim_{verify_dim}_reg_B{verify_bases}", reused(cached),
                             verify_dim, best_depth, verify_bases, verify_lr, verify_wd)
                continue

            print(f" [RUNNING Verification] Dim={verify_dim} | Bases={verify_bases} | LR={verify_lr} | WD={verify_wd}")
            row = run([verify_dim] * (best_depth - 1), verify_bases, verify_lr, verify_wd,
                      f"verify_dim{verify_dim}_B{verify_bases}_lr{verify_lr}_wd{verify_wd}")
            log_and_save("3_Verification", f"dim_{verify_dim}_reg_B{verify_bases}", row,
                         verify_dim, best_depth, verify_bases, verify_lr, verify_wd)

    # Extract winning configuration ONLY from Phase 3 Verification outputs
    phase3_runs = [r for r in results if scored(r) and r['phase'] in ['3_Verification', '3_Verification_Cached']]
    absolute_winner = max(phase3_runs or [r for r in results if scored(r)], key=lambda x: x['val_roc_auc'])

    print(f"\n=======================================================")
    print(f" OPTIMIZATION COMPLETE FOR {schema_label.upper()}")
    print(f" Best Phase 3 Score (mean over seeds): {absolute_winner['val_roc_auc']:.4f}")
    print(f" Optimal Settings: Dim={absolute_winner['hidden_dim']} | Depth={best_depth} | Bases={absolute_winner['num_bases']} | LR={absolute_winner['lr']} | WD={absolute_winner['weight_decay']}")
    print(f" Optimal Training Epochs (median best epoch + 1): {absolute_winner['best_epoch']+1}")
    print(f"=======================================================")

    cleanup()

    # ── PHASE 4: FINAL GENERALIZATION TEST ON HELD-OUT TEST SPLIT ───────────────
    # One retrain + test per seed; the spread across seeds is the run-to-run
    # variance any difference between KGs has to be judged against.
    for final_seed in final_seeds:
        final_row = evaluate_final_test_row(
            model_factory=model_factory, graph_data=graph_data,
            train_indices=train_indices, val_indices=val_indices, test_indices=test_indices,
            schema_label=schema_label, save_dir=save_dir,
            hidden_dim=absolute_winner['hidden_dim'], depth=best_depth,
            bases=absolute_winner['num_bases'],
            lr=absolute_winner['lr'], weight_decay=absolute_winner['weight_decay'],
            best_epoch=absolute_winner['best_epoch'], val_roc_auc=absolute_winner['val_roc_auc'],
            seed=final_seed,
        )
        results.append(final_row)
        pd.DataFrame(results).to_csv(csv_path, index=False)

    return pd.DataFrame(results)


def evaluate_final_test_row(
    model_factory, graph_data, train_indices, val_indices, test_indices,
    schema_label, save_dir,
    hidden_dim, depth, bases, lr, weight_decay, best_epoch, val_roc_auc=None, seed=0,
):
    """
    Trains the final model on train+val with a fixed (already-tuned) hyperparameter
    configuration and evaluates it on the held-out test split, returning one
    '4_Final_Test_Evaluation' results row with the full metric set (accuracy,
    precision, f1, roc_auc, pr_auc, plus the class-specific F1s). This is the
    Phase 4 step of run_sequential_optimization_with_verification, factored out
    so it can also be re-run on its own (see run_final_test_evaluation) without
    repeating Phases 1-3 of the search.
    """
    print(f"\n=======================================================")
    print(f" EXECUTING FINAL GENERALIZATION TEST SPLIT EVALUATION: {schema_label.upper()}")
    print(f"=======================================================")
    cleanup()

    combined_indices = torch.cat([train_indices, val_indices])
    hidden_list = [hidden_dim] * (depth - 1)
    set_seed(seed)
    final_model = model_factory(hidden_list, bases)

    final_trained_model = train_final_model(
        model=final_model, graph_data=graph_data, combined_indices=combined_indices,
        epochs=best_epoch + 1, lr=lr,
        weight_decay=weight_decay, save_path=f"{save_dir}{schema_label}_final_unified_eval_seed{seed}.pt"
    )

    test_metrics = evaluate_test_set(
        model=final_trained_model, graph_data=graph_data, test_indices=test_indices,
    )

    row_dict = {
        'phase': "4_Final_Test_Evaluation", 'parameter_tested': "unified_model", 'seed': seed,
        'val_roc_auc': val_roc_auc,
        'test_roc_auc': test_metrics['roc_auc'],
        'test_accuracy': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'test_pr_auc': test_metrics['pr_auc'],
        'test_class0_f1': test_metrics['class0_f1'],
        'test_class1_f1': test_metrics['class1_f1'],
        'num_parameters': test_metrics['num_parameters'],
        'runtime_sec': 0.0, 'status': "ok", 'hidden_dim': hidden_dim, 'layers': depth,
        'num_bases': bases, 'lr': lr, 'weight_decay': weight_decay,
        'best_epoch': best_epoch,
    }

    print(f"\n=======================================================")
    print(f" {schema_label.upper()} GENERALIZATION EVALUATION REPORT")
    print(f"=======================================================")
    print(f" * Test Accuracy  : {test_metrics['accuracy']:.4f}")
    print(f" * Test Precision : {test_metrics['precision']:.4f}")
    print(f" * Test Recall    : {test_metrics['recall']:.4f}")
    print(f" * Test F1-Score  : {test_metrics['f1']:.4f} (at fixed 0.5 threshold)")
    print(f" * Test ROC-AUC   : {test_metrics['roc_auc']:.4f}")
    print(f" * Test PR-AUC    : {test_metrics['pr_auc']:.4f}")
    print(f" * Test Class 0 F1: {test_metrics['class0_f1']:.4f}")
    print(f" * Test Class 1 F1: {test_metrics['class1_f1']:.4f}")
    print(f" * Num Parameters : {test_metrics['num_parameters']:,}")
    print(f"=======================================================")

    return row_dict


def run_final_test_evaluation(
    model_factory, graph_data, train_indices, val_indices, test_indices,
    schema_label, save_dir, csv_path,
    hidden_dim, depth, bases, lr, weight_decay, best_epoch, val_roc_auc=None, seeds=(0,),
):
    """
    Standalone re-run of just Phase 4 (final test evaluation) for an already-tuned
    configuration -- e.g. re-reading hidden_dim/depth/bases/lr/weight_decay
    /best_epoch off a previous run's '3_Verification' or '4_Final_Test_Evaluation'
    CSV row -- so the full Phase 1-3 search doesn't need to be repeated just to
    pick up new metric columns (test_accuracy/test_precision/test_f1/test_pr_auc).

    Appends the resulting row to csv_path, loading its existing rows first if the
    file already exists (so earlier Phase 1-3 rows in that file are preserved) and
    dropping any prior '4_Final_Test_Evaluation' row so re-running this doesn't
    leave two final-eval rows behind.
    """
    results = []
    if csv_path and os.path.exists(csv_path):
        results = [
            r for r in pd.read_csv(csv_path).to_dict('records')
            if r.get('phase') != '4_Final_Test_Evaluation'
        ]

    for seed in seeds:
        row_dict = evaluate_final_test_row(
            model_factory=model_factory, graph_data=graph_data,
            train_indices=train_indices, val_indices=val_indices, test_indices=test_indices,
            schema_label=schema_label, save_dir=save_dir,
            hidden_dim=hidden_dim, depth=depth, bases=bases,
            lr=lr, weight_decay=weight_decay, best_epoch=best_epoch, val_roc_auc=val_roc_auc, seed=seed,
        )
        results.append(row_dict)
        pd.DataFrame(results).to_csv(csv_path, index=False)

    return pd.DataFrame(results)


TEST_METRICS = ['test_roc_auc', 'test_pr_auc', 'test_f1', 'test_accuracy']
ABLATION_METRICS = ['val_roc_auc', *TEST_METRICS]


def run_ablation_variant(model_factory, kg_data, variant, hidden_list, bases, lr, wd,
                         epochs, patience, start_from_epoch, seeds, save_dir, csv_path):
    """
    Trains one enrichment-ablation graph at a fixed configuration, once per seed: early
    stopping on val after a warm-up of start_from_epoch epochs (run_one_config), then the
    best-val checkpoint is scored on test. Every variant gets the same configuration, so
    only the graph differs.

    Rows go to csv_path (shared by all variants) as each seed finishes. On a re-run, a
    (variant, seed) with an 'ok' row is skipped and failed rows are dropped and retried.
    Rows of another configuration (e.g. from before a re-tuning) raise; other constants are
    not checked: move the CSV away after changing one.
    """
    rows = pd.read_csv(csv_path).to_dict('records') if os.path.exists(csv_path) else []
    rows = [r for r in rows if r['status'] == 'ok']
    config = {'hidden_dim': hidden_list[0] if hidden_list else None, 'layers': len(hidden_list) + 1,
              'num_bases': bases, 'lr': lr, 'weight_decay': wd}
    if any(_config_of(r) != config for r in rows):
        raise ValueError(f"{csv_path} holds rows of another configuration than {config}; move it away")
    done = {(r['variant'], r['seed']) for r in rows}
    graph = kg_data.data

    for seed in seeds:
        if (variant, seed) in done:
            print(f"{variant} seed {seed}: already in {csv_path}")
            continue
        print(f"\n=== Ablation {variant} | seed {seed} ===")
        save_path = f"{save_dir}{variant}_seed{seed}.pt"
        run = run_one_config(model_factory, graph, kg_data.train, kg_data.val, hidden_list, bases, lr, wd,
                             epochs=epochs, patience=patience, save_path=save_path, seed=seed,
                             start_from_epoch=start_from_epoch)
        test_metrics = {}
        if run['status'] == 'ok':
            model = model_factory(hidden_list, bases)
            model.load_state_dict(torch.load(save_path, map_location='cpu', weights_only=False)['model_state_dict'])
            test_metrics = {f'test_{k}': v for k, v in evaluate_test_set(model, graph, kg_data.test).items()}
            del model
            cleanup()
        rows.append({
            'variant': variant, 'seed': seed, **run, **test_metrics,
            'hidden_dim': hidden_list[0] if hidden_list else None, 'layers': len(hidden_list) + 1,
            'num_nodes': graph.num_nodes, 'num_edges': graph.edge_index.size(1),
            'num_relations': kg_data.num_relations, 'num_feature_keys': len(kg_data.vocabulary),
        })
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    return pd.DataFrame(rows)


def summarize_ablation(results, controls, metrics=ABLATION_METRICS, delta_metrics=('val_roc_auc', 'test_roc_auc')):
    """
    Mean ± std of `metrics` per variant over its seeds, plus the difference of each of
    `delta_metrics`' means from its control's (controls: variant -> control variant; a control is
    its own control, so its deltas are 0). Variants keep the order of `controls`. The defaults are
    node prediction's; link prediction passes its MRR columns.
    """
    stats = results.groupby('variant')[list(metrics)].agg(['mean', 'std'])
    stats.columns = [f'{m}_{s}' for m, s in stats.columns]
    stats = stats.reindex(list(controls))
    stats.insert(0, 'control', [controls[v] for v in stats.index])
    stats.insert(1, 'n_seeds', results.groupby('variant').size().reindex(stats.index))
    for m in delta_metrics:
        stats[f'delta_{m}'] = stats[f'{m}_mean'] - stats[f'{m}_mean'].reindex(stats['control']).values
    return stats


TUNE_COLUMNS = ['val_roc_auc', 'val_roc_auc_std', 'val_roc_auc_seeds', 'best_epoch', 'best_epoch_seeds',
                'max_grad_norm', 'runtime_sec', 'status']
FINAL_COLUMNS = ['seed', 'val_roc_auc', 'best_epoch', 'test_roc_auc', 'test_accuracy', 'test_precision',
                 'test_recall', 'test_f1', 'test_pr_auc', 'test_class0_f1', 'test_class1_f1', 'num_parameters',
                 'status']


def _config_of(row):
    """A results row's (hidden_dim, layers, num_bases, lr, weight_decay), with NaN read as None."""
    optional_int = lambda value: None if pd.isna(value) else int(value)
    return {'hidden_dim': optional_int(row['hidden_dim']), 'layers': int(row['layers']),
            'num_bases': optional_int(row['num_bases']), 'lr': float(row['lr']),
            'weight_decay': float(row['weight_decay'])}


def tuned_winner(tuning_results, tuning_seeds, final_seeds, start_from_epoch=0):
    """
    A finished tuning run's winner as dicts: (config, phase-3 row, phase-4 rows). The config is
    the phase-4 rows' (hidden_dim, layers, num_bases, lr, weight_decay); the phase-3 row is the
    verification row of that config, which holds its per-seed val scores and best epochs.
    Raises if the run does not match the protocol it is reused under: phase-4 rows that
    disagree on the config, other final seeds, another number of tuning seeds, or a seed whose
    best epoch falls inside a warm-up of start_from_epoch epochs (the tuning had none; a seed
    that peaked after it is the same with or without it, see train_model).
    """
    final = tuning_results[tuning_results['phase'] == '4_Final_Test_Evaluation']
    final = final.astype({'seed': int}).to_dict('records')
    configs = [_config_of(r) for r in final]
    if not configs or any(c != configs[0] for c in configs):
        raise ValueError("the phase-4 rows are missing or disagree on the winning configuration")
    if sorted(r['seed'] for r in final) != sorted(final_seeds):
        raise ValueError(f"phase-4 seeds {sorted(r['seed'] for r in final)} != final_seeds {sorted(final_seeds)}")
    config = configs[0]

    phase3 = tuning_results[tuning_results['phase'].isin(['3_Verification', '3_Verification_Cached'])
                            & (tuning_results['status'] == 'ok')]
    matches = [r for r in phase3.to_dict('records') if _config_of(r) == config]
    if not matches:
        raise ValueError(f"no phase-3 row with the winning configuration {config}")
    tune = matches[0]
    if len(str(tune['val_roc_auc_seeds']).split(';')) != len(tuning_seeds):
        raise ValueError(f"the winner was tuned with seeds '{tune['val_roc_auc_seeds']}', not {len(tuning_seeds)}")
    if any(int(e) < start_from_epoch for e in str(tune['best_epoch_seeds']).split(';')):
        raise ValueError(f"a winner seed peaked inside the {start_from_epoch}-epoch warm-up "
                         f"(best epochs {tune['best_epoch_seeds']}), so it would change under it")
    return config, tune, final


def run_k_sweep(model_factory, kg_data, kg, tuning_results, k_values, epochs, patience, start_from_epoch,
                tuning_seeds, final_seeds, save_dir, csv_path):
    """
    Message-passing depth sweep for one KG at its own tuned configuration: only the number
    of layers K changes (hidden_list = [hidden_dim] * (K - 1); K = 1 has no hidden layer).
    Each K follows tuning phases 3-4: run_config_over_seeds on tuning_seeds (val early
    stopping after a warm-up of start_from_epoch epochs, mean val ROC-AUC, median best epoch),
    then evaluate_final_test_row per final seed (retrain on train+val for best_epoch + 1
    epochs, test). The winner's own depth is that computation already (tuned_winner checks
    that the warm-up does not change it), so its phase-3/4 rows are copied from
    tuning_results (reused=True) instead of re-run.

    Rows (kg, K, phase 'tune' | 'final', ...) go to csv_path, shared by all KGs of a model,
    after each K. On a re-run, a (kg, K) with an 'ok' tune row and every final seed is
    skipped; a partial or failed one is dropped and re-run. Rows of this KG with another
    configuration (e.g. from before a re-tuning) raise; other constants are not checked: move
    the CSV away after changing one. A failed (e.g. OOM) K logs its tune row only.
    """
    config, tuned, tuned_final = tuned_winner(tuning_results, tuning_seeds, final_seeds, start_from_epoch)
    if config['hidden_dim'] is None and any(k > 1 for k in k_values):
        raise ValueError(f"{kg}'s winner has one layer, so there is no hidden dim to deepen")
    rows = pd.read_csv(csv_path).to_dict('records') if os.path.exists(csv_path) else []
    at_depth = lambda k: dict(config, layers=k, hidden_dim=config['hidden_dim'] if k > 1 else None)
    if any(r['kg'] == kg and _config_of(r) != at_depth(int(r['K'])) for r in rows):
        raise ValueError(f"{csv_path} holds {kg} rows of another configuration than {config}; move it away")

    def finished(k):
        mine = [r for r in rows if r['kg'] == kg and r['K'] == k]
        return (any(r['phase'] == 'tune' and r['status'] == 'ok' for r in mine)
                and sorted(int(r['seed']) for r in mine if r['phase'] == 'final') == sorted(final_seeds))

    for k in k_values:
        if finished(k):
            print(f"{kg} K={k}: already in {csv_path}")
            continue
        rows = [r for r in rows if not (r['kg'] == kg and r['K'] == k)]
        hidden_list = [config['hidden_dim']] * (k - 1)
        dim = hidden_list[0] if hidden_list else None
        shared = {'kg': kg, 'K': k, **at_depth(k), 'reused': k == config['layers']}
        print(f"\n=== K-sweep {kg} | K={k}{' (tuned depth, reused)' if shared['reused'] else ''} ===")

        if shared['reused']:
            tune, finals = tuned, tuned_final
        else:
            tune = run_config_over_seeds(
                model_factory, kg_data.data, kg_data.train, kg_data.val, hidden_list, config['num_bases'],
                config['lr'], config['weight_decay'], epochs=epochs, patience=patience,
                save_path=f"{save_dir}tuning/{kg}_K{k}.pt", seeds=tuning_seeds, start_from_epoch=start_from_epoch,
            )
            finals = [] if tune['status'] != 'ok' else [
                evaluate_final_test_row(
                    model_factory=model_factory, graph_data=kg_data.data, train_indices=kg_data.train,
                    val_indices=kg_data.val, test_indices=kg_data.test, schema_label=f"{kg}_K{k}",
                    save_dir=save_dir, hidden_dim=dim, depth=k, bases=config['num_bases'], lr=config['lr'],
                    weight_decay=config['weight_decay'], best_epoch=tune['best_epoch'],
                    val_roc_auc=tune['val_roc_auc'], seed=seed,
                )
                for seed in final_seeds
            ]

        capped = tune['status'] == 'ok' and _ended_by_epoch_cap(tune['best_epoch_seeds'], epochs, patience)
        rows.append({**shared, 'phase': 'tune', **{c: tune[c] for c in TUNE_COLUMNS}, 'ended_by_epoch_cap': capped})
        rows += [{**shared, 'phase': 'final', **{c: f[c] for c in FINAL_COLUMNS}} for f in finals]
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        cleanup()

    return pd.DataFrame(rows)


def summarize_k_sweep(results):
    """
    One row per (kg, K): the tune row's mean val ROC-AUC over the tuning seeds and median
    best epoch, then mean ± std of the test metrics over the final seeds.
    """
    tune = results[results['phase'] == 'tune'].set_index(['kg', 'K'])
    final = results[results['phase'] == 'final'].groupby(['kg', 'K'])
    stats = final[TEST_METRICS].agg(['mean', 'std'])
    stats.columns = [f'{m}_{s}' for m, s in stats.columns]
    summary = tune[['val_roc_auc', 'val_roc_auc_std', 'best_epoch', 'reused', 'ended_by_epoch_cap']].join(stats)
    summary.insert(2, 'n_seeds', final.size())
    summary['num_parameters'] = final['num_parameters'].first()
    return summary.sort_index()
