import os
import sys

repo_path = '/Workspace/Users/dversch@emea.corpdir.net/.bundle/KG_Model/prod/files'
if repo_path not in sys.path:
    sys.path.append(repo_path)

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
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import degree

def cleanup():
    """Release GPU memory after each config, success or failure -- prevents
    fragmentation-related OOMs that look random across a long sweep."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_relation_dict(ont):
    return getattr(ont, 'enum_relations', {})

def analyze_snr_dilution(graph_data, target_indices, usage_counts, num_neighbors, batch_size=256, max_batches=10):
    """
    Measures the Signal-to-Noise Ratio (SNR) in the model's local neighborhoods
    by computing the ratio of unique text descriptor nodes to categorical fallback nodes.
    """
    # Use 0 workers for stable memory management during rapid sequential evaluations
    loader = NeighborLoader(
        graph_data, 
        num_neighbors=list(num_neighbors), 
        batch_size=batch_size,
        input_nodes=target_indices, 
        shuffle=False, 
        num_workers=0
    )
    
    all_text_pcts = []
    all_fallback_pcts = []
    all_sizes = []
    
    try:
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            
            # Map batch nodes back to their global feature-row usage profile
            batch_usages = usage_counts[batch.n_id_to_vec_row]
            num_b_nodes = batch_usages.size(0)
            
            # Count high-signal unique nodes (usage count == 1) vs. categorical fallback (usage > 1)
            text_nodes_count = (batch_usages == 1).sum().item()
            fallback_nodes_count = (batch_usages > 1).sum().item()
            
            all_sizes.append(num_b_nodes)
            all_text_pcts.append((text_nodes_count / num_b_nodes) * 100.0)
            all_fallback_pcts.append((fallback_nodes_count / num_b_nodes) * 100.0)
            
        return {
            'avg_neighborhood_size': float(np.mean(all_sizes)),
            'high_signal_text_pct': float(np.mean(all_text_pcts)),
            'fallback_noise_pct': float(np.mean(all_fallback_pcts))
        }
    finally:
        # Prevent PyG multiprocessing loaders from lingering in memory
        del loader
        cleanup()

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

def parse_neighbors(neigh_str):
    nums = re.findall(r'\d+', neigh_str)
    return tuple(int(x) for x in nums)

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

def train_final_model(model, graph_data, combined_indices, vector_table,
                      epochs, batch_size=256, num_neighbors=(25, 10), lr=0.001,
                      weight_decay=1e-4, save_path=None):
    """
    Trains the final selected model on train+validation combined indices.
    Bypasses validation loader generation, ensuring memory safety.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    
    train_loader = NeighborLoader(
        graph_data, num_neighbors=list(num_neighbors), batch_size=batch_size,
        input_nodes=combined_indices, num_workers=0, shuffle=True,
    )
    
    try:
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                edge_index = batch.edge_index.contiguous()
                edge_type = batch.edge_type.contiguous()
                
                optimizer.zero_grad()
                predictions = model(batch.n_id_to_vec_row, edge_index, edge_type, batch.degree_feat)
                
                batch_targets = batch.y[:batch.batch_size].float()
                batch_preds = predictions[:batch.batch_size].squeeze(1)
                
                loss = loss_fn(batch_preds, batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            print(f"Final Train Epoch {epoch+0:03d} | Train Loss: {avg_train_loss:.4f}")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
            }, save_path)
            print(f"Final trained model saved to {save_path}")
            
        return model
    finally:
        model.cpu()
        if 'train_loader' in locals():
            del train_loader
        if 'optimizer' in locals():
            del optimizer
        cleanup()

def evaluate_test_set(model, graph_data, test_indices, batch_size=256, num_neighbors=(25, 10)):
    """
    Evaluates the finalized model on the unseen test split using a fixed 0.5 threshold.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    test_loader = NeighborLoader(
        graph_data, num_neighbors=list(num_neighbors), batch_size=batch_size,
        input_nodes=test_indices, num_workers=0, shuffle=False,
    )
    
    all_probs, all_targets = [], []
    try:
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                edge_index = batch.edge_index.contiguous()
                edge_type = batch.edge_type.contiguous()
                logits = model(batch.n_id_to_vec_row, edge_index, edge_type, batch.degree_feat)[:batch.batch_size].squeeze(1)
                targets = batch.y[:batch.batch_size].float()
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        probs = np.array(all_probs)
        targets = np.array(all_targets)
        preds = (probs >= 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(targets, preds),
            'precision': precision_score(targets, preds, zero_division=0),
            'recall': recall_score(targets, preds, zero_division=0),
            'f1': f1_score(targets, preds, zero_division=0),
            'roc_auc': roc_auc_score(targets, probs),
            'pr_auc': average_precision_score(targets, probs),
            'num_parameters': int(sum(p.numel() for p in model.parameters())),
        }
        metrics.update(compute_class_specific_metrics(targets, probs))
        return metrics
    finally:
        model.cpu()
        if 'test_loader' in locals():
            del test_loader
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


def avg_children_per_case(graph_data, case_mask, child_mask, num_cases):
    """Average number of `child`-typed instances directly attached to each Case."""
    src, dst = graph_data.edge_index
    direct_edges = case_mask[src] & child_mask[dst]
    return direct_edges.sum().item() / num_cases if num_cases > 0 else float('nan')


def train_model(model, graph_data, train_indices, val_indices, vector_table,
                 epochs=100, batch_size=64, num_neighbors=(10, 10, 5), lr=0.001,
                 weight_decay=5e-4, eval_frequency=1,
                 save_path=None, resume=False, patience=10, monitor='roc_auc'):
    """
    Model-agnostic training loop. `model` must accept
    forward(node_row_ids, edge_index, edge_type, degree_feat) -- for
    architectures with no relation typing (e.g. GraphSAGE), forward should
    accept and simply ignore edge_type so this loop never needs to branch
    on model type.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    start_epoch = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'val_accuracy': [], 'val_precision': [], 'val_recall': [],
        'val_f1': [], 'val_roc_auc': [], 'val_pr_auc': [],
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

    # num_workers=0 required: forked loader workers duplicate host RAM via
    # copy-on-write and can trigger an uncatchable SIGKILL on a single-node
    # cluster at deeper K/larger fanouts.
    train_loader = NeighborLoader(
        graph_data, num_neighbors=list(num_neighbors), batch_size=batch_size,
        input_nodes=train_indices, num_workers=0, shuffle=True,
    )
    val_loader = NeighborLoader(
        graph_data, num_neighbors=list(num_neighbors), batch_size=batch_size,
        input_nodes=val_indices, num_workers=0, shuffle=False,
    )

    try:
        def run_eval(loader):
            model.eval()
            all_probs, all_targets, total_loss, total_samples = [], [], 0, 0
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    edge_index = batch.edge_index.contiguous()
                    edge_type = batch.edge_type.contiguous()
                    logits = model(batch.n_id_to_vec_row, edge_index, edge_type, batch.degree_feat)[:batch.batch_size].squeeze(1)
                    targets = batch.y[:batch.batch_size].float()
                    total_loss += loss_fn(logits, targets).item() * batch.batch_size
                    total_samples += batch.batch_size
                    all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

            probs = np.array(all_probs)
            targets = np.array(all_targets)
            preds = (probs >= 0.5).astype(int)

            try:
                metrics = {
                    'loss': total_loss / total_samples,
                    'accuracy': accuracy_score(targets, preds),
                    'precision': precision_score(targets, preds, zero_division=0),
                    'recall': recall_score(targets, preds, zero_division=0),
                    'f1': f1_score(targets, preds, zero_division=0),
                    'roc_auc': roc_auc_score(targets, probs),
                    'pr_auc': average_precision_score(targets, probs),
                }
            except ValueError:
                metrics = {'loss': total_loss / total_samples, 'accuracy': accuracy_score(targets, preds),
                           'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'roc_auc': 0.0, 'pr_auc': 0.0}
            return metrics

        for epoch in range(start_epoch, epochs):
            model.train()
            total_loss = 0

            for batch in train_loader:
                batch = batch.to(device)
                edge_index = batch.edge_index.contiguous()
                edge_type = batch.edge_type.contiguous()

                optimizer.zero_grad()
                predictions = model(batch.n_id_to_vec_row, edge_index, edge_type, batch.degree_feat)

                batch_targets = batch.y[:batch.batch_size].float()
                batch_preds = predictions[:batch.batch_size].squeeze(1)

                loss = loss_fn(batch_preds, batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)

            if (epoch + 1) % eval_frequency == 0 or epoch == epochs - 1:
                val_metrics = run_eval(val_loader)
                history['val_loss'].append(val_metrics['loss'])
                history['val_accuracy'].append(val_metrics['accuracy'])
                history['val_precision'].append(val_metrics['precision'])
                history['val_recall'].append(val_metrics['recall'])
                history['val_f1'].append(val_metrics['f1'])
                history['val_roc_auc'].append(val_metrics['roc_auc'])
                history['val_pr_auc'].append(val_metrics['pr_auc'])

                print(f"Epoch {epoch+0:03d} | Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
                      f"Val F1: {val_metrics['f1']:.4f} | Val ROC-AUC: {val_metrics['roc_auc']:.4f}")

                current_metric = val_metrics[monitor]

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
                        best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
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
        if 'train_loader' in locals():
            del train_loader
        if 'val_loader' in locals():
            del val_loader
        if 'optimizer' in locals():
            del optimizer
        cleanup()


def evaluate_ablation_configuration(model, graph_data, train_indices, val_indices, test_indices,
                                    vector_table, usage_counts, neighbors, lr, wd, epochs=50, patience=15,
                                    batch_size=256):
    """
    Executes a single ablation run. It trains the target ablated graph,
    tracks runtime and early-stopping complexity, computes class-specific metrics,
    PR-AUC, and runs the neighborhood SNR diagnostic.
    """
    start_wall_time = time.time()

    trained_model, history = train_model(
        model=model, graph_data=graph_data,
        train_indices=train_indices, val_indices=val_indices,
        vector_table=vector_table, epochs=epochs, batch_size=batch_size,
        num_neighbors=neighbors, lr=lr, weight_decay=wd,
        save_path=None, resume=False, patience=patience, monitor='roc_auc'
    )

    wall_runtime = time.time() - start_wall_time
    best_epoch = history.get('best_epoch', 0)
    epochs_run = len(history.get('train_loss', []))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trained_model = trained_model.to(device)
    trained_model.eval()

    eval_loader = NeighborLoader(
        graph_data, num_neighbors=list(neighbors), batch_size=batch_size,
        input_nodes=test_indices, num_workers=0, shuffle=False
    )

    all_probs, all_targets = [], []
    try:
        with torch.no_grad():
            for batch in eval_loader:
                batch = batch.to(device)
                edge_index = batch.edge_index.contiguous()
                edge_type = batch.edge_type.contiguous()
                logits = trained_model(batch.n_id_to_vec_row, edge_index, edge_type, batch.degree_feat)[:batch.batch_size].squeeze(1)
                targets = batch.y[:batch.batch_size].float()
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
    finally:
        del eval_loader
        cleanup()

    # train_model returns a CPU model, but the .to(device) above for eval put
    # it back on GPU -- move back explicitly so loop-callers (e.g. run_k_sweep)
    # don't accumulate GPU memory across iterations.
    trained_model.cpu()
    del trained_model
    cleanup()

    targets_arr = np.array(all_targets)
    probs_arr = np.array(all_probs)

    class_metrics = compute_class_specific_metrics(targets_arr, probs_arr)
    snr_metrics = analyze_snr_dilution(graph_data, test_indices, usage_counts, neighbors)

    global_roc_auc = roc_auc_score(targets_arr, probs_arr)
    global_f1 = f1_score(targets_arr, (probs_arr >= 0.5).astype(int), zero_division=0)
    global_pr_auc = average_precision_score(targets_arr, probs_arr)

    results = {
        'test_roc_auc': global_roc_auc,
        'test_f1_score': global_f1,
        'test_pr_auc': global_pr_auc,
        'training_time_sec': wall_runtime,
        'best_epoch': best_epoch,
        'epochs_run': epochs_run,
        **class_metrics,
        **snr_metrics
    }

    return results


def run_ablation_run(name, test_data, pct_removed, model_factory, baseline_test_auc, baseline_test_f1,
                      graph_data, schema_label, train_indices, val_indices, test_indices,
                      vector_table, usage_counts, neighbors, lr, wd,
                      epochs=50, patience=15, batch_sizes=(256, 128)):
    """
    Trains+evaluates one ablation configuration, returns its result record.
    Stateless -- no side effects beyond training the model it's given.

    model_factory: zero-arg callable returning a fresh model (architecture
        specifics already bound via closure by the caller).
    baseline_test_auc/f1: caller-owned running baseline. Pass None before
        the "Baseline Control" run; pass through the previous return value
        for every other call.

    Returns (record, updated_baseline_test_auc, updated_baseline_test_f1).
    """
    start_time = time.time()

    baseline_edge_count = graph_data.edge_index.size(1)
    ablated_edge_count = test_data.edge_index.size(1)
    edges_removed = baseline_edge_count - ablated_edge_count

    # A node counts as "removed" if it had edges in the full graph but none
    # left after ablation. Degree is summed over both edge_index rows since
    # every triple is doubled into a forward+inverse pair.
    baseline_deg = (degree(graph_data.edge_index[0], num_nodes=graph_data.num_nodes) +
                     degree(graph_data.edge_index[1], num_nodes=graph_data.num_nodes))
    ablated_deg = (degree(test_data.edge_index[0], num_nodes=graph_data.num_nodes) +
                    degree(test_data.edge_index[1], num_nodes=graph_data.num_nodes))
    isolated_mask = (baseline_deg > 0) & (ablated_deg == 0)
    nodes_removed = int(isolated_mask.sum().item())
    baseline_node_count = int((baseline_deg > 0).sum().item())
    pct_nodes_removed = (nodes_removed / baseline_node_count * 100.0) if baseline_node_count > 0 else 0.0

    schema_name = 'SULO' if schema_label == 'sulo' else 'Baseline'
    print(f"\n[EVALUATING] Masking: {name} (Pruned {edges_removed} edges, {pct_removed:.2f}% of graph; "
          f"{nodes_removed} nodes isolated, {pct_nodes_removed:.2f}% of graph)")

    record = {
        'ablation': name,
        'schema': schema_name,
        'edges_removed': int(edges_removed),
        'pct_edges_removed': float(pct_removed),
        'nodes_removed': nodes_removed,
        'pct_nodes_removed': float(pct_nodes_removed),
        'ROC_AUC': np.nan,
        'F1': np.nan,
        'PR_AUC': np.nan,
        'delta_ROC_AUC': np.nan,
        'delta_F1': np.nan,
        'RID': np.nan,
        'class0_F1': np.nan,
        'class1_F1': np.nan,
        'avg_neighborhood_size': np.nan,
        'high_signal_text_pct': np.nan,
        'fallback_noise_pct': np.nan,
        'runtime_sec': 0.0,
        'best_epoch': 0,
        'epochs_run': 0,
        'status': 'pending'
    }

    try:
        # On CUDA OOM, retry at a smaller batch size instead of skipping --
        # some concept masks concentrate sampling onto remaining hub nodes,
        # ballooning the induced subgraph even with fewer edges removed.
        for i, attempt_batch_size in enumerate(batch_sizes):
            model = model_factory()
            try:
                metrics = evaluate_ablation_configuration(
                    model=model, graph_data=test_data,
                    train_indices=train_indices, val_indices=val_indices, test_indices=test_indices,
                    vector_table=vector_table, usage_counts=usage_counts, neighbors=neighbors,
                    lr=lr, wd=wd, epochs=epochs, patience=patience,
                    batch_size=attempt_batch_size,
                )
                break
            except RuntimeError as oom_err:
                if "out of memory" not in str(oom_err).lower() or i == len(batch_sizes) - 1:
                    raise
                print(f"  -> CUDA OOM at batch_size={attempt_batch_size}, retrying at batch_size={batch_sizes[i + 1]}...")
                # del before cleanup() -- the exception's traceback keeps the
                # failed GPU-resident model alive until it's dropped.
                del oom_err, model
                cleanup()

        # "Baseline Control" is the reference point every other ablation's
        # delta/RID is measured against.
        if name == "Baseline Control":
            baseline_test_auc = metrics['test_roc_auc']
            baseline_test_f1 = metrics['test_f1_score']

        # Absolute effect: M_ablated - M_baseline
        delta_roc_auc = metrics['test_roc_auc'] - baseline_test_auc
        delta_f1 = metrics['test_f1_score'] - baseline_test_f1

        # RID: (Baseline - Ablated) / % edges removed -- positive RID means
        # removing this concept hurt performance (high-signal, low-density
        # relation). Note the sign is opposite delta_roc_auc.
        rid = -delta_roc_auc / pct_removed if pct_removed > 0.0 else 0.0

        record.update({
            'ROC_AUC': float(metrics['test_roc_auc']),
            'F1': float(metrics['test_f1_score']),
            'PR_AUC': float(metrics['test_pr_auc']),
            'delta_ROC_AUC': float(delta_roc_auc),
            'delta_F1': float(delta_f1),
            'RID': float(rid),
            'class0_F1': float(metrics['class0_f1']),
            'class1_F1': float(metrics['class1_f1']),
            'avg_neighborhood_size': float(metrics['avg_neighborhood_size']),
            'high_signal_text_pct': float(metrics['high_signal_text_pct']),
            'fallback_noise_pct': float(metrics['fallback_noise_pct']),
            'runtime_sec': float(metrics['training_time_sec']),
            'best_epoch': int(metrics['best_epoch']),
            'epochs_run': int(metrics['epochs_run']),
            'status': 'completed'
        })

        print(f"  -> ROC-AUC: {metrics['test_roc_auc']:.4f} (Delta: {delta_roc_auc:.4f}) | F1: {metrics['test_f1_score']:.4f} (Delta: {delta_f1:.4f}) | RID: {rid:.4f}")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("  -> CUDA OOM: Gracefully skipping configuration.")
            record.update({
                'status': 'oom',
                'runtime_sec': float(time.time() - start_time)
            })
            del e  # drop the traceback so cleanup() can actually reclaim GPU memory
            cleanup()
        else:
            record.update({'status': 'failed'})
            raise e

    return record, baseline_test_auc, baseline_test_f1


def run_one_config(model_factory, graph_data, train_indices, val_indices, vector_table,
                    hidden_list, bases, neighbors, lr, wd, epochs, patience, save_path):
    """
    Memory-safe single hyperparameter-configuration executor, returning
    val_roc_auc/best_epoch/status for use by a sweep.

    model_factory(hidden_list, bases) -> nn.Module: builds a fresh model for
        this configuration. All other architecture-specific construction
        (vector_table, categorical ids, num_relations, FEATURE_DIM, ...)
        must already be bound via closure by the caller. Pass bases=None
        for architectures with no relation-basis concept (e.g. GraphSAGE)
        -- the factory just ignores it.
    """
    start_time = time.time()

    model = model_factory(hidden_list, bases)

    history = None
    peak_auc = None
    best_epoch = None
    status = 'ok'

    try:
        gpu_model, history = train_model(
            model=model, graph_data=graph_data,
            train_indices=train_indices, val_indices=val_indices,
            vector_table=vector_table, epochs=epochs, batch_size=256,
            num_neighbors=neighbors, lr=lr, weight_decay=wd,
            save_path=save_path, resume=False, patience=patience, monitor='roc_auc',
        )
        auc_values = [
            x for x in history.get("val_roc_auc", [])
            if x is not None and not np.isnan(x)
        ]
        peak_auc = max(auc_values) if auc_values else None

        best_epoch = history.get('best_epoch', None)
        elapsed = time.time() - start_time
        print(f"  -> Finished in {elapsed:.1f}s | Peak ROC-AUC: {peak_auc:.4f} (Best Epoch: {best_epoch})")
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
        'neighbors': str(neighbors), 'val_roc_auc': peak_auc,
        'best_epoch': best_epoch,
        'runtime_sec': elapsed, 'status': status,
    }


def run_sequential_optimization_with_verification(
    model_factory, graph_data, train_indices, val_indices, test_indices,
    vector_table, schema_label, save_dir, csv_path,
    grid_dim, grid_depth, grid_bases, grid_lr, grid_wd,
    default_bases, default_lr, default_wd,
    epochs, patience,
):
    """
    Sequential/greedy hyperparameter search: dim -> depth/neighbors -> bases
    -> lr -> wd, then a 2x2 verification grid over the top-2 dims x top-2
    (bases, lr, wd) combos, then a final held-out test evaluation of the
    overall winner. Returns the full results log as a DataFrame.

    model_factory(hidden_list, bases) -> nn.Module: builds a fresh model.
    All other architecture-specific construction (vector_table, categorical
    ids, num_relations, FEATURE_DIM, ...) must already be bound via closure
    by the caller. For architectures with no relation-basis concept (e.g.
    GraphSAGE), pass grid_bases=[None] and default_bases=None -- the
    factory just ignores the bases argument it's given.

    save_dir: root directory for this model type's checkpoints (e.g.
        f"{volume}rgcn/") -- per-phase checkpoints go in
        f"{save_dir}robust_tuning_logs/", the final unified-eval checkpoint
        goes directly under save_dir.
    """
    cleanup()

    results = []
    phase_1a_results = {}
    phase_2_results = {}

    def log_and_save(phase, parameter, val_auc, elapsed, status, d, l, n, b, lr, wd, best_epoch):
        row_dict = {
            'phase': phase, 'parameter_tested': parameter,
            'val_roc_auc': val_auc,
            'runtime_sec': elapsed, 'status': status, 'hidden_dim': d, 'layers': l,
            'neighbors': str(n), 'num_bases': b, 'lr': lr, 'weight_decay': wd,
            'best_epoch': best_epoch
        }
        results.append(row_dict)
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"  -> [{phase}] Metric: {val_auc if val_auc else 'OOM/FAIL':.4f} | Best Epoch: {best_epoch} | Status: {status}")
        return row_dict

    print(f"\n=======================================================")
    print(f" STARTING SEQUENTIAL OPTIMIZATION: {schema_label.upper()}")
    print(f"=======================================================")

    # ── PHASE 1A: HIDDEN DIMENSION SWEEP ─────────────────────────────────────
    print("\n>>> Phase 1a: Tuning Hidden Dimension (Depth=2, Neighbors=(25, 10))")
    best_dim = 64
    best_dim_auc = -float('inf')

    for dim in grid_dim:
        print(f" [RUNNING] Hidden Dim: {dim}")
        row = run_one_config(
            model_factory, graph_data, train_indices, val_indices, vector_table,
            hidden_list=[dim], bases=default_bases, neighbors=(25, 10), lr=default_lr, wd=default_wd,
            epochs=epochs, patience=patience,
            save_path=f"{save_dir}robust_tuning_logs/{schema_label}_1a_dim{dim}.pt"
        )
        log_and_save("1a_HiddenDim", f"dim_{dim}", row['val_roc_auc'], row['runtime_sec'], row['status'],
                     dim, 2, (25, 10), default_bases, default_lr, default_wd, row['best_epoch'])

        if row['status'] == 'ok' and row['val_roc_auc'] is not None:
            phase_1a_results[dim] = row['val_roc_auc']
            if row['val_roc_auc'] > best_dim_auc:
                best_dim_auc = row['val_roc_auc']
                best_dim = dim

    sorted_dims = sorted(phase_1a_results.items(), key=lambda x: x[1], reverse=True)
    top_dims = [dim for dim, auc in sorted_dims[:2]]
    if len(top_dims) == 1:
        top_dims.append(top_dims[0])

    print(f"\n>>> Phase 1a Top Dimensions isolated for 2x2 validation: {top_dims}")
    cleanup()

    # ── PHASE 1B: DEPTH SWEEP ────────────────────────────────────────────────
    print(f"\n>>> Phase 1b: Tuning Model Depth & Sampling (hidden_dim locked at {best_dim})")
    best_depth = 2
    best_neighbors = (25, 10)
    best_depth_auc = -float('inf')

    best_dim_row = next((r for r in results if r['hidden_dim'] == best_dim and r['phase'] == '1a_HiddenDim'), None)
    best_dim_epoch = best_dim_row['best_epoch'] if best_dim_row else None

    for layers, neighbors in grid_depth:
        if layers == 2 and neighbors == (25, 10):
            print(f" [SKIPPING] Reusing 2-Layer result from Phase 1a.")
            log_and_save("1b_Depth", f"layers_{layers}", best_dim_auc, 0.0, "ok",
                         best_dim, layers, neighbors, default_bases, default_lr, default_wd, best_dim_epoch)
            if best_dim_auc > best_depth_auc:
                best_depth_auc = best_dim_auc
                best_depth, best_neighbors = layers, neighbors
            continue

        print(f" [RUNNING] Depth: {layers} | Neighbors: {neighbors}")
        hidden_list = [best_dim] * (layers - 1)
        row = run_one_config(
            model_factory, graph_data, train_indices, val_indices, vector_table,
            hidden_list, bases=default_bases, neighbors=neighbors, lr=default_lr, wd=default_wd,
            epochs=epochs, patience=patience,
            save_path=f"{save_dir}robust_tuning_logs/{schema_label}_1b_L{layers}.pt"
        )
        log_and_save("1b_Depth", f"layers_{layers}", row['val_roc_auc'], row['runtime_sec'], row['status'],
                     best_dim, layers, neighbors, default_bases, default_lr, default_wd, row['best_epoch'])

        if row['status'] == 'ok' and row['val_roc_auc'] is not None and row['val_roc_auc'] > best_depth_auc:
            best_depth_auc = row['val_roc_auc']
            best_depth, best_neighbors = layers, neighbors

    best_depth_row = next((r for r in results if r['layers'] == best_depth and r['neighbors'] == str(best_neighbors) and r['phase'] == '1b_Depth'), None)
    best_depth_epoch = best_depth_row['best_epoch'] if best_depth_row else None

    cleanup()

    # ── PHASE 2A: RELATIONAL REGULARIZATION (BASES) SWEEP ────────────────────
    print(f"\n>>> Phase 2a: Tuning Bases (Architecture locked to L={best_depth}, H={best_dim})")
    best_bases = default_bases
    best_bases_auc = -float('inf')
    hidden_list = [best_dim] * (best_depth - 1)

    for bases in grid_bases:
        if bases == default_bases:
            print(f" [SKIPPING] Reusing default Base state from Phase 1b.")
            log_and_save("2a_Bases", f"bases_{bases}", best_depth_auc, 0.0, "ok",
                         best_dim, best_depth, best_neighbors, bases, default_lr, default_wd, best_depth_epoch)
            if best_depth_auc > best_bases_auc:
                best_bases_auc = best_depth_auc
                best_bases = bases
            continue

        print(f" [RUNNING] Bases: {bases}")
        row = run_one_config(
            model_factory, graph_data, train_indices, val_indices, vector_table,
            hidden_list, bases, best_neighbors, lr=default_lr, wd=default_wd,
            epochs=epochs, patience=patience,
            save_path=f"{save_dir}robust_tuning_logs/{schema_label}_2a_B{bases}.pt"
        )
        log_and_save("2a_Bases", f"bases_{bases}", row['val_roc_auc'], row['runtime_sec'], row['status'],
                     best_dim, best_depth, best_neighbors, bases, default_lr, default_wd, row['best_epoch'])

        if row['status'] == 'ok' and row['val_roc_auc'] is not None and row['val_roc_auc'] > best_bases_auc:
            best_bases_auc = row['val_roc_auc']
            best_bases = bases

    best_bases_row = next((r for r in results if r['num_bases'] == best_bases and r['phase'] == '2a_Bases'), None)
    best_bases_epoch = best_bases_row['best_epoch'] if best_bases_row else None

    cleanup()

    # ── PHASE 2B: LEARNING RATE SWEEP ────────────────────────────────────────
    print(f"\n>>> Phase 2b: Tuning LR (Bases locked at {best_bases})")
    best_lr = default_lr
    best_lr_auc = -float('inf')

    for lr in grid_lr:
        if lr == default_lr:
            print(f" [SKIPPING] Reusing current runner from Phase 2a.")
            log_and_save("2b_LR", f"lr_{lr}", best_bases_auc, 0.0, "ok",
                         best_dim, best_depth, best_neighbors, best_bases, lr, default_wd, best_bases_epoch)
            if best_bases_auc > best_lr_auc:
                best_lr_auc = best_bases_auc
                best_lr = lr
            continue

        print(f" [RUNNING] Learning Rate: {lr}")
        row = run_one_config(
            model_factory, graph_data, train_indices, val_indices, vector_table,
            hidden_list, best_bases, best_neighbors, lr, wd=default_wd,
            epochs=epochs, patience=patience,
            save_path=f"{save_dir}robust_tuning_logs/{schema_label}_2b_lr{lr}.pt"
        )
        log_and_save("2b_LR", f"lr_{lr}", row['val_roc_auc'], row['runtime_sec'], row['status'],
                     best_dim, best_depth, best_neighbors, best_bases, lr, default_wd, row['best_epoch'])

        if row['status'] == 'ok' and row['val_roc_auc'] is not None and row['val_roc_auc'] > best_lr_auc:
            best_lr_auc = row['val_roc_auc']
            best_lr = lr

    best_lr_row = next((r for r in results if r['lr'] == best_lr and r['phase'] == '2b_LR'), None)
    best_lr_epoch = best_lr_row['best_epoch'] if best_lr_row else None

    cleanup()

    # ── PHASE 2C: WEIGHT DECAY SWEEP ─────────────────────────────────────────
    print(f"\n>>> Phase 2c: Tuning Weight Decay (Bases locked to {best_bases}, LR to {best_lr})")
    best_wd = default_wd
    best_wd_auc = -float('inf')

    for wd in grid_wd:
        if wd == default_wd:
            print(f" [SKIPPING] Reusing current runner from Phase 2b.")
            log_and_save("2c_WD", f"wd_{wd}", best_lr_auc, 0.0, "ok",
                         best_dim, best_depth, best_neighbors, best_bases, best_lr, wd, best_lr_epoch)
            phase_2_results[(best_bases, best_lr, wd)] = best_lr_auc
            if best_lr_auc > best_wd_auc:
                best_wd_auc = best_lr_auc
                best_wd = wd
            continue

        print(f" [RUNNING] Weight Decay: {wd}")
        row = run_one_config(
            model_factory, graph_data, train_indices, val_indices, vector_table,
            hidden_list, best_bases, best_neighbors, best_lr, wd,
            epochs=epochs, patience=patience,
            save_path=f"{save_dir}robust_tuning_logs/{schema_label}_2c_wd{wd}.pt"
        )
        log_and_save("2c_WD", f"wd_{wd}", row['val_roc_auc'], row['runtime_sec'], row['status'],
                     best_dim, best_depth, best_neighbors, best_bases, best_lr, wd, row['best_epoch'])

        if row['status'] == 'ok' and row['val_roc_auc'] is not None:
            phase_2_results[(best_bases, best_lr, wd)] = row['val_roc_auc']
            if row['val_roc_auc'] > best_wd_auc:
                best_wd_auc = row['val_roc_auc']
                best_wd = wd

    sorted_reg = sorted(phase_2_results.items(), key=lambda x: x[1], reverse=True)
    top_reg = [params for params, auc in sorted_reg[:2]]
    if len(top_reg) == 1:
        top_reg.append(top_reg[0])
    print(f"\n>>> Phase 2 Regularization coordinates isolated for 2x2 validation: {top_reg}")

    cleanup()

    # ── PHASE 3: 2X2 VERIFICATION GRID ───────────────────────────────────────
    print(f"\n=======================================================")
    print(f" STARTING PHASE 3 (2X2 INTERACTION VERIFICATION): {schema_label.upper()}")
    print(f"=======================================================")

    for verify_dim in top_dims:
        for verify_bases, verify_lr, verify_wd in top_reg:

            match_found = False
            for r in results:
                if (r['hidden_dim'] == verify_dim and
                    r['num_bases'] == verify_bases and
                    r['lr'] == verify_lr and
                    r['weight_decay'] == verify_wd and
                    r['status'] == 'ok'):

                    print(f" [SKIPPING] Caching verified cross-state: Dim={verify_dim} | Bases={verify_bases} | LR={verify_lr} | WD={verify_wd}")
                    log_and_save("3_Verification_Cached", f"dim_{verify_dim}_reg_B{verify_bases}", r['val_roc_auc'], 0.0, "ok",
                                 verify_dim, best_depth, best_neighbors, verify_bases, verify_lr, verify_wd, r['best_epoch'])
                    match_found = True
                    break

            if match_found:
                continue

            print(f" [RUNNING Verification] Dim={verify_dim} | Bases={verify_bases} | LR={verify_lr} | WD={verify_wd}")
            verify_hidden_list = [verify_dim] * (best_depth - 1)
            row = run_one_config(
                model_factory, graph_data, train_indices, val_indices, vector_table,
                verify_hidden_list, verify_bases, best_neighbors, verify_lr, verify_wd,
                epochs=epochs, patience=patience,
                save_path=(
                    f"{save_dir}robust_tuning_logs/"
                    f"{schema_label}_"
                    f"verify_dim{verify_dim}_"
                    f"B{verify_bases}_"
                    f"lr{verify_lr}_"
                    f"wd{verify_wd}.pt"
                )
            )
            log_and_save("3_Verification", f"dim_{verify_dim}_reg_B{verify_bases}", row['val_roc_auc'], row['runtime_sec'], row['status'],
                         verify_dim, best_depth, best_neighbors, verify_bases, verify_lr, verify_wd, row['best_epoch'])

    # Extract winning configuration ONLY from Phase 3 Verification outputs
    phase3_runs = [r for r in results if r['status'] == 'ok' and r['phase'] in ['3_Verification', '3_Verification_Cached']]
    absolute_winner = max(phase3_runs, key=lambda x: x['val_roc_auc']) if len(phase3_runs) > 0 else max(results, key=lambda x: x['val_roc_auc'])

    print(f"\n=======================================================")
    print(f" OPTIMIZATION COMPLETE FOR {schema_label.upper()}")
    print(f" Best Phase 3 Score: {absolute_winner['val_roc_auc']:.4f}")
    print(f" Optimal Settings: Dim={absolute_winner['hidden_dim']} | Depth={best_depth} | Bases={absolute_winner['num_bases']} | LR={absolute_winner['lr']} | WD={absolute_winner['weight_decay']}")
    print(f" Optimal Training Epochs: {absolute_winner['best_epoch']+1}")
    print(f"=======================================================")

    cleanup()

    # ── PHASE 4: FINAL GENERALIZATION TEST ON HELD-OUT TEST SPLIT ───────────────
    final_row = evaluate_final_test_row(
        model_factory=model_factory, graph_data=graph_data,
        train_indices=train_indices, val_indices=val_indices, test_indices=test_indices,
        vector_table=vector_table, schema_label=schema_label, save_dir=save_dir,
        hidden_dim=absolute_winner['hidden_dim'], depth=best_depth,
        neighbors=absolute_winner['neighbors'], bases=absolute_winner['num_bases'],
        lr=absolute_winner['lr'], weight_decay=absolute_winner['weight_decay'],
        best_epoch=absolute_winner['best_epoch'], val_roc_auc=absolute_winner['val_roc_auc'],
    )
    results.append(final_row)
    pd.DataFrame(results).to_csv(csv_path, index=False)

    return pd.DataFrame(results)


def evaluate_final_test_row(
    model_factory, graph_data, train_indices, val_indices, test_indices,
    vector_table, schema_label, save_dir,
    hidden_dim, depth, neighbors, bases, lr, weight_decay, best_epoch, val_roc_auc=None,
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
    final_model = model_factory(hidden_list, bases)

    final_trained_model = train_final_model(
        model=final_model, graph_data=graph_data, combined_indices=combined_indices,
        vector_table=vector_table, epochs=best_epoch + 1, batch_size=256,
        num_neighbors=parse_neighbors(neighbors), lr=lr,
        weight_decay=weight_decay, save_path=f"{save_dir}{schema_label}_final_unified_eval.pt"
    )

    test_metrics = evaluate_test_set(
        model=final_trained_model, graph_data=graph_data, test_indices=test_indices,
        batch_size=256, num_neighbors=parse_neighbors(neighbors)
    )

    row_dict = {
        'phase': "4_Final_Test_Evaluation", 'parameter_tested': "unified_model",
        'val_roc_auc': val_roc_auc,
        'test_roc_auc': test_metrics['roc_auc'],
        'test_accuracy': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'],
        'test_f1': test_metrics['f1'],
        'test_pr_auc': test_metrics['pr_auc'],
        'test_class0_f1': test_metrics['class0_f1'],
        'test_class1_f1': test_metrics['class1_f1'],
        'num_parameters': test_metrics['num_parameters'],
        'runtime_sec': 0.0, 'status': "ok", 'hidden_dim': hidden_dim, 'layers': depth,
        'neighbors': str(neighbors), 'num_bases': bases, 'lr': lr, 'weight_decay': weight_decay,
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
    vector_table, schema_label, save_dir, csv_path,
    hidden_dim, depth, neighbors, bases, lr, weight_decay, best_epoch, val_roc_auc=None,
):
    """
    Standalone re-run of just Phase 4 (final test evaluation) for an already-tuned
    configuration -- e.g. re-reading hidden_dim/depth/neighbors/bases/lr/weight_decay
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

    row_dict = evaluate_final_test_row(
        model_factory=model_factory, graph_data=graph_data,
        train_indices=train_indices, val_indices=val_indices, test_indices=test_indices,
        vector_table=vector_table, schema_label=schema_label, save_dir=save_dir,
        hidden_dim=hidden_dim, depth=depth, neighbors=neighbors, bases=bases,
        lr=lr, weight_decay=weight_decay, best_epoch=best_epoch, val_roc_auc=val_roc_auc,
    )
    results.append(row_dict)
    pd.DataFrame(results).to_csv(csv_path, index=False)

    return pd.DataFrame(results)


def run_k_sweep(model_factory, graph_data, schema_label,
                 train_indices, val_indices, test_indices,
                 vector_table, usage_counts,
                 hidden_dim, bases, lr, wd,
                 k_values, epochs=50, patience=15,
                 batch_sizes=(256, 128, 64, 32), csv_path=None, resume=False):
    """
    Controlled K-hop depth sweep (Experiment 2 / RQ1b): varies neighborhood
    depth/fanout (K) with every other hyperparameter held constant, so any
    metric difference between rows is attributable to K alone.

    model_factory(hidden_list, bases) -> nn.Module: same factory shape used
        elsewhere in gnn_tools (RGCN or GraphSAGE unchanged).
    k_values: list of (K, num_neighbors) pairs, e.g.
        [(1, (25,)), (2, (25, 10)), (3, (15, 10, 5))]. hidden_list is built
        as [hidden_dim] * (K - 1). Pass just the K values you want to
        (re-)run, e.g. to retry a single K that previously OOM'd.
    resume: if True and csv_path exists, its rows are loaded first and
        upserted by K (matching K replaced, others untouched) before saving.

    Returns a DataFrame with one row per K. Model name is read off the
    constructed model's own class (type(model).__name__).
    """
    if resume and csv_path and os.path.exists(csv_path):
        rows = pd.read_csv(csv_path).to_dict('records')
    else:
        rows = []

    for k, neighbors in k_values:
        print(f"\n[K-SWEEP] K={k} | Neighbors={neighbors}")
        hidden_list = [hidden_dim] * (k - 1)

        record = {
            'model': None, 'schema': schema_label, 'K': k, 'neighbors': str(neighbors),
            'ROC_AUC': np.nan, 'F1': np.nan, 'PR_AUC': np.nan,
            'param_count': np.nan, 'runtime_sec': 0.0,
            'best_epoch': 0, 'epochs_run': 0,
            'avg_neighborhood_size': np.nan, 'high_signal_text_pct': np.nan, 'fallback_noise_pct': np.nan,
            'status': 'pending',
        }

        try:
            # Same OOM-retry safety net as run_ablation_run -- only
            # triggered on failure, doesn't compromise the "everything but
            # K is constant" design.
            for i, attempt_batch_size in enumerate(batch_sizes):
                model = model_factory(hidden_list, bases)
                record['model'] = type(model).__name__
                record['param_count'] = int(sum(p.numel() for p in model.parameters()))
                try:
                    metrics = evaluate_ablation_configuration(
                        model=model, graph_data=graph_data,
                        train_indices=train_indices, val_indices=val_indices, test_indices=test_indices,
                        vector_table=vector_table, usage_counts=usage_counts, neighbors=neighbors,
                        lr=lr, wd=wd, epochs=epochs, patience=patience,
                        batch_size=attempt_batch_size,
                    )
                    break
                except RuntimeError as oom_err:
                    if "out of memory" not in str(oom_err).lower() or i == len(batch_sizes) - 1:
                        raise
                    print(f"  -> CUDA OOM at batch_size={attempt_batch_size}, retrying at batch_size={batch_sizes[i + 1]}...")
                    del oom_err, model
                    cleanup()

            record.update({
                'ROC_AUC': float(metrics['test_roc_auc']),
                'F1': float(metrics['test_f1_score']),
                'PR_AUC': float(metrics['test_pr_auc']),
                'runtime_sec': float(metrics['training_time_sec']),
                'best_epoch': int(metrics['best_epoch']),
                'epochs_run': int(metrics['epochs_run']),
                'avg_neighborhood_size': float(metrics['avg_neighborhood_size']),
                'high_signal_text_pct': float(metrics['high_signal_text_pct']),
                'fallback_noise_pct': float(metrics['fallback_noise_pct']),
                'status': 'completed',
            })
            print(f"  -> ROC-AUC: {metrics['test_roc_auc']:.4f} | F1: {metrics['test_f1_score']:.4f} | Params: {record['param_count']:,}")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("  -> CUDA OOM: Gracefully skipping this K.")
                record['status'] = 'oom'
                del e  # drop the traceback so cleanup() can actually reclaim GPU memory
                cleanup()
            else:
                record['status'] = 'failed'
                raise e

        # Upsert by K: drop any existing row for this K (from a loaded
        # resume, or an earlier K value in this same call) before adding
        # the fresh one, so re-running a K never leaves a stale duplicate.
        rows = [r for r in rows if r.get('K') != k]
        rows.append(record)
        if csv_path:
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    return pd.DataFrame(rows)