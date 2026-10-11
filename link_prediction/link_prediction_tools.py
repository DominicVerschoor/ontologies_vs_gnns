"""
Shared infrastructure for the link-prediction supporting experiment, kept
separate from gnn_tools.py (node classification).

Link prediction runs on a small (~10k-case) induced subgraph, not the full
graph -- satisfaction labels are irrelevant to this task, so cases can be
sampled freely, making full filtered ranking (MRR / Hits@k) affordable.
RGCN/GraphSAGE share one DistMult decoder; TransE is scored through the
same evaluate_link_prediction() ranking function via a manual score_fn, so
metric definitions can't drift between models.
"""

import os
import re
import sys

repo_path = '/Workspace/Users/dversch@emea.corpdir.net/.bundle/KG_Model/prod/files'
if repo_path not in sys.path:
    sys.path.append(repo_path)

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.nn import RGCNConv, SAGEConv
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from node_prediction.gnn_tools import cleanup, get_graph_mask, get_relation_dict


# ─────────────────────────────────────────────────────────────────────────
# Shared case sampling (same cases in both KGs despite different node ids)
# ─────────────────────────────────────────────────────────────────────────

def _entity_id(uri):
    """Trailing path segment of a raw N-Triples URI token, e.g.
    '<http://.../Case/12345>' -> '12345'. Same convention
    training_data.build_feature_lookup already uses to split URIs."""
    s = uri.strip('<>').strip('"')
    return s.split('/')[-1] if '/' in s else s


def get_case_uri_to_node_id(graph_data, ontology_obj, schema_label):
    """Maps this graph's Case entity id (URI suffix) -> local node id."""
    case_mask = get_graph_mask(graph_data, ontology_obj, schema_label, 'Case')
    case_node_ids = torch.where(case_mask)[0].tolist()
    idx_to_uri = {idx: uri for uri, idx in ontology_obj.enum_nodes.items()}
    return {_entity_id(idx_to_uri[idx]): idx for idx in case_node_ids}


def sample_case_ids(graph_data, ontology_obj, schema_label, n=10000, seed=42):
    """
    Samples `n` Case entity ids from THIS graph alone -- no dependency on
    the other schema's Ontology being in memory at the same time (base and
    sulo are each a multi-million-node parse; this project's established
    workflow, same as RGCN.ipynb/graph_sage.ipynb's schema_label toggle, is
    one schema per kernel session, never both at once).

    Returns (case_ids, node_ids): index-aligned lists -- case_ids[i] is the
    entity id string, node_ids[i] is its local node id in THIS graph.
    Callers should persist case_ids to disk (see resolve_case_node_ids) so
    the OTHER schema's separate run reuses the exact same sample instead of
    drawing an independent one.
    """
    case_to_node_id = get_case_uri_to_node_id(graph_data, ontology_obj, schema_label)
    available = sorted(case_to_node_id.keys())
    print(f"{len(available)} case ids available in this graph.")

    if len(available) < n:
        print(f"WARNING: only {len(available)} case ids available -- requested {n}.")
        n = len(available)

    rng = np.random.default_rng(seed)
    sampled = rng.choice(np.array(available, dtype=object), size=n, replace=False).tolist()
    node_ids = [case_to_node_id[c] for c in sampled]
    return sampled, node_ids


def resolve_case_node_ids(graph_data, ontology_obj, schema_label, case_ids):
    """
    Resolves a previously-sampled list of Case entity id strings (loaded
    from disk, sampled during the OTHER schema's separate run) to THIS
    graph's own local node ids -- so both schemas' link-prediction
    subgraphs are built around the exact same cases without ever needing
    both Ontology parses in memory simultaneously.

    Both KGs are populated from the same underlying balanced case sample
    (see Graph_population.ipynb / CLAUDE.md's fixed experimental
    conditions), so missing ids should be rare or zero -- warns rather than
    failing if any aren't found, and proceeds with whatever subset resolves.

    Returns (resolved_case_ids, node_ids): index-aligned, same shape
    contract as sample_case_ids().
    """
    case_to_node_id = get_case_uri_to_node_id(graph_data, ontology_obj, schema_label)
    resolved_ids, node_ids = [], []
    for cid in case_ids:
        if cid in case_to_node_id:
            resolved_ids.append(cid)
            node_ids.append(case_to_node_id[cid])

    missing = len(case_ids) - len(resolved_ids)
    if missing:
        print(f"WARNING: {missing}/{len(case_ids)} previously-sampled case ids were not "
              f"found in this schema's graph -- both KGs are expected to share the same "
              f"case population; investigate if this number is large.")
    return resolved_ids, node_ids


# ─────────────────────────────────────────────────────────────────────────
# Induced subgraph construction
# ─────────────────────────────────────────────────────────────────────────

def _bfs_expand(edge_index, num_nodes, seed_mask, num_hops):
    """Vectorized multi-hop frontier expansion -- same technique as
    extract_graph_geometry.ipynb's hop_distance_from_mask, just returning
    the reached-node mask instead of per-node hop distances."""
    src, dst = edge_index
    visited = seed_mask.clone()
    frontier = seed_mask.clone()
    for _ in range(num_hops):
        if not frontier.any():
            break
        active_src = frontier[src]
        active_dst = frontier[dst]
        next_frontier = torch.zeros(num_nodes, dtype=torch.bool)
        next_frontier[dst[active_src]] = True
        next_frontier[src[active_dst]] = True
        next_frontier &= ~visited
        visited |= next_frontier
        frontier = next_frontier
    return visited


def build_induced_subgraph(graph_data, case_node_ids, num_hops=2):
    """
    Builds a small induced subgraph around the given seed Case node ids by
    BFS-expanding `num_hops` hops through graph_data.edge_index, then
    remapping surviving nodes to a local contiguous id space.

    graph_data is expected to be the already-loaded {schema}_vars.pt graph
    (already degree-capped via cap_max_degree(max_degree=128) in
    RGCN.ipynb/graph_sage.ipynb), so hub-node blowup during expansion is
    already bounded -- no additional capping is applied here, but the
    resulting node/edge counts are printed so an unexpectedly large result
    is caught immediately rather than deep into training.

    Returns (sub_data, local_to_orig, orig_to_local):
      sub_data        -- new Data(edge_index, edge_type, num_nodes,
                          n_id_to_vec_row, degree_feat) in local id space.
      local_to_orig    -- LongTensor[num_sub_nodes], local id -> original id.
      orig_to_local     -- LongTensor[num_orig_nodes], original id -> local id
                          (-1 for nodes not kept).
    """
    num_nodes = graph_data.num_nodes
    seed_mask = torch.zeros(num_nodes, dtype=torch.bool)
    seed_mask[torch.tensor(case_node_ids, dtype=torch.long)] = True

    keep_mask = _bfs_expand(graph_data.edge_index, num_nodes, seed_mask, num_hops)
    local_to_orig = torch.where(keep_mask)[0]

    orig_to_local = torch.full((num_nodes,), -1, dtype=torch.long)
    orig_to_local[local_to_orig] = torch.arange(local_to_orig.size(0))

    src, dst = graph_data.edge_index
    edge_keep = keep_mask[src] & keep_mask[dst]

    sub_edge_index = torch.stack([
        orig_to_local[src[edge_keep]],
        orig_to_local[dst[edge_keep]],
    ])
    sub_edge_type = graph_data.edge_type[edge_keep]
    sub_n_id_to_vec_row = graph_data.n_id_to_vec_row[local_to_orig]
    sub_degree_feat = graph_data.degree_feat[local_to_orig]

    sub_data = Data(
        edge_index=sub_edge_index,
        edge_type=sub_edge_type,
        num_nodes=local_to_orig.size(0),
        n_id_to_vec_row=sub_n_id_to_vec_row,
        degree_feat=sub_degree_feat,
    )
    print(f"Induced LP subgraph: {sub_data.num_nodes:,} nodes, "
          f"{sub_data.edge_index.size(1):,} edges "
          f"(from {len(case_node_ids):,} seed cases, {num_hops}-hop expansion).")
    return sub_data, local_to_orig, orig_to_local


# ─────────────────────────────────────────────────────────────────────────
# Relation triple extraction, splitting, held-out masking
# ─────────────────────────────────────────────────────────────────────────

def _relation_local_name(uri):
    """Local name of a relation URI, e.g. '<https://w3id.org/sulo#hasPart>' ->
    'hasPart', '<.../base_kg#hasTaskDescription>' -> 'hasTaskDescription'."""
    return re.split(r'[/#]', uri.strip('<>'))[-1]


def get_relation_edge_type_ids(ontology_obj, relation_name):
    """
    Resolves a relation name to its (forward_ids, inverse_ids) edge_type id
    lists (doubled per Ontology's rel*2/rel*2+1 convention). Returns None if
    no relation URI matches.

    Matches the URI's exact local name (case-insensitive), NOT a substring
    match -- several relation names here are prefixes of other real
    predicates (e.g. 'hasTask' of 'hasTaskDescription'), so substring
    matching would risk aliasing two different relations onto the same
    decoder embedding row.
    """
    relation_dict = get_relation_dict(ontology_obj)
    target = relation_name.lower()
    raw_rel_ids = [r_id for r_uri, r_id in relation_dict.items()
                   if _relation_local_name(r_uri).lower() == target]
    if not raw_rel_ids:
        return None
    if len(raw_rel_ids) > 1:
        print(f"WARNING: relation name '{relation_name}' matched {len(raw_rel_ids)} "
              f"relation URIs with the exact same local name -- using all of them together.")
    fwd_ids = [rid * 2 for rid in raw_rel_ids]
    inv_ids = [rid * 2 + 1 for rid in raw_rel_ids]
    return fwd_ids, inv_ids


def get_graph_mask_local(full_graph_data, ontology_obj, schema_label, concept_name, local_to_orig):
    """
    gnn_tools.get_graph_mask, but safe to use against an induced subgraph.

    get_graph_mask builds its mask sized to `graph_data.num_nodes` while
    iterating `ontology_obj.enum_nodes` -- both must be in the SAME id
    space. `full_graph_data` and `ontology_obj` satisfy that (both are the
    original, un-remapped graph), but `build_induced_subgraph`'s output
    (`sub_data`) is NOT -- its node ids are a small LOCAL renumbering, so
    calling get_graph_mask(sub_data, ontology_obj, ...) directly indexes
    ontology_obj's (up to multi-million) original node ids into a mask
    sized to the subgraph's (thousands of) local ids and throws an
    IndexError. This computes the mask in the correct (full) space first,
    then translates it into subgraph-local space via local_to_orig (local
    id -> original id, from build_induced_subgraph).
    """
    full_mask = get_graph_mask(full_graph_data, ontology_obj, schema_label, concept_name)
    return full_mask[local_to_orig]


def extract_direct_triples(sub_data, ontology_obj, schema_label, predicate, target_concept,
                            full_graph_data, local_to_orig):
    """
    One-hop triple extraction: edges matching `predicate` whose TAIL is a
    member of `target_concept` (resolved via get_graph_mask_local).

    The target_concept filter matters because some schemas share one
    predicate across multiple concepts (e.g. SULO's generic 'isReferredToIn'
    for CaseDescription/TaskDescription/CaseSolutionFinalStatement) -- only
    the tail's node-URI pattern tells them apart, not the predicate alone.

    full_graph_data/local_to_orig: the original (un-remapped) graph and
    build_induced_subgraph's id mapping, needed so the concept mask is
    computed in the right id space (see get_graph_mask_local).

    Returns LongTensor[num_triples, 2] of (h, t) LOCAL subgraph node ids.
    """
    ids = get_relation_edge_type_ids(ontology_obj, predicate)
    if ids is None:
        return torch.empty((0, 2), dtype=torch.long)
    fwd_ids, _ = ids
    mask = torch.isin(sub_data.edge_type, torch.tensor(fwd_ids))
    h = sub_data.edge_index[0, mask]
    t = sub_data.edge_index[1, mask]

    target_mask = get_graph_mask_local(full_graph_data, ontology_obj, schema_label, target_concept, local_to_orig)
    keep = target_mask[t]
    return torch.stack([h[keep], t[keep]], dim=1)


def extract_role_chain_triples(sub_data, ontology_obj, schema_label,
                                to_role_predicate, role_concept, from_role_predicate, target_concept,
                                full_graph_data, local_to_orig):
    """
    Two-hop composite triple extraction for SULO's reified Role pattern:
        source --to_role_predicate--> {role_concept instance} --from_role_predicate--> target
    e.g. Case --hasParticipant--> VehicleRole --isFeatureOf--> Vehicle.

    No single edge_type connects source to target directly -- the (h, t)
    pair is synthesized by joining the two edge sets on the Role node's id.

    full_graph_data/local_to_orig: see extract_direct_triples.

    Returns (triples, role_node_ids):
      triples       -- LongTensor[num_triples, 2] of (h, t) LOCAL node ids.
      role_node_ids -- LongTensor[num_triples], the intermediate Role node's
                       local id for each triple (needed by
                       mask_role_chain_nodes() -- there's no direct (h, t)
                       edge to remove, only the Role node's two edges).
    """
    ids1 = get_relation_edge_type_ids(ontology_obj, to_role_predicate)
    ids2 = get_relation_edge_type_ids(ontology_obj, from_role_predicate)
    empty_triples = torch.empty((0, 2), dtype=torch.long)
    empty_aux = torch.empty(0, dtype=torch.long)
    if ids1 is None or ids2 is None:
        return empty_triples, empty_aux

    role_mask = get_graph_mask_local(full_graph_data, ontology_obj, schema_label, role_concept, local_to_orig)
    target_mask = get_graph_mask_local(full_graph_data, ontology_obj, schema_label, target_concept, local_to_orig)

    fwd1, _ = ids1
    mask1 = torch.isin(sub_data.edge_type, torch.tensor(fwd1))
    src1 = sub_data.edge_index[0, mask1]
    mid1 = sub_data.edge_index[1, mask1]
    keep1 = role_mask[mid1]
    src1, mid1 = src1[keep1], mid1[keep1]

    fwd2, _ = ids2
    mask2 = torch.isin(sub_data.edge_type, torch.tensor(fwd2))
    mid2 = sub_data.edge_index[0, mask2]
    dst2 = sub_data.edge_index[1, mask2]
    keep2 = role_mask[mid2] & target_mask[dst2]
    mid2, dst2 = mid2[keep2], dst2[keep2]

    # Join on the role node id via a plain dict -- role instances are
    # per-(source, target) pairs by construction (see Graph_population.ipynb),
    # so this join is small and a Python-level dict is simple and sufficient.
    mid2_to_dst = {}
    for m, d in zip(mid2.tolist(), dst2.tolist()):
        mid2_to_dst.setdefault(m, []).append(d)

    h_out, t_out, mid_out = [], [], []
    for s, m in zip(src1.tolist(), mid1.tolist()):
        for d in mid2_to_dst.get(m, []):
            h_out.append(s)
            t_out.append(d)
            mid_out.append(m)

    if not h_out:
        return empty_triples, empty_aux
    triples = torch.tensor(list(zip(h_out, t_out)), dtype=torch.long)
    role_node_ids = torch.tensor(mid_out, dtype=torch.long)
    return triples, role_node_ids


def resolve_relation_triples(sub_data, ontology_obj, schema_label, spec, full_graph_data, local_to_orig):
    """
    Dispatches on a per-schema relation spec to the right extractor. spec is
    one of:
      {'type': 'direct', 'predicate': str, 'target_concept': str}
      {'type': 'role_chain', 'to_role_predicate': str, 'role_concept': str,
       'from_role_predicate': str, 'target_concept': str}

    Returns (triples, aux) -- aux is None for 'direct', the role_node_ids
    tensor for 'role_chain'.
    """
    if spec['type'] == 'direct':
        triples = extract_direct_triples(sub_data, ontology_obj, schema_label,
                                          spec['predicate'], spec['target_concept'],
                                          full_graph_data, local_to_orig)
        return triples, None
    elif spec['type'] == 'role_chain':
        return extract_role_chain_triples(
            sub_data, ontology_obj, schema_label,
            spec['to_role_predicate'], spec['role_concept'],
            spec['from_role_predicate'], spec['target_concept'],
            full_graph_data, local_to_orig,
        )
    else:
        raise ValueError(f"Unknown relation spec type: {spec['type']}")


def split_relation_triples(triples, aux=None, val_frac=0.1, test_frac=0.1, seed=42, min_triples=20):
    """
    80/10/10-style shuffle-split of one relation's triples. Returns None
    (with a warning) if there are too few triples in the subgraph to split
    meaningfully -- callers should skip that relation for that schema
    rather than force a degenerate split.

    aux: optional parallel tensor (same length as triples) split with the
        SAME permutation/indices as triples -- used to carry a ROLE_CHAIN
        relation's role_node_ids through the split so val/test triples keep
        what mask_role_chain_nodes() needs. Adds 'train_aux'/'val_aux'/
        'test_aux' keys to the result when given.
    """
    n = triples.size(0)
    if n < min_triples:
        print(f"WARNING: only {n} triples available for this relation in the "
              f"subgraph (need >= {min_triples}) -- skipping split.")
        return None
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(n * val_frac))
    n_test = max(1, int(n * test_frac))
    val_idx = perm[:n_val]
    test_idx = perm[n_val:n_val + n_test]
    train_idx = perm[n_val + n_test:]
    result = {'train': triples[train_idx], 'val': triples[val_idx], 'test': triples[test_idx]}
    if aux is not None:
        result['train_aux'] = aux[train_idx]
        result['val_aux'] = aux[val_idx]
        result['test_aux'] = aux[test_idx]
    return result


def split_train_disjoint(train_triples, aux=None, frac=0.3, seed=43, min_triples=10):
    """
    Further partitions a relation's TRAIN triples into two DISJOINT sets:
      - 'mp'  -- stays in the message-passing graph, never supervised.
      - 'sup' -- removed from message passing, used only as a BCE
                 supervision target in train_link_prediction_model.

    Without this split, every train triple would be both a message-passing
    edge and a supervision target -- the encoder could trivially read the
    h->t edge out of its own 1-hop neighborhood as a shortcut, driving
    train_loss down without learning anything that transfers to held-out
    edges.

    frac: fraction assigned to 'sup' (default 0.3, matching PyG's
        RandomLinkSplit disjoint_train_ratio) -- kept well under 1.0 so
        message passing isn't starved of most of that relation's structure.
    Uses a different default seed (43) than split_relation_triples's val/test
    split (42) so this is an independent partition.

    Returns None (with a warning) if there are too few train triples to
    split meaningfully -- callers should treat all train triples as 'mp'.
    """
    n = train_triples.size(0)
    if n < min_triples:
        print(f"WARNING: only {n} train triples for this relation -- skipping "
              f"disjoint train/message-passing split (all {n} kept as "
              f"message-passing edges, none used as supervision).")
        return None
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_sup = max(1, int(n * frac))
    sup_idx = perm[:n_sup]
    mp_idx = perm[n_sup:]
    result = {'mp': train_triples[mp_idx], 'sup': train_triples[sup_idx]}
    if aux is not None:
        result['mp_aux'] = aux[mp_idx]
        result['sup_aux'] = aux[sup_idx]
    return result


def mask_held_out_triples(sub_data, ontology_obj, held_out_by_predicate):
    """
    Removes the given held-out (h, t) pairs -- both the forward edge and
    its paired inverse -- from sub_data.edge_index/edge_type, for every
    DIRECT relation in held_out_by_predicate at once. Vectorized via integer
    pair encoding (src * num_nodes + dst), same isin-based approach
    gnn_tools.mask_graph_topology already uses for relation masking. Not
    applicable to ROLE_CHAIN relations -- see mask_role_chain_nodes().

    held_out_by_predicate: dict PREDICATE name (e.g. 'hasVehicle', or
        SULO's shared 'isReferredToIn'/'hasPart') -> LongTensor[N, 2] of
        (h, t) local-id pairs to remove (val+test triples combined) for
        that predicate. Only these specific triples are removed -- most of
        each relation's edges stay visible during training, so this is NOT
        relation ablation.
    """
    num_nodes = sub_data.num_nodes
    src, dst = sub_data.edge_index
    edge_key = src * num_nodes + dst
    remove_mask = torch.zeros(sub_data.edge_index.size(1), dtype=torch.bool)

    for predicate, triples in held_out_by_predicate.items():
        if triples is None or triples.size(0) == 0:
            continue
        ids = get_relation_edge_type_ids(ontology_obj, predicate)
        if ids is None:
            continue
        fwd_ids, inv_ids = ids
        h, t = triples[:, 0], triples[:, 1]
        fwd_keys = h * num_nodes + t
        inv_keys = t * num_nodes + h

        fwd_type_mask = torch.isin(sub_data.edge_type, torch.tensor(fwd_ids))
        inv_type_mask = torch.isin(sub_data.edge_type, torch.tensor(inv_ids))

        remove_mask |= fwd_type_mask & torch.isin(edge_key, fwd_keys)
        remove_mask |= inv_type_mask & torch.isin(edge_key, inv_keys)

    keep_mask = ~remove_mask
    masked = sub_data.clone()
    masked.edge_index = sub_data.edge_index[:, keep_mask]
    masked.edge_type = sub_data.edge_type[keep_mask]
    print(f"Masked {remove_mask.sum().item():,} held-out direct-relation edges "
          f"(of {sub_data.edge_index.size(1):,}).")
    return masked


def mask_role_chain_nodes(sub_data, held_out_role_node_ids):
    """
    Severs held-out ROLE_CHAIN triples by removing EVERY edge touching
    their intermediate Role node (both the incoming to_role_predicate edge
    and the outgoing from_role_predicate edge, both directions). A Role
    instance exists solely to link one source entity to one target entity
    (see Graph_population.ipynb), so removing all of its edges cleanly
    removes that one fact without affecting any other triple.

    held_out_role_node_ids: 1D LongTensor of Role node LOCAL ids (val+test
        triples' role_node_ids, concatenated across all ROLE_CHAIN relations).
    """
    if held_out_role_node_ids.numel() == 0:
        return sub_data
    node_mask = torch.zeros(sub_data.num_nodes, dtype=torch.bool)
    node_mask[held_out_role_node_ids] = True

    src, dst = sub_data.edge_index
    remove_mask = node_mask[src] | node_mask[dst]
    keep_mask = ~remove_mask

    masked = sub_data.clone()
    masked.edge_index = sub_data.edge_index[:, keep_mask]
    masked.edge_type = sub_data.edge_type[keep_mask]
    print(f"Masked {remove_mask.sum().item():,} edges touching "
          f"{held_out_role_node_ids.numel():,} held-out Role nodes "
          f"(of {sub_data.edge_index.size(1):,}).")
    return masked


# ─────────────────────────────────────────────────────────────────────────
# Candidate pools & negative sampling
# ─────────────────────────────────────────────────────────────────────────

def build_candidate_pool(all_relation_triples):
    """
    Candidate pool for ranking/negative-sampling = the set of tail nodes
    empirically observed for this relation in the subgraph (data-driven
    type constraint, not reliant on schema rdfs:range declarations).

    IMPORTANT: pass the FULL (train+val+test, pre-mask) triples tensor for
    this relation, not the post-mask training triples -- a tail that only
    appears via a held-out test triple must still be a valid ranking
    candidate, it just won't be visible to message passing during training.
    """
    if all_relation_triples.size(0) == 0:
        return torch.empty(0, dtype=torch.long)
    return torch.unique(all_relation_triples[:, 1])


def build_true_tails_by_head(all_relation_triples):
    """dict: head local node id -> set of ALL true tail ids for this
    relation (train+val+test) -- the 'filtered' set to exclude when
    building negatives or ranking, so other correct answers for the same
    (h, r) aren't penalized as false negatives."""
    true_tails = {}
    for h, t in all_relation_triples.tolist():
        true_tails.setdefault(h, set()).add(t)
    return true_tails


def sample_negatives(h_ids, candidate_pool, true_tails_by_h, k=1):
    """
    For each h in h_ids, samples k tails uniformly from candidate_pool,
    excluding h's true tails under this relation. Returns a flat
    LongTensor[h_ids.size(0) * k] in repeat_interleave(k)-compatible order
    (h_ids[i]'s k negatives occupy positions [i*k : i*k+k]).
    """
    pool_np = candidate_pool.numpy()
    out = np.empty((h_ids.size(0), k), dtype=np.int64)
    for i, h in enumerate(h_ids.tolist()):
        excluded = true_tails_by_h.get(h, set())
        if excluded:
            choices = pool_np[~np.isin(pool_np, np.fromiter(excluded, dtype=np.int64))]
        else:
            choices = pool_np
        if len(choices) == 0:
            choices = pool_np  # pool too small to exclude anything -- allow overlap rather than crash
        out[i] = np.random.choice(choices, size=k, replace=len(choices) < k)
    return torch.from_numpy(out.reshape(-1))


# ─────────────────────────────────────────────────────────────────────────
# Encoders (embedding output -- no classification head)
# ─────────────────────────────────────────────────────────────────────────

class _FeaturizationMixin:
    """Shared frozen-table + trainable-categorical-fallback featurization,
    copied from RGCNModel/GraphSAGEModel in RGCN.ipynb/graph_sage.ipynb
    (same convention analysis.ipynb already documents as 'mirrored
    verbatim' -- this codebase accepts some duplication across
    notebooks/modules for these model classes rather than force a shared
    import)."""

    def _init_featurization(self, vector_table, categorical_row_ids):
        num_rows, emb_dim = vector_table.shape
        is_categorical = torch.zeros(num_rows, dtype=torch.bool)
        is_categorical[categorical_row_ids] = True
        self.register_buffer('is_categorical_mask', is_categorical)
        self.frozen_table = vector_table

        self.categorical_embedding = nn.Embedding(len(categorical_row_ids), emb_dim)
        with torch.no_grad():
            self.categorical_embedding.weight.copy_(
                vector_table[categorical_row_ids.to(vector_table.device)]
            )

        local_id = torch.full((num_rows,), -1, dtype=torch.long)
        local_id[categorical_row_ids] = torch.arange(len(categorical_row_ids))
        self.register_buffer('local_categorical_id', local_id)

    def _featurize(self, node_row_ids, degree_feat):
        # frozen_table stays on CPU -- holding the full multi-million-row
        # table on GPU alongside the model caused OOMs independent of
        # hidden_dim. Only the gathered per-batch slice moves to GPU here.
        x = self.frozen_table[node_row_ids.cpu()].to(degree_feat.device)
        cat_mask = self.is_categorical_mask[node_row_ids]
        if cat_mask.any():
            local_ids = self.local_categorical_id[node_row_ids[cat_mask]]
            x[cat_mask] = self.categorical_embedding(local_ids)
        return torch.cat([x, degree_feat.unsqueeze(1)], dim=1)


class RGCNEncoder(_FeaturizationMixin, nn.Module):
    """Same as RGCNModel but every layer (including the last) outputs
    hidden_dim node embeddings -- no final 1-dim classification conv."""

    def __init__(self, in_channels, hidden_channels_list, num_relations,
                 vector_table, categorical_row_ids, num_bases=None, dropout=0.2):
        super().__init__()
        self._init_featurization(vector_table, categorical_row_ids)
        self.convs = nn.ModuleList()
        dims = [in_channels] + list(hidden_channels_list)
        for i in range(len(dims) - 1):
            self.convs.append(RGCNConv(dims[i], dims[i + 1], num_relations=num_relations, num_bases=num_bases))
        self.dropout = dropout

    def forward(self, node_row_ids, edge_index, edge_type, degree_feat):
        x = self._featurize(node_row_ids, degree_feat)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GraphSAGEEncoder(_FeaturizationMixin, nn.Module):
    """Same as GraphSAGEModel but every layer outputs hidden_dim node
    embeddings -- no final 1-dim classification conv. `edge_type` is
    accepted and ignored, matching gnn_tools.train_model's documented
    convention for relation-agnostic architectures."""

    def __init__(self, in_channels, hidden_channels_list, vector_table, categorical_row_ids,
                 aggr='mean', dropout=0.5):
        super().__init__()
        self._init_featurization(vector_table, categorical_row_ids)
        self.convs = nn.ModuleList()
        dims = [in_channels] + list(hidden_channels_list)
        for i in range(len(dims) - 1):
            self.convs.append(SAGEConv(dims[i], dims[i + 1], aggr=aggr, project=True))
        self.dropout = dropout

    def forward(self, node_row_ids, edge_index, edge_type, degree_feat):
        x = self._featurize(node_row_ids, degree_feat)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ─────────────────────────────────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────────────────────────────────

class DistMultDecoder(nn.Module):
    """Relation-aware triple scorer: score(h, r, t) = sum(h_emb * rel_emb[r] * t_emb).
    Reused unchanged on top of either encoder -- a decoder doesn't care
    which encoder produced the embeddings, which is what keeps the RGCN and
    GraphSAGE LP pipelines maximally parallel. `num_relations` should be
    graph_data.edge_type.max()+1 on the (post-mask) training subgraph, same
    convention RGCNConv's num_relations already uses elsewhere."""

    def __init__(self, num_relations, hidden_dim):
        super().__init__()
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

    def forward(self, h_emb, r_ids, t_emb):
        r_emb = self.relation_embedding(r_ids)
        return (h_emb * r_emb * t_emb).sum(dim=-1)


# ─────────────────────────────────────────────────────────────────────────
# Score-fn helpers (cached embeddings -- encoder runs ONCE, not per triple)
# ─────────────────────────────────────────────────────────────────────────

def compute_node_embeddings_batched(encoder, sub_graph_data, node_ids, num_neighbors, batch_size, device):
    """
    Computes embeddings for exactly `node_ids` via mini-batched NeighborLoader
    inference -- the same proven approach gnn_tools.evaluate_test_set already
    uses for node classification -- instead of one forward pass over the
    whole induced subgraph. Bounds per-batch GPU memory to batch_size x
    fan-out regardless of how many total nodes are being embedded or how big
    the subgraph is overall.

    sub_graph_data is expected to stay CPU-resident (NeighborLoader's sampler
    operates on CPU); only each yielded mini-batch is moved to GPU.

    Returns (emb_table, id_to_row): a CPU-resident [len(node_ids), hidden_dim]
    table plus a [sub_graph_data.num_nodes] lookup (original node id -> row
    in emb_table, -1 if not computed) -- callers index into this directly
    instead of re-running the encoder per lookup.
    """
    encoder = encoder.to(device).eval()
    loader = NeighborLoader(
        sub_graph_data, num_neighbors=list(num_neighbors), batch_size=batch_size,
        input_nodes=node_ids, num_workers=0, shuffle=False,
    )
    emb_chunks, orig_id_chunks = [], []
    try:
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = encoder(batch.n_id_to_vec_row, batch.edge_index, batch.edge_type, batch.degree_feat)
                emb_chunks.append(out[:batch.batch_size].cpu())
                orig_id_chunks.append(batch.n_id[:batch.batch_size].cpu())
    finally:
        del loader
        cleanup()

    emb_table = torch.cat(emb_chunks, dim=0)
    all_ids = torch.cat(orig_id_chunks, dim=0)

    id_to_row = torch.full((sub_graph_data.num_nodes,), -1, dtype=torch.long)
    id_to_row[all_ids] = torch.arange(all_ids.size(0))
    return emb_table, id_to_row


def make_gnn_score_fn(encoder, decoder, sub_graph_data, triples_by_relation, candidate_pools,
                       num_neighbors, batch_size=1024, device=None):
    """
    Builds a score_fn(h_ids, r_ids, t_ids) backed by mini-batch-computed
    embeddings (see compute_node_embeddings_batched) for exactly the nodes
    evaluate_link_prediction() will need: each relation's held-out triples'
    heads, plus that relation's full candidate pool (ranking scores every
    candidate against the true tail). Embeddings are computed ONCE per call
    (not per candidate/per triple), then score_fn is a cheap tensor lookup.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    decoder = decoder.to(device).eval()

    needed = set()
    for rel_name, triples in triples_by_relation.items():
        if triples is not None and triples.size(0) > 0:
            needed.update(triples[:, 0].tolist())
        pool = candidate_pools.get(rel_name)
        if pool is not None and pool.numel() > 0:
            needed.update(pool.tolist())

    node_ids = torch.tensor(sorted(needed), dtype=torch.long)
    emb_table, id_to_row = compute_node_embeddings_batched(
        encoder, sub_graph_data, node_ids, num_neighbors, batch_size, device,
    )

    def score_fn(h_ids, r_ids, t_ids):
        with torch.no_grad():
            h_emb = emb_table[id_to_row[h_ids]].to(device)
            t_emb = emb_table[id_to_row[t_ids]].to(device)
            return decoder(h_emb, r_ids.to(device), t_emb).cpu()

    return score_fn


# ─────────────────────────────────────────────────────────────────────────
# Training loop (mini-batched via LinkNeighborLoader)
# ─────────────────────────────────────────────────────────────────────────

def train_link_prediction_model(encoder, decoder, sub_graph_data,
                                 train_triples_by_relation, val_triples_by_relation,
                                 candidate_pools, true_tails_by_relation, relation_fwd_ids,
                                 num_neighbors, epochs=100, lr=1e-3, weight_decay=1e-4,
                                 neg_per_pos=1, batch_size=1024, patience=15, eval_frequency=1,
                                 seed=42, decoder_weight_decay=None,
                                 save_path=None, resume=False, monitor='pooled',
                                 max_val_eval_triples=None):
    """
    Mini-batched training via LinkNeighborLoader -- one sampled neighborhood
    per batch of (h, r, t) triples, not a full forward pass over the whole
    subgraph every epoch (this subgraph is too large, ~2.6M nodes, for
    full-batch to fit regardless of hidden_dim). One LinkNeighborLoader is
    built per relation per epoch, with fresh negatives resampled from that
    relation's candidate pool each time.

    num_neighbors: per-hop fan-out, typically anchored on this model+schema's
        NC-tuned 'neighbors' config.
    eval_frequency: validate every N epochs -- validation re-embeds each
        relation's full candidate pool, so this trades check frequency for
        overhead. `epochs_no_improve` advances by eval_frequency (not 1) per
        non-improving check, so `patience` stays measured in epochs
        regardless of check frequency.
    seed: reseeds torch/numpy's global RNG at the start of this call, so
        loader shuffle order and per-epoch negative sampling are
        reproducible.
    decoder_weight_decay: L2 for the decoder specifically, via a separate
        optimizer param group. Defaults to `weight_decay` (one shared rate)
        when not given.
    save_path/resume: checkpoint/resume, saving on every eval checkpoint
        (not just improvements) so a manually-interrupted run is fully
        recoverable. `early_stopped` is set only when patience genuinely
        triggers -- merely reaching the epoch budget doesn't block a later
        resume call with a higher `epochs`.
    monitor: 'pooled' (default) uses a triple-count-weighted MRR average,
        which large near-unrankable relations can dominate; 'macro' uses
        the unweighted mean across relations instead, giving each an equal
        vote. Both are always recorded in `history` regardless of selection.
    max_val_eval_triples: caps how many of each relation's val triples are
        ranked per periodic eval check (candidate_pools stay full-size) --
        only affects in-training checks, not the final test evaluation.

    Returns (encoder, decoder, best_val_mrr, history) with the best-val-MRR
    weights restored. `best_val_mrr` reflects whichever metric `monitor`
    selected.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    decoder_wd = weight_decay if decoder_weight_decay is None else decoder_weight_decay
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'weight_decay': weight_decay},
        {'params': decoder.parameters(), 'weight_decay': decoder_wd},
    ], lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    train_by_fwd_id = {}
    for rel_name, triples in train_triples_by_relation.items():
        if triples is None or triples.size(0) == 0:
            continue
        train_by_fwd_id[relation_fwd_ids[rel_name]] = (rel_name, triples)

    # See max_val_eval_triples' docstring -- a fixed (seeded once, not
    # per-checkpoint) subsample of each relation's val triples, used ONLY
    # by the periodic eval checks below. candidate_pools are untouched, so
    # each sampled triple is still ranked against its relation's full pool.
    eval_val_triples_by_relation = val_triples_by_relation
    if max_val_eval_triples is not None:
        eval_val_triples_by_relation = {}
        _val_sample_gen = torch.Generator().manual_seed(seed)
        for rel_name, triples in val_triples_by_relation.items():
            if triples is not None and triples.size(0) > max_val_eval_triples:
                keep_idx = torch.randperm(triples.size(0), generator=_val_sample_gen)[:max_val_eval_triples]
                eval_val_triples_by_relation[rel_name] = triples[keep_idx]
            else:
                eval_val_triples_by_relation[rel_name] = triples

    best_val_mrr = -1.0
    best_state = None
    epochs_no_improve = 0
    history = []
    start_epoch = 0
    already_completed = False

    if resume and save_path and os.path.exists(save_path):
        print(f"Found LP checkpoint at {save_path}. Resuming...")
        checkpoint = torch.load(save_path, map_location=device, weights_only=False)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_mrr = checkpoint.get('best_val_mrr', -1.0)
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        history = checkpoint.get('history', [])
        best_state = checkpoint.get('best_state')
        # Only a genuine patience trigger should permanently block further
        # training -- merely reaching the epoch budget shouldn't, so a later
        # resume call with a higher `epochs` can still continue.
        early_stopped = checkpoint.get('early_stopped', False)
        already_completed = early_stopped or start_epoch >= epochs
        if early_stopped:
            print(f"Checkpoint early-stopped at epoch {checkpoint['epoch']} "
                  f"(best_val_mrr={best_val_mrr:.4f}, patience triggered) -- "
                  f"stays done regardless of `epochs`; restoring its best weights.")
        elif already_completed:
            print(f"Checkpoint already reached epoch {checkpoint['epoch']}, which "
                  f"meets or exceeds the requested epochs={epochs} -- nothing more "
                  f"to train; restoring its best weights.")
        else:
            print(f"Resuming from epoch {start_epoch} (best_val_mrr so far: "
                  f"{best_val_mrr:.4f}) -- training up to epochs={epochs}.")

    epochs_to_run = () if already_completed else range(start_epoch, epochs)
    try:
        for epoch in epochs_to_run:
            encoder.train()
            decoder.train()
            epoch_loss, epoch_batches = 0.0, 0

            for fwd_id, (rel_name, triples) in train_by_fwd_id.items():
                neg_t = sample_negatives(
                    triples[:, 0], candidate_pools[rel_name],
                    true_tails_by_relation[rel_name], k=neg_per_pos,
                )
                h_rep = triples[:, 0].repeat_interleave(neg_per_pos)
                edge_label_index = torch.cat([triples.t(), torch.stack([h_rep, neg_t])], dim=1)
                edge_label = torch.cat([
                    torch.ones(triples.size(0)), torch.zeros(h_rep.size(0)),
                ])

                loader = LinkNeighborLoader(
                    sub_graph_data, num_neighbors=list(num_neighbors), batch_size=batch_size,
                    edge_label_index=edge_label_index, edge_label=edge_label,
                    neg_sampling_ratio=0.0, num_workers=0, shuffle=True,
                )

                try:
                    for batch in loader:
                        batch = batch.to(device)
                        optimizer.zero_grad()
                        node_emb = encoder(batch.n_id_to_vec_row, batch.edge_index,
                                            batch.edge_type, batch.degree_feat)

                        h_local, t_local = batch.edge_label_index
                        r_batch = torch.full_like(h_local, fwd_id)
                        scores = decoder(node_emb[h_local], r_batch, node_emb[t_local])
                        loss = loss_fn(scores, batch.edge_label)

                        loss.backward()
                        # Same guard as gnn_tools.train_model -- prevents one
                        # bad mini-batch's gradient from collapsing the model
                        # into an uninformative ~50/50 score.
                        torch.nn.utils.clip_grad_norm_(
                            list(encoder.parameters()) + list(decoder.parameters()), max_norm=1.0,
                        )
                        optimizer.step()
                        epoch_loss += loss.item()
                        epoch_batches += 1
                finally:
                    del loader
                    cleanup()

            avg_loss = epoch_loss / max(epoch_batches, 1)
            record = {
                'epoch': epoch, 'train_loss': avg_loss,
                'val_mrr': np.nan, 'val_hits@1': np.nan, 'val_hits@3': np.nan, 'val_hits@10': np.nan,
            }

            if (epoch + 1) % eval_frequency == 0 or epoch == epochs - 1:
                score_fn = make_gnn_score_fn(
                    encoder, decoder, sub_graph_data, eval_val_triples_by_relation, candidate_pools,
                    num_neighbors, batch_size=batch_size, device=device,
                )
                val_metrics = evaluate_link_prediction(
                    score_fn, eval_val_triples_by_relation, candidate_pools,
                    true_tails_by_relation, relation_fwd_ids,
                )
                val_mrr = val_metrics['pooled']['mrr']
                record.update({
                    'val_mrr': val_mrr,
                    'val_hits@1': val_metrics['pooled'].get('hits@1', np.nan),
                    'val_hits@3': val_metrics['pooled'].get('hits@3', np.nan),
                    'val_hits@10': val_metrics['pooled'].get('hits@10', np.nan),
                })
                # Per-relation MRR alongside the pooled figure, since pool
                # size varies drastically by relation and 'pooled' alone can
                # mask real learning on the easier, smaller-pool relations.
                per_rel_bits = []
                per_rel_mrrs = []
                for rel_name, m in val_metrics.items():
                    if rel_name == 'pooled' or m.get('status') != 'ok':
                        continue
                    record[f'val_mrr_{rel_name}'] = m['mrr']
                    per_rel_mrrs.append(m['mrr'])
                    per_rel_bits.append(f"{rel_name}={m['mrr']:.4f} (pool={m['candidate_pool_size']})")
                per_rel_str = " | ".join(per_rel_bits)
                macro_mrr = float(np.mean(per_rel_mrrs)) if per_rel_mrrs else float('nan')
                record['val_mrr_macro'] = macro_mrr
                selection_mrr = macro_mrr if monitor == 'macro' and not np.isnan(macro_mrr) else val_mrr
                print(f"Epoch {epoch:03d} | Train Loss: {avg_loss:.4f} | "
                      f"Val MRR pooled={val_mrr:.4f} macro={macro_mrr:.4f} (monitor={monitor}) | {per_rel_str}")

                should_stop = False
                if selection_mrr > best_val_mrr:
                    best_val_mrr = selection_mrr
                    epochs_no_improve = 0
                    best_state = {
                        'encoder': {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                        'decoder': {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()},
                    }
                else:
                    epochs_no_improve += eval_frequency
                    if epochs_no_improve >= patience:
                        print(f"Early stopping at epoch {epoch} "
                              f"(no Val MRR improvement in {patience} epochs).")
                        should_stop = True
            else:
                print(f"Epoch {epoch:03d} | Train Loss: {avg_loss:.4f}")
                should_stop = False

            history.append(record)

            # Save on every eval checkpoint, not just improvements (see
            # save_path/resume docstring).
            if save_path and not np.isnan(record.get('val_mrr', np.nan)):
                reached_epoch_limit = epoch == epochs - 1
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_mrr': best_val_mrr,
                    'epochs_no_improve': epochs_no_improve,
                    'best_state': best_state,
                    'history': history,
                    'early_stopped': should_stop,
                }, save_path)
                status_str = (' (early stopped)' if should_stop else
                              ' (reached epoch limit)' if reached_epoch_limit else '')
                print(f"  -> LP checkpoint saved to {save_path} (epoch {epoch}{status_str}).")

            if should_stop:
                break

        if best_state is not None:
            encoder.load_state_dict(best_state['encoder'])
            decoder.load_state_dict(best_state['decoder'])

        return encoder.cpu(), decoder.cpu(), best_val_mrr, history
    finally:
        if 'optimizer' in locals():
            del optimizer
        cleanup()


# ─────────────────────────────────────────────────────────────────────────
# Filtered, type-constrained ranking evaluation (model-agnostic)
# ─────────────────────────────────────────────────────────────────────────

def evaluate_link_prediction(score_fn, test_triples_by_relation, candidate_pools,
                              true_tails_by_relation, relation_fwd_ids, ks=(1, 3, 10), seed=42):
    """
    Model-agnostic filtered, type-constrained ranking evaluation -- works
    identically for RGCN/GraphSAGE (score_fn from cached GNN embeddings)
    and TransE (score_fn from trained pykeen embeddings), so all three
    models are scored by the exact same metric definition.

    For each held-out test triple (h, r, t): candidate pool = this
    relation's observed tails, minus h's other true tails (the 'filtered'
    setting), plus the true t if not already present. Ranks t among those
    candidates (average of optimistic/pessimistic rank on ties).

    Returns {relation_name: {mrr, hits@1, hits@3, hits@10, num_test_triples,
    candidate_pool_size, auc, ap, f1, status}, ..., 'pooled': {...}}.
    'pooled' is a triple-count-weighted average, reported only as a
    secondary summary -- candidate-pool sizes vary drastically by relation,
    so prefer the per-relation breakdown when writing this up.
    """
    rng = np.random.default_rng(seed)
    results = {}
    all_ranks = []

    for rel_name, triples in test_triples_by_relation.items():
        pool = candidate_pools.get(rel_name)
        if triples is None or triples.size(0) == 0 or pool is None or pool.size(0) == 0:
            results[rel_name] = {'status': 'skipped', 'num_test_triples': 0}
            continue

        fwd_id = relation_fwd_ids[rel_name]
        true_tails = true_tails_by_relation.get(rel_name, {})
        pool_np = pool.numpy()

        ranks = []
        pos_scores_rel, neg_scores_rel = [], []
        tie_count = 0  # test triples where >=1 OTHER candidate exactly ties the true score

        for h, t in triples.tolist():
            excluded = true_tails.get(h, set()) - {t}
            if excluded:
                excl_arr = np.fromiter(excluded, dtype=np.int64)
                filtered_pool = pool_np[~np.isin(pool_np, excl_arr)]
            else:
                filtered_pool = pool_np
            if t not in filtered_pool:
                filtered_pool = np.concatenate([filtered_pool, [t]])

            cand_t = torch.from_numpy(filtered_pool).long()
            cand_h = torch.full_like(cand_t, h)
            cand_r = torch.full_like(cand_t, fwd_id)
            scores = score_fn(cand_h, cand_r, cand_t)

            true_idx = int((cand_t == t).nonzero(as_tuple=True)[0][0])
            true_score = scores[true_idx]
            # Average of optimistic/pessimistic rank (standard KG-benchmark
            # convention) -- pure optimistic ranking lets a degenerate,
            # non-discriminating score_fn report a "perfect" rank=1 via ties.
            optimistic_rank = int((scores > true_score).sum().item()) + 1
            pessimistic_rank = int((scores >= true_score).sum().item())
            rank = (optimistic_rank + pessimistic_rank) / 2.0
            ranks.append(rank)
            if int((scores == true_score).sum().item()) > 1:
                tie_count += 1

            # Reuse `scores` (already computed above) for the AUC/AP negative
            # instead of a second score_fn call, so pos/neg scores can't
            # drift apart via two different code paths.
            neg_idx_choices = np.where(filtered_pool != t)[0]
            if len(neg_idx_choices) > 0:
                neg_idx = int(rng.choice(neg_idx_choices))
                pos_scores_rel.append(float(scores[true_idx]))
                neg_scores_rel.append(float(scores[neg_idx]))

        ranks_arr = np.array(ranks)
        metrics = {
            'mrr': float(np.mean(1.0 / ranks_arr)),
            **{f'hits@{k}': float(np.mean(ranks_arr <= k)) for k in ks},
            'num_test_triples': int(len(ranks_arr)),
            'candidate_pool_size': int(pool.size(0)),
            # Exact-rank breakdown (not cumulative like hits@k) + tie rate --
            # diagnostic for patterns like "hits@1 near 0 but hits@3 high",
            # which would show up here as mass concentrated at rank 2/3
            # rather than spread out, and/or a high tie_rate if near-ties
            # (not exact ties, which the strict '>' rank convention already
            # doesn't penalize) are pushing the true answer down narrowly.
            'pct_rank_1': float(np.mean(ranks_arr == 1)),
            'pct_rank_2': float(np.mean(ranks_arr == 2)),
            'pct_rank_3': float(np.mean(ranks_arr == 3)),
            'tie_rate': float(tie_count / len(ranks_arr)),
            'status': 'ok',
        }

        if pos_scores_rel:
            y_true = [1] * len(pos_scores_rel) + [0] * len(neg_scores_rel)
            y_score = pos_scores_rel + neg_scores_rel
            try:
                metrics['auc'] = roc_auc_score(y_true, y_score)
                metrics['ap'] = average_precision_score(y_true, y_score)
                median_score = float(np.median(y_score))
                y_pred = [1 if s >= median_score else 0 for s in y_score]
                metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
            except ValueError:
                pass  # degenerate (e.g. all-one-class) -- leave auc/ap/f1 unset for this relation

        results[rel_name] = metrics
        all_ranks.extend(ranks)

    if all_ranks:
        ranks_arr = np.array(all_ranks)
        results['pooled'] = {
            'mrr': float(np.mean(1.0 / ranks_arr)),
            **{f'hits@{k}': float(np.mean(ranks_arr <= k)) for k in ks},
            'num_test_triples': int(len(ranks_arr)),
        }
    else:
        results['pooled'] = {'mrr': 0.0, 'num_test_triples': 0}

    return results


# ─────────────────────────────────────────────────────────────────────────
# Small, justified validation search
# ─────────────────────────────────────────────────────────────────────────

def run_small_lp_validation_search(model_name, encoder_factory, decoder_factory,
                                    sub_graph_data,
                                    train_triples_by_relation, val_triples_by_relation,
                                    test_triples_by_relation,
                                    candidate_pools, true_tails_by_relation, relation_fwd_ids,
                                    dim_grid, fixed_hparams, csv_path, schema_label, seed=42,
                                    checkpoint_dir=None, resume=False, batch_sizes=None,
                                    max_val_eval_triples=None):
    """
    Small, justified validation search: sweeps ONLY hidden/embedding
    dimension (dim_grid) with every other hyperparameter anchored on
    `fixed_hparams` (the already-tuned node-classification winner for this
    model+schema). Evaluates each on validation MRR, keeps the best, runs
    one final test evaluation. Deliberately not a full multi-phase sweep --
    this is a supporting representation-quality experiment, not the main
    predictive benchmark.

    encoder_factory(hidden_dim) -> nn.Module; decoder_factory(hidden_dim) ->
        DistMultDecoder. Architecture-specific args already bound via
        closure by the caller.
    seed: reseeds torch/numpy before each dim's factory calls and is also
        passed into train_link_prediction_model, isolating hidden_dim's
        effect from init/training-dynamics randomness.
    checkpoint_dir/resume: per-dim checkpoint/resume via
        train_link_prediction_model's save_path/resume -- resume=True picks
        up each dim from wherever it left off, or skips a finished one.
    max_val_eval_triples: forwarded to train_link_prediction_model -- only
        affects in-training checks, not the final test evaluation.
    fixed_hparams['monitor']: forwarded to train_link_prediction_model, also
        determines this function's own best_dim selection.
    batch_sizes: OOM-retry ladder (same pattern as
        gnn_tools.run_ablation_run/run_k_sweep) -- retries the same dim at
        a smaller batch size on CUDA OOM. Defaults to
        (fixed_hparams.get('batch_size', 1024), 512, 256, 128, 64).

    Returns (final_encoder, final_decoder, test_metrics, results_df, best_dim).
    """
    num_neighbors = fixed_hparams['neighbors']
    if batch_sizes is None:
        configured_bs = fixed_hparams.get('batch_size', 1024)
        batch_sizes = tuple(bs for bs in (configured_bs, 512, 256, 128, 64) if bs <= configured_bs)
        if not batch_sizes:
            batch_sizes = (configured_bs,)
    rows = []
    best_val_mrr, best_dim, best_state, best_history = -1.0, None, None, None

    for dim in dim_grid:
        print(f"\n[LP VALIDATION SEARCH] {model_name}/{schema_label} | hidden_dim={dim}")
        t0 = time.time()
        dim_save_path = (f"{checkpoint_dir.rstrip('/')}/{model_name}_{schema_label}_dim{dim}_lp_checkpoint.pt"
                          if checkpoint_dir else None)
        encoder, decoder = None, None  # set before try/finally -- encoder_factory(dim) itself
                                        # can be what fails, and `finally` below must not crash
                                        # trying to del a name that was never bound (this masked
                                        # the real exception with an UnboundLocalError before)
        try:
            for i, attempt_batch_size in enumerate(batch_sizes):
                torch.manual_seed(seed)
                np.random.seed(seed)
                encoder = encoder_factory(dim)
                decoder = decoder_factory(dim)
                try:
                    encoder, decoder, val_mrr, history = train_link_prediction_model(
                        encoder, decoder, sub_graph_data,
                        train_triples_by_relation, val_triples_by_relation,
                        candidate_pools, true_tails_by_relation, relation_fwd_ids,
                        num_neighbors=num_neighbors,
                        epochs=fixed_hparams.get('epochs', 100),
                        lr=fixed_hparams.get('lr', 1e-3),
                        weight_decay=fixed_hparams.get('weight_decay', 1e-4),
                        patience=fixed_hparams.get('patience', 15),
                        eval_frequency=fixed_hparams.get('eval_frequency', 1),
                        batch_size=attempt_batch_size,
                        neg_per_pos=fixed_hparams.get('neg_per_pos', 1),
                        decoder_weight_decay=fixed_hparams.get('decoder_weight_decay'),
                        seed=seed,
                        save_path=dim_save_path,
                        resume=resume,
                        monitor=fixed_hparams.get('monitor', 'pooled'),
                        max_val_eval_triples=max_val_eval_triples,
                    )
                    break
                except RuntimeError as oom_err:
                    if "out of memory" not in str(oom_err).lower() or i == len(batch_sizes) - 1:
                        raise
                    print(f"  -> CUDA OOM at batch_size={attempt_batch_size}, retrying at "
                          f"batch_size={batch_sizes[i + 1]}...")
                    # oom_err's traceback keeps every frame in the failed call
                    # chain alive (including the GPU-resident encoder/decoder)
                    # for as long as it stays bound -- del before cleanup() so
                    # gc.collect()/empty_cache() can actually reclaim it. Same
                    # reasoning as gnn_tools.run_ablation_run's retry ladder.
                    del oom_err, encoder, decoder
                    cleanup()

            elapsed = time.time() - t0
            rows.append({'model': model_name, 'schema': schema_label, 'phase': 'search',
                         'hidden_dim': dim, 'val_mrr': val_mrr, 'runtime_sec': elapsed,
                         'status': 'ok'})
            if val_mrr > best_val_mrr:
                best_val_mrr, best_dim, best_history = val_mrr, dim, history
                best_state = {
                    'encoder': {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                    'decoder': {k: v.cpu().clone() for k, v in decoder.state_dict().items()},
                }
        except RuntimeError as e:
            print(f"  -> FAILED at hidden_dim={dim}: {e}")
            rows.append({'model': model_name, 'schema': schema_label, 'phase': 'search',
                         'hidden_dim': dim, 'val_mrr': np.nan, 'runtime_sec': np.nan,
                         'status': 'failed'})
        finally:
            if encoder is not None:
                del encoder
            if decoder is not None:
                del decoder
            cleanup()

        pd.DataFrame(rows).to_csv(csv_path, index=False)

    if best_dim is None:
        raise RuntimeError(f"All hidden_dim configs failed for {model_name}/{schema_label} -- see {csv_path}.")

    print(f"\n>>> Best hidden_dim for {model_name}/{schema_label}: {best_dim} "
          f"(val MRR: {best_val_mrr:.4f})")

    history_path = csv_path.replace('.csv', '_training_history.csv')
    history_df = pd.DataFrame(best_history)
    history_df.insert(0, 'hidden_dim', best_dim)
    history_df.to_csv(history_path, index=False)
    print(f"Saved per-epoch training history for the winning hidden_dim={best_dim}: {history_path}")

    final_encoder = encoder_factory(best_dim)
    final_decoder = decoder_factory(best_dim)
    final_encoder.load_state_dict(best_state['encoder'])
    final_decoder.load_state_dict(best_state['decoder'])

    test_score_fn = make_gnn_score_fn(
        final_encoder, final_decoder, sub_graph_data, test_triples_by_relation, candidate_pools,
        num_neighbors, batch_size=fixed_hparams.get('batch_size', 1024),
    )
    test_metrics = evaluate_link_prediction(
        test_score_fn, test_triples_by_relation, candidate_pools,
        true_tails_by_relation, relation_fwd_ids,
    )

    rows.append({'model': model_name, 'schema': schema_label, 'phase': 'final_test',
                 'hidden_dim': best_dim, 'val_mrr': best_val_mrr,
                 'test_mrr_pooled': test_metrics['pooled']['mrr'], 'status': 'ok'})
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return final_encoder, final_decoder, test_metrics, pd.DataFrame(rows), best_dim


# ─────────────────────────────────────────────────────────────────────────
# Small, sequential FOLLOW-UP search (additive LP tuning extension)
# ─────────────────────────────────────────────────────────────────────────

def run_small_lp_secondary_search(model_name, param_name, param_grid, dim,
                                   encoder_factory, decoder_factory,
                                   sub_graph_data,
                                   train_triples_by_relation, val_triples_by_relation,
                                   test_triples_by_relation,
                                   candidate_pools, true_tails_by_relation, relation_fwd_ids,
                                   fixed_hparams, csv_path, schema_label, seed=42,
                                   checkpoint_dir=None, resume=False, batch_sizes=None,
                                   max_val_eval_triples=None, rebuild_encoder=False):
    """
    Small, sequential FOLLOW-UP search over exactly ONE additional
    hyperparameter, run AFTER run_small_lp_validation_search has already
    picked `dim` -- e.g. num_bases (RGCN only) or neg_per_pos. Keeps the
    search sequential/greedy (dim -> num_bases -> neg_per_pos, each locking
    in a winner before the next) rather than a full cross-product grid.
    Mirrors run_small_lp_validation_search's OOM-retry/checkpoint/final-test
    structure, keyed on an arbitrary param_name instead of hidden_dim.

    rebuild_encoder: False (default) when param_name doesn't change the
    encoder's construction -- encoder_factory(dim) is called normally. True
    when param_name changes how the encoder is built (e.g. num_bases) --
    encoder_factory must then accept it as a keyword, e.g.
    encoder_factory(dim, num_bases=value).

    fixed_hparams: same anchored dict as run_small_lp_validation_search,
    holding every hyperparameter except dim and param_name.

    Returns (final_encoder, final_decoder, test_metrics, results_df,
    best_value) -- same shape as run_small_lp_validation_search's return,
    with best_value replacing best_dim.
    """
    num_neighbors = fixed_hparams['neighbors']
    if batch_sizes is None:
        configured_bs = fixed_hparams.get('batch_size', 1024)
        batch_sizes = tuple(bs for bs in (configured_bs, 512, 256, 128, 64) if bs <= configured_bs)
        if not batch_sizes:
            batch_sizes = (configured_bs,)

    rows = []
    best_val_mrr, best_value, best_state, best_history = -1.0, None, None, None

    for value in param_grid:
        print(f"\n[LP SECONDARY SEARCH] {model_name}/{schema_label} | "
              f"{param_name}={value} (dim={dim} fixed)")
        t0 = time.time()
        value_save_path = (f"{checkpoint_dir.rstrip('/')}/{model_name}_{schema_label}_dim{dim}_"
                            f"{param_name}{value}_lp_checkpoint.pt" if checkpoint_dir else None)
        call_hparams = dict(fixed_hparams)
        call_hparams[param_name] = value
        encoder, decoder = None, None  # set before try/finally -- see the matching comment in
                                        # run_small_lp_validation_search for why this is required
        try:
            for i, attempt_batch_size in enumerate(batch_sizes):
                torch.manual_seed(seed)
                np.random.seed(seed)
                encoder = (encoder_factory(dim, **{param_name: value}) if rebuild_encoder
                           else encoder_factory(dim))
                decoder = decoder_factory(dim)
                try:
                    encoder, decoder, val_mrr, history = train_link_prediction_model(
                        encoder, decoder, sub_graph_data,
                        train_triples_by_relation, val_triples_by_relation,
                        candidate_pools, true_tails_by_relation, relation_fwd_ids,
                        num_neighbors=num_neighbors,
                        epochs=call_hparams.get('epochs', 100),
                        lr=call_hparams.get('lr', 1e-3),
                        weight_decay=call_hparams.get('weight_decay', 1e-4),
                        patience=call_hparams.get('patience', 15),
                        eval_frequency=call_hparams.get('eval_frequency', 1),
                        batch_size=attempt_batch_size,
                        neg_per_pos=call_hparams.get('neg_per_pos', 1),
                        decoder_weight_decay=call_hparams.get('decoder_weight_decay'),
                        seed=seed,
                        save_path=value_save_path,
                        resume=resume,
                        monitor=call_hparams.get('monitor', 'pooled'),
                        max_val_eval_triples=max_val_eval_triples,
                    )
                    break
                except RuntimeError as oom_err:
                    if "out of memory" not in str(oom_err).lower() or i == len(batch_sizes) - 1:
                        raise
                    print(f"  -> CUDA OOM at batch_size={attempt_batch_size}, retrying at "
                          f"batch_size={batch_sizes[i + 1]}...")
                    del oom_err, encoder, decoder
                    cleanup()

            elapsed = time.time() - t0
            rows.append({'model': model_name, 'schema': schema_label, 'phase': f'secondary_{param_name}',
                         param_name: value, 'dim': dim, 'val_mrr': val_mrr, 'runtime_sec': elapsed,
                         'status': 'ok'})
            if val_mrr > best_val_mrr:
                best_val_mrr, best_value, best_history = val_mrr, value, history
                best_state = {
                    'encoder': {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                    'decoder': {k: v.cpu().clone() for k, v in decoder.state_dict().items()},
                }
        except RuntimeError as e:
            print(f"  -> FAILED at {param_name}={value}: {e}")
            rows.append({'model': model_name, 'schema': schema_label, 'phase': f'secondary_{param_name}',
                         param_name: value, 'dim': dim, 'val_mrr': np.nan, 'runtime_sec': np.nan,
                         'status': 'failed'})
        finally:
            if encoder is not None:
                del encoder
            if decoder is not None:
                del decoder
            cleanup()

        pd.DataFrame(rows).to_csv(csv_path, index=False)

    if best_value is None:
        raise RuntimeError(f"All {param_name} configs failed for {model_name}/{schema_label} -- see {csv_path}.")

    print(f"\n>>> Best {param_name} for {model_name}/{schema_label} (dim={dim}): {best_value} "
          f"(val MRR: {best_val_mrr:.4f})")

    history_path = csv_path.replace('.csv', '_training_history.csv')
    history_df = pd.DataFrame(best_history)
    history_df.insert(0, param_name, best_value)
    history_df.to_csv(history_path, index=False)
    print(f"Saved per-epoch training history for the winning {param_name}={best_value}: {history_path}")

    final_encoder = (encoder_factory(dim, **{param_name: best_value}) if rebuild_encoder
                      else encoder_factory(dim))
    final_decoder = decoder_factory(dim)
    final_encoder.load_state_dict(best_state['encoder'])
    final_decoder.load_state_dict(best_state['decoder'])

    test_score_fn = make_gnn_score_fn(
        final_encoder, final_decoder, sub_graph_data, test_triples_by_relation, candidate_pools,
        num_neighbors, batch_size=fixed_hparams.get('batch_size', 1024),
    )
    test_metrics = evaluate_link_prediction(
        test_score_fn, test_triples_by_relation, candidate_pools,
        true_tails_by_relation, relation_fwd_ids,
    )

    rows.append({'model': model_name, 'schema': schema_label, 'phase': 'final_test',
                 param_name: best_value, 'dim': dim, 'val_mrr': best_val_mrr,
                 'test_mrr_pooled': test_metrics['pooled']['mrr'], 'status': 'ok'})
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return final_encoder, final_decoder, test_metrics, pd.DataFrame(rows), best_value


# ─────────────────────────────────────────────────────────────────────────
# Post-run sanity check (shared by all three model notebooks)
# ─────────────────────────────────────────────────────────────────────────

def quick_sanity_check(results_path, history_path, model_name, schema_label, dim_col, expected_dim):
    """
    Minimal post-run check on the just-written {model}_{schema}_lp_results.csv
    (and, if given, _lp_training_history.csv -- TransE doesn't produce one,
    so pass history_path=None there). Catches obviously-wrong recorded data
    (wrong model/schema/dim stamped on a row, out-of-range metrics,
    non-monotonic Hits@k, missing pooled row, empty/NaN triple or pool
    counts, no validation ever recorded) before you trust the numbers.
    Prints PASS/FAIL, doesn't raise. Returns the list of issues found (empty
    if none).
    """
    issues = []
    results_df = pd.read_csv(results_path)

    if results_df.empty:
        issues.append(f"{results_path} is empty")
    else:
        if not (results_df['model'] == model_name).all():
            issues.append("results: unexpected value in 'model' column")
        if not (results_df['schema'] == schema_label).all():
            issues.append("results: unexpected value in 'schema' column")
        if dim_col in results_df.columns and not (results_df[dim_col] == expected_dim).all():
            issues.append(f"results: '{dim_col}' doesn't match expected_dim={expected_dim} on every row")
        if 'pooled' not in results_df['relation'].values:
            issues.append("results: no 'pooled' row found")

        per_rel = results_df[results_df['relation'] != 'pooled']
        for col in ('mrr', 'hits@1', 'hits@3', 'hits@10'):
            if col in per_rel.columns:
                bad = per_rel[(per_rel[col] < 0) | (per_rel[col] > 1)]
                if len(bad) > 0:
                    issues.append(f"results: '{col}' outside [0,1] for relations {list(bad['relation'])}")
        if {'hits@1', 'hits@3', 'hits@10'}.issubset(per_rel.columns):
            non_monotonic = per_rel[(per_rel['hits@1'] > per_rel['hits@3'] + 1e-9) |
                                     (per_rel['hits@3'] > per_rel['hits@10'] + 1e-9)]
            if len(non_monotonic) > 0:
                issues.append(f"results: Hits@k not monotonic (1<=3<=10) for relations {list(non_monotonic['relation'])}")
        for col in ('num_test_triples', 'candidate_pool_size'):
            if col in per_rel.columns:
                bad = per_rel[per_rel[col].isna() | (per_rel[col] <= 0)]
                if len(bad) > 0:
                    issues.append(f"results: '{col}' missing/non-positive for relations {list(bad['relation'])}")

    if history_path is not None:
        history_df = pd.read_csv(history_path)
        if history_df.empty:
            issues.append(f"{history_path} is empty")
        else:
            if 'hidden_dim' in history_df.columns and not (history_df['hidden_dim'] == expected_dim).all():
                issues.append("history: 'hidden_dim' doesn't match expected_dim on every row")
            if history_df['train_loss'].isna().any():
                issues.append("history: NaN train_loss recorded")
            if 'val_mrr' in history_df.columns and history_df['val_mrr'].notna().sum() == 0:
                issues.append("history: no validation MRR was ever recorded (eval_frequency never triggered?)")

    if issues:
        print(f"[SANITY CHECK] {model_name}/{schema_label} -- {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"[SANITY CHECK] {model_name}/{schema_label} -- all checks passed.")
    return issues
