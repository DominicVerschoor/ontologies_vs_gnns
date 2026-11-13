"""
Link prediction on the F1 KGs, identically for all three KGs (Ad-hoc, Pure SULO, Hybrid SULO).

Three target queries (h, r, ?), each a domain-level fact read from the assertions, never a KG
predicate, so Pure SULO's shared predicates (sulo:hasFeature, the hasParticipant -> Role ->
isFeatureOf chain) change how a fact is spelled in the graph but not what is asked:
  team      RaceEntry -> Team      (hasTeam)
  position  RaceEntry -> Position  (hasStartPosition, the grid slot)
  circuit   Race      -> Circuit   (heldAtCircuit, Hybrid: isInCircuit)
Only tail queries are scored: heads are unique RaceEntry/Race nodes. The candidates of a query
are all nodes of its answer class (182 Teams, 34 Positions, 78 Circuits); every target is
functional, so the filtered and the raw ranking coincide.

One joint model per KG x encoder: a node_prediction.models encoder (whole graph, full-batch)
and a DistMult decoder with one vector per target, trained with 1-vs-all cross-entropy over the
candidates (Dettmers et al. 2018; Ruffinelli et al. 2020), every query weighted equally.

Holdout is per race: a val/test race loses the facts of all three targets from message
passing (grid slots are a permutation within a race, teammates would reveal the team).
hasFinishPosition is removed for every entry, because in Pure SULO it is the same
sulo:hasFeature -> Position/<n> triple as the start position whenever the two are equal.
hasTeamMember (Hybrid/Pure) pairs are visible only while one of the entries behind them shows
its team, so a held-out team never survives as a membership. During training, each epoch hides
the facts of a random share of the training races and supervises exactly those, so training
queries look like test queries (PyG RandomLinkSplit's disjoint_train_ratio).
"""
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, fields, replace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from f1_node_data import build_graph, degree_feature, node_features
from f1_population import NS, SCHEMA_FILES, assertion_lines, read_schema, render_ntriples
from node_prediction.gnn_tools import _config_of, _fmt, _grad_norm, cleanup, set_seed

# target -> (head class, properties stating the fact, answer class)
TARGETS = {
    "team": ("RaceEntry", frozenset({"hasTeam"}), "Team"),
    "position": ("RaceEntry", frozenset({"hasStartPosition"}), "Position"),
    "circuit": ("Race", frozenset({"heldAtCircuit", "isInCircuit"}), "Circuit"),
}
MASKED_PROPERTIES = frozenset({"hasFinishPosition"})
MEMBER_PROPERTY = "hasTeamMember"
SPLITS = ("train", "val", "test")
KS = (1, 3, 10)


def race_split(entities: pd.DataFrame, seed=42) -> tuple[list[str], list[str], list[str]]:
    """Random 80/10/10 split of all Race IRIs. Shared by all KGs and both encoders."""
    races = sorted(entities.loc[entities["class"] == "Race", "iri"])
    train, rest = train_test_split(races, test_size=0.2, random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, random_state=seed)
    return train, val, test


def target_facts(assertions: pd.DataFrame, schema) -> dict[str, pd.DataFrame]:
    """target -> (subject, subject_class, property, value, race) of its facts, in the property the
    schema renders (Ad-hoc heldAtCircuit, Hybrid isInCircuit); one fact per head."""
    entry_race = assertions[assertions["property"] == "hasRaceEntry"].set_index("value")["subject"]
    facts = {}
    for name, (cls, props, _) in TARGETS.items():
        rows = assertions[(assertions["subject_class"] == cls) & assertions["property"].isin(props)
                          & assertions["property"].isin(schema.properties.keys())]
        rows = rows.sort_values("subject", ignore_index=True)
        assert rows["subject"].is_unique, f"{name}: more than one fact per head"
        race = rows["subject"].map(entry_race) if cls == "RaceEntry" else rows["subject"]
        assert race.notna().all(), f"{name}: a head without a race"
        facts[name] = rows.assign(race=race.values)
    return facts


@dataclass
class TargetData:
    head: torch.Tensor        # node id per fact
    answer: torch.Tensor      # index of the true tail in `candidates`, per fact
    race: torch.Tensor        # race index per fact (into LPData.races)
    candidates: torch.Tensor  # node ids of every node of the answer class, sorted by IRI
    edge_fact: torch.Tensor   # (2, E): edge column, fact index -- the edges of each fact's path


@dataclass
class LPData:
    data: Data                    # the whole graph: edge_index, edge_type, key_id, value
    targets: dict[str, TargetData]
    race_split: torch.Tensor      # per race index: 0 train, 1 val, 2 test
    member_edges: torch.Tensor    # (2, E): edge column, hasTeamMember pair index
    member_support: torch.Tensor  # (2, S): pair index, team-fact index of an entry behind the pair
    num_relations: int            # edge types, forward + inverse
    vocabulary: list[str]         # feature keys; models need len(vocabulary)
    node_ids: dict[str, int]      # N-Triples term -> node id
    races: list[str]              # race index -> IRI

    def to(self, device):
        """A copy with every tensor on `device` (Data.to moves in place, so the graph is cloned)."""
        move = lambda obj: replace(obj, **{f.name: getattr(obj, f.name).to(device) for f in fields(obj)
                                           if isinstance(getattr(obj, f.name), torch.Tensor)})
        return replace(move(self), data=self.data.clone().to(device),
                       targets={name: move(t) for name, t in self.targets.items()})


def build_lp_data(kg: str, entities: pd.DataFrame, assertions: pd.DataFrame,
                  split: tuple[list[str], list[str], list[str]], schema=None) -> LPData:
    """The KG without hasFinishPosition, its target facts, their path edges and the race split."""
    schema = schema or read_schema(SCHEMA_FILES[kg])
    kept = assertions[~assertions["property"].isin(MASKED_PROPERTIES)]
    lines, _ = render_ntriples(entities, kept, schema)
    edge_index, edge_type, node_ids, relation_ids = build_graph(lines)
    key_id, value, vocabulary = node_features(lines, node_ids)
    line_index = {line: i for i, line in enumerate(lines)}   # build_graph: line i -> edges 2i, 2i + 1
    rendered = kept[kept["property"].isin(schema.properties.keys())]
    owners = Counter(line for row in rendered.itertuples(index=False)
                     for line in assertion_lines(*row, schema.properties[row.property]))
    node = lambda iri: node_ids[f"<{iri}>"]

    def path_edges(rows):
        """(2, E) edge column, row index of every edge on the rows' paths; each line must be theirs alone."""
        cols, owner = [], []
        for i, row in enumerate(rows.itertuples(index=False)):
            for line in assertion_lines(row.subject, row.subject_class, row.property, row.value,
                                        schema.properties[row.property]):
                assert owners[line] == 1, f"{line} also states another fact, so hiding it would hide that too"
                cols += [2 * line_index[line], 2 * line_index[line] + 1]
                owner += [i, i]
        return torch.tensor([cols, owner], dtype=torch.long).reshape(2, -1)

    races = [r for part in split for r in part]
    race_ids = {r: i for i, r in enumerate(races)}
    race_split = torch.tensor([i for i, part in enumerate(split) for _ in part])
    facts = target_facts(kept, schema)
    targets = {}
    for name, rows in facts.items():
        answer_iris = sorted(entities.loc[entities["class"] == TARGETS[name][2], "iri"])
        answer_ids = {iri: i for i, iri in enumerate(answer_iris)}
        targets[name] = TargetData(
            head=torch.tensor([node(s) for s in rows["subject"]]),
            answer=torch.tensor([answer_ids[v] for v in rows["value"]]),
            race=torch.tensor([race_ids[r] for r in rows["race"]]),
            candidates=torch.tensor([node(iri) for iri in answer_iris]),
            edge_fact=path_edges(rows),
        )

    # hasTeamMember pairs and the team facts of the entries behind each one (Ad-hoc has none).
    members = kept[(kept["property"] == MEMBER_PROPERTY) & (MEMBER_PROPERTY in schema.properties)]
    members = members.reset_index(drop=True)
    driver = kept[kept["property"] == "hasDriver"].set_index("subject")["value"]
    team_facts = facts["team"].assign(driver=lambda f: f["subject"].map(driver), fact=lambda f: f.index)
    support = members.reset_index().merge(team_facts, left_on=["subject", "value"],
                                          right_on=["value", "driver"])[["index", "fact"]]
    assert support["index"].nunique() == len(members), "a hasTeamMember pair without a race entry behind it"

    data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=len(node_ids), key_id=key_id, value=value)
    return LPData(data, targets, race_split, path_edges(members),
                  torch.tensor(support.T.values, dtype=torch.long).reshape(2, -1),
                  2 * len(relation_ids), vocabulary, node_ids, races)


def hidden_edges(lp: LPData, hidden_race: torch.Tensor) -> torch.Tensor:
    """
    Bool per edge column: the path edges of every fact of a hidden race, and of every
    hasTeamMember pair whose entries all have their team fact hidden.
    """
    hidden = torch.zeros(lp.data.edge_index.size(1), dtype=torch.bool, device=hidden_race.device)
    for t in lp.targets.values():
        cols, fact = t.edge_fact
        hidden[cols[hidden_race[t.race[fact]]]] = True
    if lp.member_edges.size(1):
        team_shown = (~hidden_race[lp.targets["team"].race]).float()
        pair, fact = lp.member_support
        shown = torch.zeros(int(pair.max()) + 1, device=hidden_race.device).index_add_(0, pair, team_shown[fact])
        cols, pair = lp.member_edges
        hidden[cols[shown[pair] == 0]] = True
    return hidden


def visible_graph(lp: LPData, hidden_race: torch.Tensor) -> Data:
    """The message-passing graph with hidden_race's facts removed; degree computed on what is left."""
    keep = ~hidden_edges(lp, hidden_race)
    edge_index = lp.data.edge_index[:, keep]
    return Data(edge_index=edge_index, edge_type=lp.data.edge_type[keep], num_nodes=lp.data.num_nodes,
                key_id=lp.data.key_id, value=lp.data.value,
                degree_feat=degree_feature(edge_index, lp.data.num_nodes))


class LinkPredictor(nn.Module):
    """
    A node encoder (RGCNModel / GraphSAGEModel with out_channels = dim) and a DistMult decoder,
    score(h, r, t) = <e_h, w_r, e_t> (Yang et al. 2015; the R-GCN link-prediction decoder,
    Schlichtkrull et al. 2018), with one vector w_r per target.
    """

    def __init__(self, encoder, num_targets, dim):
        super().__init__()
        self.encoder = encoder
        self.relation = nn.Parameter(torch.empty(num_targets, dim))
        nn.init.xavier_uniform_(self.relation)

    def forward(self, graph):
        return self.encoder(graph)

    def scores(self, emb, target, heads, candidates):
        """(len(heads), len(candidates)) scores of target index `target`."""
        return (emb[heads] * self.relation[target]) @ emb[candidates].T


def _query_loss(model, emb, lp, queried):
    """1-vs-all cross-entropy summed over the queried facts of every target, divided by their count."""
    total, count = 0.0, 0
    for i, t in enumerate(lp.targets.values()):
        on = queried[t.race]
        if on.any():
            total = total + F.cross_entropy(model.scores(emb, i, t.head[on], t.candidates), t.answer[on],
                                            reduction='sum')
            count += int(on.sum())
    return total / count


def rank_of_answer(scores: torch.Tensor, answer: torch.Tensor) -> torch.Tensor:
    """1-based rank of each row's true candidate; a tie counts as the mean of the optimistic and
    pessimistic rank, so equal scores give no credit."""
    true = scores.gather(1, answer.unsqueeze(1))
    higher = (scores > true).sum(1)
    ties = (scores == true).sum(1) - 1
    return higher + 1 + ties / 2


def ranking_metrics(ranks: torch.Tensor) -> dict:
    return {'mrr': (1 / ranks).mean().item(),
            **{f'hits@{k}': (ranks <= k).float().mean().item() for k in KS},
            'num_queries': len(ranks)}


def query_ranks(model, lp: LPData, split: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """
    Per target, the (head node id, rank of the true answer) of every query of `split`'s races.
    The graph hides that split's races and every later one (val: val + test; test: test), so
    the facts of the races before it are the context.
    """
    s = SPLITS.index(split)
    model.eval()
    with torch.no_grad():
        emb = model(visible_graph(lp, lp.race_split >= s))
        ranks = {}
        for i, (name, t) in enumerate(lp.targets.items()):
            on = lp.race_split[t.race] == s
            ranks[name] = (t.head[on], rank_of_answer(model.scores(emb, i, t.head[on], t.candidates), t.answer[on]))
    return ranks


def evaluate_lp(model, lp: LPData, split: str) -> dict:
    """Ranking metrics of query_ranks, per target and pooled over all queries ('pooled')."""
    ranks = {name: r for name, (_, r) in query_ranks(model, lp, split).items()}
    metrics = {name: ranking_metrics(r) for name, r in ranks.items()}
    metrics['pooled'] = ranking_metrics(torch.cat(list(ranks.values())))
    return metrics


def frequency_baseline(lp: LPData, split: str) -> dict:
    """
    evaluate_lp's metrics for ranking every candidate by how often it answers the target in the
    races before `split` (ties count as the mean rank): the best a model can do when the graph
    says nothing about a query, i.e. the chance level of the shuffled-answer canary.
    """
    s = SPLITS.index(split)
    ranks = {}
    for name, t in lp.targets.items():
        known, on = lp.race_split[t.race] < s, lp.race_split[t.race] == s
        counts = torch.bincount(t.answer[known], minlength=len(t.candidates)).float()
        ranks[name] = rank_of_answer(counts.expand(int(on.sum()), -1), t.answer[on])
    metrics = {name: ranking_metrics(r) for name, r in ranks.items()}
    metrics['pooled'] = ranking_metrics(torch.cat(list(ranks.values())))
    return metrics


def leaked_facts(lp: LPData, hidden_race: torch.Tensor) -> dict[str, int]:
    """
    Per target, the facts of hidden races still reachable in visible_graph: head and answer
    adjacent, or the fact's SULO intermediate node <head>/<property> (Pure's Role) still has an
    edge. 'team_via_member' counts hidden team facts whose answer is two hops away through the
    entry's driver, entry -> Driver -> hasTeamMember -> Team, although no entry of that driver
    shows that team in the visible graph (so the pair can only come from the hidden fact). Must be
    0 everywhere; read from the graph, independent of how hidden_edges finds a fact's edges.
    """
    graph = visible_graph(lp, hidden_race)
    n = lp.data.num_nodes
    src, dst = graph.edge_index
    linked = torch.unique(src * n + dst)
    adjacent = lambda a, b: torch.isin(a * n + b, linked)
    has_edge = torch.zeros(n, dtype=torch.bool)
    has_edge[src] = True
    iri_of = {i: iri for iri, i in lp.node_ids.items()}
    inner = lambda heads, prop: torch.tensor([lp.node_ids.get(f"{iri_of[h][:-1]}/{prop}>", -1) for h in heads.tolist()],
                                             dtype=torch.long)
    leaks = {}
    for name, t in lp.targets.items():
        on = hidden_race[t.race]
        heads, tails = t.head[on], t.candidates[t.answer[on]]
        leaks[name] = int(adjacent(heads, tails).sum())
        for prop in TARGETS[name][1]:
            node = inner(heads, prop)
            leaks[name] += int(has_edge[node[node >= 0]].sum())

    # Which team does each entry show, and which driver drives it (driver facts are never hidden)?
    team = lp.targets["team"]
    tails, driver = team.candidates[team.answer], _driver_of(lp)
    role = inner(team.head, "hasTeam")
    via_role = torch.zeros_like(tails, dtype=torch.bool)
    via_role[role >= 0] = adjacent(team.head[role >= 0], role[role >= 0]) & adjacent(role[role >= 0], tails[role >= 0])
    shown = adjacent(team.head, tails) | via_role
    shown_pairs = set(zip(tails[shown].tolist(), driver[shown].tolist()))
    unexplained = torch.tensor([pair not in shown_pairs for pair in zip(tails.tolist(), driver.tolist())])
    leaks["team_via_member"] = int((hidden_race[team.race] & adjacent(tails, driver) & unexplained).sum())
    return leaks


def _driver_of(lp: LPData) -> torch.Tensor:
    """The Driver node of every team fact's entry, read from the whole graph: the Driver next to
    the entry, or next to its <entry>/hasDriver Role node (Pure)."""
    is_driver = torch.zeros(lp.data.num_nodes, dtype=torch.bool)
    is_driver[[i for iri, i in lp.node_ids.items()     # Driver/<id>, not Pure's Driver/<id>/<property>
               if iri.startswith(f"<{NS}Driver/") and "/" not in iri[len(NS) + len("<Driver/"):]]] = True
    src, dst = lp.data.edge_index
    on = is_driver[dst]
    driver_next_to = dict(zip(src[on].tolist(), dst[on].tolist()))   # entries/Roles have one Driver neighbour
    iri_of = {i: iri for iri, i in lp.node_ids.items()}
    return torch.tensor([driver_next_to.get(h, driver_next_to.get(lp.node_ids.get(f"{iri_of[h][:-1]}/hasDriver>")))
                         for h in lp.targets["team"].head.tolist()])


def _train_step(model, lp, pool, always_hidden, disjoint_ratio, generator, optimizer):
    """
    One full-batch step: a random disjoint_ratio share of the pool races is supervised and
    hidden together with always_hidden; returns (loss, pre-step gradient norm).
    """
    races = pool.nonzero().squeeze(1)
    chosen = races[torch.randperm(len(races), generator=generator)[:round(disjoint_ratio * len(races))].to(races.device)]
    queried = torch.zeros_like(pool)
    queried[chosen] = True
    model.train()
    optimizer.zero_grad()
    emb = model(visible_graph(lp, always_hidden | queried))
    loss = _query_loss(model, emb, lp, queried)
    loss.backward()
    grad_norm = _grad_norm(model)
    optimizer.step()
    return loss.item(), grad_norm


def train_lp_model(model, lp: LPData, epochs, lr, weight_decay, patience, disjoint_ratio,
                   seed=0, start_from_epoch=0, save_path=None):
    """
    Trains on the train races and early-stops on pooled val MRR, as gnn_tools.train_model:
    patience in epochs, an early-stopping warm-up of start_from_epoch epochs (trained and logged,
    never checkpointed nor counted towards patience), best weights restored.
    seed: seeds which races are supervised in each epoch (model init/dropout: set_seed before).
    Returns (model, history); history['val_mrr_<target>'] logs every target's val MRR. save_path
    gets the best weights and the whole history once training ends.
    """
    if start_from_epoch >= epochs:
        raise ValueError(f"start_from_epoch {start_from_epoch} leaves no epoch to select from (epochs {epochs})")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    lp_dev = lp.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    pool, always_hidden = lp_dev.race_split == 0, lp_dev.race_split > 0
    history = {'train_loss': [], 'grad_norm': [], 'val_mrr': [], **{f'val_mrr_{t}': [] for t in lp.targets}}
    best, best_epoch, epochs_no_improve, best_state = -float('inf'), None, 0, None

    try:
        for epoch in range(epochs):
            loss, grad_norm = _train_step(model, lp_dev, pool, always_hidden, disjoint_ratio, generator, optimizer)
            val = evaluate_lp(model, lp_dev, 'val')
            history['train_loss'].append(loss)
            history['grad_norm'].append(grad_norm)
            history['val_mrr'].append(val['pooled']['mrr'])
            for t in lp.targets:
                history[f'val_mrr_{t}'].append(val[t]['mrr'])
            print(f"Epoch {epoch:03d} | Train Loss: {loss:.4f} | Val MRR: {val['pooled']['mrr']:.4f} ("
                  + ", ".join(f"{t} {val[t]['mrr']:.4f}" for t in lp.targets) + f") | Grad Norm: {grad_norm:.4f}")
            if epoch < start_from_epoch:
                continue
            if val['pooled']['mrr'] > best:
                best, best_epoch, epochs_no_improve = val['pooled']['mrr'], epoch, 0
                history['best_epoch'] = epoch
                # copy=True: on CPU, .cpu() would alias the live weights the next steps overwrite
                best_state = {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping: no val MRR improvement for {epochs_no_improve} epochs.")
                    break
        model.load_state_dict(best_state)
        print(f"Restored best weights (epoch {best_epoch}, val MRR {best:.4f})")
        if save_path:   # the best weights with the history of every epoch (see target_convergence)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({'epoch': best_epoch, 'model_state_dict': best_state, 'history': history}, save_path)
        return model, history
    finally:
        model.cpu()
        del lp_dev, optimizer
        cleanup()


def train_final_lp_model(model, lp: LPData, epochs, lr, weight_decay, disjoint_ratio, seed=0, save_path=None):
    """Trains on the train + val races for a fixed number of epochs (no validation, no early stopping)."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    lp_dev = lp.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    pool, always_hidden = lp_dev.race_split < 2, lp_dev.race_split == 2
    try:
        for epoch in range(epochs):
            loss, grad_norm = _train_step(model, lp_dev, pool, always_hidden, disjoint_ratio, generator, optimizer)
            print(f"Final Train Epoch {epoch:03d} | Train Loss: {loss:.4f} | Grad Norm: {grad_norm:.4f}")
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({'model_state_dict': model.state_dict()}, save_path)
        return model
    finally:
        model.cpu()
        del lp_dev, optimizer
        cleanup()


def evaluate_lp_on(model, lp: LPData, split: str) -> dict:
    """evaluate_lp on the GPU if there is one; the model is moved back to the CPU afterwards."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lp_dev = lp.to(device)
    try:
        return evaluate_lp(model.to(device), lp_dev, split)
    finally:
        model.cpu()
        del lp_dev
        cleanup()


def query_ranks_on(model, lp: LPData, split: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """query_ranks on the GPU if there is one, returned on the CPU; the model is moved back to the CPU."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lp_dev = lp.to(device)
    try:
        return {name: (head.cpu(), rank.cpu()) for name, (head, rank) in
                query_ranks(model.to(device), lp_dev, split).items()}
    finally:
        model.cpu()
        del lp_dev
        cleanup()


def run_lp_config_over_seeds(model_factory, lp, hidden_list, bases, lr, wd, epochs, patience,
                             start_from_epoch, disjoint_ratio, seeds, save_path):
    """
    train_lp_model once per seed, as gnn_tools.run_config_over_seeds: the config is scored by
    its mean val MRR over the seeds (each seed's peak from start_from_epoch on), its best epoch is
    the median per-seed best epoch (median_low). An OOM fails the config and ends its seeds.
    """
    mrrs, best_epochs, grad_norms, status, start = [], [], [], 'ok', time.time()
    for seed in seeds:
        set_seed(seed)
        model = model_factory(hidden_list, bases)
        try:
            model, history = train_lp_model(model, lp, epochs, lr, wd, patience, disjoint_ratio, seed=seed,
                                            start_from_epoch=start_from_epoch,
                                            save_path=f"{save_path.removesuffix('.pt')}_seed{seed}.pt")
            mrrs.append(max(history['val_mrr'][start_from_epoch:]))
            best_epochs.append(history['best_epoch'])
            grad_norms.append(max(history['grad_norm']))
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            print("  -> CUDA OOM: skipping this configuration.")
            status = 'oom'
            for attr in ('last_traceback', 'last_type', 'last_value'):
                if hasattr(sys, attr):
                    delattr(sys, attr)
            del e
            break
        finally:
            model.cpu()
            del model
            cleanup()
    ok = status == 'ok'
    return {
        'val_mrr': float(np.mean(mrrs)) if ok else None,
        'val_mrr_std': float(np.std(mrrs, ddof=1)) if ok and len(mrrs) > 1 else None,
        'val_mrr_seeds': ';'.join(_fmt(m) for m in mrrs),
        'best_epoch': statistics.median_low(best_epochs) if ok else None,
        'best_epoch_seeds': ';'.join(str(e) for e in best_epochs),
        'max_grad_norm': max(grad_norms) if grad_norms else None,
        'runtime_sec': time.time() - start, 'status': status,
    }


def _test_columns(test: dict) -> dict:
    """evaluate_lp's test metrics as CSV columns: test_<metric> pooled, test_<target>_<metric> per target."""
    names = ('mrr', *(f'hits@{k}' for k in KS))
    return {**{f'test_{m}': test['pooled'][m] for m in names},
            **{f'test_{t}_{m}': test[t][m] for t in test if t != 'pooled' for m in names}}


def evaluate_final_lp_row(model_factory, lp, kg, save_dir, hidden_dim, depth, bases, lr, weight_decay,
                          best_epoch, val_mrr, disjoint_ratio, seed):
    """Retrains on train + val for best_epoch + 1 epochs and scores the test races: one phase-4 row."""
    set_seed(seed)
    model = model_factory([hidden_dim] * (depth - 1), bases)
    model = train_final_lp_model(model, lp, best_epoch + 1, lr, weight_decay, disjoint_ratio, seed=seed,
                                 save_path=f"{save_dir}{kg}_final_seed{seed}.pt")
    test = evaluate_lp_on(model, lp, 'test')
    num_parameters = sum(p.numel() for p in model.parameters())
    del model
    cleanup()
    print(f"{kg} seed {seed}: test MRR {test['pooled']['mrr']:.4f} ("
          + ", ".join(f"{t} {test[t]['mrr']:.4f}" for t in lp.targets) + ")")
    return {
        'phase': '4_Final_Test_Evaluation', 'parameter_tested': 'unified_model', 'seed': seed,
        'val_mrr': val_mrr, **_test_columns(test),
        'num_parameters': num_parameters, 'runtime_sec': 0.0, 'status': 'ok',
        'hidden_dim': hidden_dim, 'layers': depth, 'num_bases': bases, 'lr': lr, 'weight_decay': weight_decay,
        'best_epoch': best_epoch,
    }


def tuning_checkpoint(save_dir, kg, dim, layers, bases, lr, wd, seed=None):
    """A tuning config's checkpoint path, named by the config (so a reused row points at it), per seed.
    lr and wd are written as floats: a grid's int 0 comes back from the results CSV as 0.0."""
    path = f"{save_dir}tuning/{kg}_L{layers}_dim{dim}_B{bases}_lr{float(lr)}_wd{float(wd)}"
    return f"{path}.pt" if seed is None else f"{path}_seed{seed}.pt"


def target_convergence(save_dir, kg, row, seeds, start_from_epoch=0):
    """
    Per seed and target of a tuning row (e.g. the winner): the target's val MRR at the epoch
    selected on pooled val MRR, its own peak from start_from_epoch on, and the epoch of that peak.
    A target whose peak comes after the selected epoch was still improving when pooled MRR chose.
    """
    optional = lambda v: None if pd.isna(v) else int(v)
    rows = []
    for seed in seeds:
        path = tuning_checkpoint(save_dir, kg, optional(row['hidden_dim']), int(row['layers']),
                                 optional(row['num_bases']), row['lr'], row['weight_decay'], seed)
        history = torch.load(path, map_location='cpu', weights_only=False)['history']
        best = history['best_epoch']
        for target in (k.removeprefix('val_mrr_') for k in history if k.startswith('val_mrr_')):
            curve = history[f'val_mrr_{target}']
            peak = start_from_epoch + int(np.argmax(curve[start_from_epoch:]))
            rows.append({'kg': kg, 'seed': seed, 'target': target, 'selected_epoch': best,
                         'val_mrr_at_selected': curve[best], 'peak_epoch': peak, 'peak_val_mrr': curve[peak]})
    return pd.DataFrame(rows)


def run_lp_search(model_factory, lp, kg, save_dir, csv_path,
                  grid_dim, grid_depth, grid_bases, grid_lr, grid_wd, default_bases, default_lr, default_wd,
                  epochs, patience, start_from_epoch, disjoint_ratio, tuning_seeds, final_seeds):
    """
    The node-prediction search (gnn_tools.run_sequential_optimization_with_verification) on
    pooled val MRR: a joint dim x depth grid (one 1-layer run, hidden_dim None), greedy bases ->
    lr -> wd at the winning architecture, a 2x2 verification grid over the top-2 dims at the
    winning depth x the top-2 (bases, lr, wd) combos of phase 2, then a train + val retrain and
    test once per final seed. Same phase names and columns (val_mrr for val_roc_auc), so
    gnn_tools.finished_tuning_results(metric='val_mrr') and runs_ended_by_epoch_cap read the CSV.
    model_factory(hidden_list, bases) -> LinkPredictor; GraphSAGE passes grid_bases=[None].
    """
    results = []

    def log(phase, parameter, row, dim, layers, bases, lr, wd):
        row = {'phase': phase, 'parameter_tested': parameter, **row, 'hidden_dim': dim, 'layers': layers,
               'num_bases': bases, 'lr': lr, 'weight_decay': wd}
        results.append(row)
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"  -> [{phase}] {parameter}: mean val MRR {_fmt(row['val_mrr'])} (seeds {row['val_mrr_seeds']}) "
              f"| best epoch {row['best_epoch']} | {row['status']}")
        return row

    def run(phase, parameter, dim, layers, bases, lr, wd):
        row = run_lp_config_over_seeds(
            model_factory, lp, [dim] * (layers - 1), bases, lr, wd, epochs, patience, start_from_epoch,
            disjoint_ratio, tuning_seeds, save_path=tuning_checkpoint(save_dir, kg, dim, layers, bases, lr, wd))
        return log(phase, parameter, row, dim, layers, bases, lr, wd)

    def reuse(phase, parameter, row):
        """A row trained in an earlier phase, logged again without its runtime."""
        keep = {k: row[k] for k in ('val_mrr', 'val_mrr_std', 'val_mrr_seeds', 'best_epoch', 'best_epoch_seeds',
                                    'max_grad_norm', 'status')}
        return log(phase, parameter, dict(keep, runtime_sec=0.0), row['hidden_dim'], row['layers'],
                   row['num_bases'], row['lr'], row['weight_decay'])

    scored = lambda row: row['status'] == 'ok' and row['val_mrr'] is not None
    print(f"\n=== Link-prediction search: {kg} (seeds {list(tuning_seeds)}) ===")

    # Phase 1: joint hidden dim x depth.
    for layers in grid_depth:
        for dim in (grid_dim if layers > 1 else [None]):
            run('1_Architecture', f'layers_{layers}_dim_{dim}', dim, layers, default_bases, default_lr, default_wd)
    phase_1 = [r for r in results if scored(r)]
    if not phase_1:
        raise RuntimeError(f"every phase-1 configuration failed for {kg}; see {csv_path}")
    best_arch = max(phase_1, key=lambda r: r['val_mrr'])
    dim, depth = best_arch['hidden_dim'], best_arch['layers']
    top_dims = [r['hidden_dim'] for r in sorted((r for r in phase_1 if r['layers'] == depth),
                                                key=lambda r: r['val_mrr'], reverse=True)[:2]]
    print(f"\nPhase 1 winner: depth {depth}, dim {dim}; top dims at that depth: {top_dims}")

    # Phase 2: greedy bases -> lr -> wd at the winning architecture.
    phase_2 = {}

    def sweep(phase, name, grid, default, current, config):
        best = None
        for value in grid:
            bases, lr, wd = config(value)
            row = (reuse(phase, f'{name}_{value}', current) if value == default
                   else run(phase, f'{name}_{value}', dim, depth, bases, lr, wd))
            if scored(row):
                phase_2[(bases, lr, wd)] = row['val_mrr']
                best = row if best is None or row['val_mrr'] > best['val_mrr'] else best
        return best

    best_bases = sweep('2a_Bases', 'bases', grid_bases, default_bases, best_arch,
                       lambda b: (b, default_lr, default_wd))
    best_lr = sweep('2b_LR', 'lr', grid_lr, default_lr, best_bases,
                    lambda lr: (best_bases['num_bases'], lr, default_wd))
    sweep('2c_WD', 'wd', grid_wd, default_wd, best_lr,
          lambda wd: (best_bases['num_bases'], best_lr['lr'], wd))
    top_reg = [params for params, _ in sorted(phase_2.items(), key=lambda x: x[1], reverse=True)[:2]]
    print(f"\nPhase 2 top (bases, lr, wd): {top_reg}")

    # Phase 3: 2x2 verification grid; the winner comes from these rows only.
    for verify_dim in top_dims:
        for bases, lr, wd in top_reg:
            parameter = f'dim_{verify_dim}_reg_B{bases}_lr{lr}_wd{wd}'
            cached = next((r for r in results if scored(r) and (r['hidden_dim'], r['layers'], r['num_bases'], r['lr'],
                                                                r['weight_decay']) == (verify_dim, depth, bases, lr, wd)), None)
            if cached is not None:
                reuse('3_Verification_Cached', parameter, cached)
            else:
                run('3_Verification', parameter, verify_dim, depth, bases, lr, wd)
    phase_3 = [r for r in results if scored(r) and r['phase'].startswith('3_')]
    winner = max(phase_3 or [r for r in results if scored(r)], key=lambda r: r['val_mrr'])
    print(f"\n{kg} winner: dim {winner['hidden_dim']}, depth {depth}, bases {winner['num_bases']}, lr {winner['lr']}, "
          f"wd {winner['weight_decay']} | val MRR {winner['val_mrr']:.4f} | final epochs {winner['best_epoch'] + 1}")

    # Phase 4: retrain on train + val and test, once per final seed.
    for seed in final_seeds:
        results.append(evaluate_final_lp_row(
            model_factory, lp, kg, save_dir, winner['hidden_dim'], depth, winner['num_bases'], winner['lr'],
            winner['weight_decay'], winner['best_epoch'], winner['val_mrr'], disjoint_ratio, seed))
        pd.DataFrame(results).to_csv(csv_path, index=False)
    return pd.DataFrame(results)


# ── Enrichment ablation (f1_node_data.ABLATION_VARIANTS) ──────────────────

# Summarised per variant by gnn_tools.summarize_ablation(metrics=..., delta_metrics=...).
LP_ABLATION_METRICS = ['val_mrr', 'test_mrr', *(f'test_{t}_mrr' for t in TARGETS)]


def run_lp_ablation_variant(model_factory, lp, variant, hidden_list, bases, lr, wd, epochs, patience,
                            start_from_epoch, disjoint_ratio, seeds, save_dir, csv_path):
    """
    gnn_tools.run_ablation_variant for link prediction: trains one enrichment-ablation graph at a
    fixed configuration, once per seed (run_lp_config_over_seeds: train races, early stopping on
    pooled val MRR after a warm-up of start_from_epoch epochs), then scores the best-val checkpoint
    on the test races (evaluate_lp: train + val facts as context). No train + val retrain. Every
    variant gets the same configuration, so only the graph differs.

    Rows go to csv_path (shared by all variants) as each seed finishes. On a re-run, a
    (variant, seed) with an 'ok' row is skipped and failed rows are dropped and retried.
    Rows of another configuration raise; other constants are not checked: move the CSV away
    after changing one.
    """
    rows = pd.read_csv(csv_path).to_dict('records') if os.path.exists(csv_path) else []
    rows = [r for r in rows if r['status'] == 'ok']
    config = {'hidden_dim': hidden_list[0] if hidden_list else None, 'layers': len(hidden_list) + 1,
              'num_bases': bases, 'lr': lr, 'weight_decay': wd}
    if any(_config_of(r) != config for r in rows):
        raise ValueError(f"{csv_path} holds rows of another configuration than {config}; move it away")
    done = {(r['variant'], r['seed']) for r in rows}

    for seed in seeds:
        if (variant, seed) in done:
            print(f"{variant} seed {seed}: already in {csv_path}")
            continue
        print(f"\n=== LP ablation {variant} | seed {seed} ===")
        run = run_lp_config_over_seeds(model_factory, lp, hidden_list, bases, lr, wd, epochs, patience,
                                       start_from_epoch, disjoint_ratio, [seed], save_path=f"{save_dir}{variant}.pt")
        test = {}
        if run['status'] == 'ok':
            model = model_factory(hidden_list, bases)
            checkpoint = torch.load(f"{save_dir}{variant}_seed{seed}.pt", map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            test = _test_columns(evaluate_lp_on(model, lp, 'test'))
            del model, checkpoint
            cleanup()
        rows.append({
            'variant': variant, 'seed': seed,
            **{k: run[k] for k in ('val_mrr', 'best_epoch', 'max_grad_norm', 'runtime_sec', 'status')},
            **test, **config,
            'num_nodes': lp.data.num_nodes, 'num_edges': lp.data.edge_index.size(1),
            'num_relations': lp.num_relations, 'num_feature_keys': len(lp.vocabulary),
            'member_pairs': lp.member_edges.size(1) // 2,
        })
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    return pd.DataFrame(rows)


# ── Mechanism checks (link_prediction_analysis.ipynb) ─────────────────────

# The literal a prediction can identify its answer by, as the chain of properties that leads to it
# from the query head or from a candidate. Driver/team/race/circuit names stand for every literal of
# that entity: all of an entity's literals sit at the same distance in each KG.
EVIDENCE_CHAINS = {
    "LP team: the driver's literals, from the entry": ("hasDriver", "hasDriverName"),
    "LP team: the team's literals, from a candidate": ("hasTeamName",),
    "LP position: a session's rank, from the entry": ("hasSessionResult", "hasPosition", "hasRank"),
    "LP position: the rank, from a candidate": ("hasRank",),
    "LP circuit: the race's literals, from the race": ("hasRaceName",),
    "LP circuit: the circuit's literals, from a candidate": ("hasCircuitName",),
    "NP: the entry's own grid rank": ("hasStartPosition", "hasRank"),
}


def evidence_hops(schema, chain) -> int:
    """
    Message-passing hops along a chain of properties in a KG: 1 per directly stated property, 2 per
    property the schema spells as a SULO path (subject -> <subject>/<property> -> value, see
    f1_population.assertion_lines; the intermediate node's rdf:type adds no hop to the path).
    """
    return sum(1 if schema.properties[p].hop is None else 2 for p in chain)


def session_position_entries(assertions: pd.DataFrame, session_type: str | None = None) -> set[str]:
    """RaceEntry IRIs with at least one SessionResult (of `session_type`, if given) that has a position."""
    of = lambda prop: assertions[assertions["property"] == prop]
    sessions = set(of("hasPosition")["subject"])
    if session_type is not None:
        typed = of("hasSessionType")
        sessions &= set(typed.loc[typed["value"] == session_type, "subject"])
    links = of("hasSessionResult")
    return set(links.loc[links["value"].isin(sessions), "subject"])


# ── Simple baselines (reported next to the GNNs' test Hits@1) ─────────────

def _majority(known: pd.DataFrame, keys: list[str], value: str) -> pd.Series:
    """keys -> most frequent value among `known` rows; ties go to the smallest value (IRI order)."""
    counts = known.groupby(keys + [value]).size().rename('n').reset_index()
    counts = counts.sort_values(['n', value], ascending=[False, True], kind='stable')
    return counts.drop_duplicates(keys).set_index(keys)[value]


def _cascade(queries: pd.DataFrame, known: pd.DataFrame, levels: list[list[str]], value: str) -> pd.Series:
    """Each query's majority value at the first key level it has a known value for, else overall."""
    prediction = pd.Series(known[value].value_counts().sort_index().idxmax(), index=queries.index)
    for keys in reversed(levels):
        found = queries.join(_majority(known, keys, value).rename('_p'), on=keys)['_p']
        prediction = found.fillna(prediction)
    return prediction


def baseline_hits_at_1(assertions: pd.DataFrame, split: tuple[list[str], list[str], list[str]]) -> dict[str, float]:
    """
    Test Hits@1 of one-rule predictors that see the train + val races only:
      team      the driver's most frequent team in the same season, else over all seasons
      circuit   the race name's most frequent circuit
      position  the qualifying position, else the driver's most frequent grid slot in the same season,
                else over all seasons
    each falling back to the overall most frequent answer.
    """
    value = lambda prop: assertions[assertions["property"] == prop].set_index("subject")["value"]
    entry_race = assertions[assertions["property"] == "hasRaceEntry"].set_index("value")["subject"]
    session = assertions[assertions["property"] == "hasSessionResult"][["subject", "value"]]
    quali = session[session["value"].map(value("hasSessionType")) == "Qualifying"]
    quali = quali.assign(q=quali["value"].map(value("hasPosition"))).dropna().set_index("subject")["q"]
    known_races, test_races = set(split[0]) | set(split[1]), set(split[2])

    entries = pd.DataFrame({"race": entry_race})
    entries["season"] = entries["race"].map(value("partOfSeason"))
    entries["driver"] = value("hasDriver")
    entries["team"] = value("hasTeam")
    entries["position"] = value("hasStartPosition")
    entries["quali"] = quali
    races = pd.DataFrame({"name": value("hasRaceName"), "circuit": value("heldAtCircuit")})

    def predict(frame, target, levels):
        """(test rows with a known answer, their one-rule predictions)."""
        frame = frame[frame[target].notna()]
        test = frame[frame["race"].isin(test_races)]
        return test, _cascade(test, frame[frame["race"].isin(known_races)], levels, target)

    team, team_rule = predict(entries, "team", [["driver", "season"], ["driver"]])
    grid, grid_rule = predict(entries, "position", [["driver", "season"], ["driver"]])
    circuit, circuit_rule = predict(races.assign(race=races.index), "circuit", [["name"]])
    return {
        "team": float((team_rule == team["team"]).mean()),
        "position": float((grid["quali"].fillna(grid_rule) == grid["position"]).mean()),
        "circuit": float((circuit_rule == circuit["circuit"]).mean()),
    }
