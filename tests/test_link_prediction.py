import contextlib
from dataclasses import replace
import io

import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from f1_node_data import ABLATION_VARIANTS, ablation_schema, degree_feature
from f1_population import NS, render_ntriples
from link_prediction import link_prediction_tools as lpt
from link_prediction.link_prediction_tools import (
    LP_ABLATION_METRICS, MASKED_PROPERTIES, TARGETS, LinkPredictor, LPData, TargetData, _cascade, _majority, baseline_hits_at_1,
    EVIDENCE_CHAINS, SPLITS, build_lp_data, evaluate_lp, evidence_hops, hidden_edges, query_ranks, race_split,
    rank_of_answer, ranking_metrics, session_position_entries, target_facts, train_lp_model, visible_graph,
)
from node_prediction import gnn_tools
from node_prediction.models import GraphSAGEModel, RGCNModel

KGS = ("adhoc", "pure_sulo", "hybrid_sulo")
R = lambda i: NS + f"Race/{i}"
SPLIT = ([R(1), R(3), R(12), R(13)], [R(10)], [R(11)])


@pytest.fixture
def cpu(monkeypatch):
    """Train on CPU: GPU scatter ops are not bit-reproducible, CPU ones are."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture(scope="module")
def lp(built, schemas):
    entities, assertions = built
    return {kg: build_lp_data(kg, entities, assertions, SPLIT, schemas[kg]) for kg in KGS}


def _hide(lp, races):
    return torch.tensor([r in races for r in lp.races])


def _adjacent(graph, a, b):
    src, dst = graph.edge_index
    return bool(((src == a) & (dst == b)).any())


# ── Split and facts ───────────────────────────────────────────────────────

def test_race_split_is_disjoint_complete_reproducible_and_80_10_10():
    entities = pd.DataFrame({"iri": [f"race{i:03d}" for i in range(200)] + ["d"], "class": ["Race"] * 200 + ["Driver"]})
    train, val, test = race_split(entities)
    assert sorted(train + val + test) == sorted(entities["iri"][:200])
    assert (len(train), len(val), len(test)) == (160, 20, 20)
    assert race_split(entities) == (train, val, test)


def test_every_kg_holds_the_same_target_facts(built, schemas):
    _, assertions = built
    pairs = {kg: {t: set(zip(f["subject"], f["value"])) for t, f in target_facts(assertions, schemas[kg]).items()}
             for kg in KGS}
    assert pairs["adhoc"] == pairs["pure_sulo"] == pairs["hybrid_sulo"]
    assert len(pairs["adhoc"]["team"]) == 5 and len(pairs["adhoc"]["circuit"]) == 6
    assert len(pairs["adhoc"]["position"]) == 4   # entry 102 started from the pit lane (grid 0)


@pytest.mark.parametrize("kg", KGS)
def test_candidates_are_every_node_of_the_answer_class(lp, kg):
    d = lp[kg]
    for name, t in d.targets.items():
        iris = [iri for iri, i in d.node_ids.items() if i in set(t.candidates.tolist())]
        assert all(iri.startswith(f"<{NS}{TARGETS[name][2]}/") for iri in iris)
    assert [len(t.candidates) for t in d.targets.values()] == [1, 3, 2]


@pytest.mark.parametrize("kg", KGS)
def test_finish_position_is_not_in_the_graph(built, schemas, lp, kg):
    """Entry 101 started and finished P1: in Pure that was one merged triple. Now every
    RaceEntry -> Position edge is a start position."""
    entities, assertions = built
    lines, _ = render_ntriples(entities, assertions[~assertions["property"].isin(MASKED_PROPERTIES)], schemas[kg])
    assert not any("hasFinishPosition" in line for line in lines)
    d = lp[kg]
    entries = {i for iri, i in d.node_ids.items() if iri.startswith(f"<{NS}RaceEntry/") and iri.count("/") == 5}
    positions = set(d.targets["position"].candidates.tolist())
    src, dst = d.data.edge_index
    to_position = [(s, t) for s, t in zip(src.tolist(), dst.tolist()) if s in entries and t in positions]
    grid = d.targets["position"]
    assert sorted(to_position) == sorted(zip(grid.head.tolist(), grid.candidates[grid.answer].tolist()))


@pytest.mark.parametrize("kg, team_lines", [("adhoc", 1), ("pure_sulo", 3), ("hybrid_sulo", 1)])
def test_a_fact_owns_every_edge_of_its_path(lp, kg, team_lines):
    """Pure's team is hasParticipant -> Role -> isFeatureOf plus the Role's rdf:type: 3 lines."""
    t = lp[kg].targets["team"]
    assert t.edge_fact.size(1) == 2 * team_lines * len(t.head)
    assert torch.bincount(t.edge_fact[1]).eq(2 * team_lines).all()


# ── Hiding ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kg", KGS)
def test_hiding_nothing_keeps_the_whole_graph(lp, kg):
    d = lp[kg]
    assert not hidden_edges(d, _hide(d, set())).any()
    assert visible_graph(d, _hide(d, set())).edge_index.equal(d.data.edge_index)


@pytest.mark.parametrize("kg", KGS)
def test_hidden_races_lose_every_fact_and_others_keep_theirs(lp, kg):
    d = lp[kg]
    hide = _hide(d, {R(10), R(11)})
    graph = visible_graph(d, hide)
    iri_of = {i: iri for iri, i in d.node_ids.items()}
    for name, t in d.targets.items():
        for head, answer, race in zip(t.head.tolist(), t.answer.tolist(), t.race.tolist()):
            tail = int(t.candidates[answer])
            if kg == "pure_sulo" and name == "team":   # the answer sits behind a Role node
                role = d.node_ids[f"{iri_of[head][:-1]}/hasTeam>"]
                assert _adjacent(graph, head, role) == _adjacent(graph, role, tail) == (not hide[race])
                if hide[race]:
                    assert not (graph.edge_index == role).any()
            else:
                assert _adjacent(graph, head, tail) == _adjacent(graph, tail, head) == (not hide[race])


def test_a_member_pair_stays_while_one_of_its_entries_shows_its_team(lp):
    """Driver 1 drove for Team 1 in races 1, 10, 11; Driver 2 only in 10 and 11."""
    d = lp["hybrid_sulo"]
    team, d1, d2 = (d.node_ids[f"<{NS}{x}>"] for x in ("Team/1", "Driver/1", "Driver/2"))
    graph = visible_graph(d, _hide(d, {R(10), R(11)}))
    assert _adjacent(graph, team, d1) and not _adjacent(graph, team, d2)
    graph = visible_graph(d, _hide(d, {R(1), R(10), R(11)}))
    assert not _adjacent(graph, team, d1)


def test_adhoc_has_no_member_pairs(lp):
    assert lp["adhoc"].member_edges.size(1) == 0
    assert lp["pure_sulo"].member_edges.size(1) == lp["hybrid_sulo"].member_edges.size(1) == 4


@pytest.mark.parametrize("kg", KGS)
def test_degree_is_computed_on_the_visible_graph(lp, kg):
    d = lp[kg]
    graph = visible_graph(d, _hide(d, {R(11)}))
    assert graph.degree_feat.equal(degree_feature(graph.edge_index, d.data.num_nodes))
    assert not graph.degree_feat.equal(degree_feature(d.data.edge_index, d.data.num_nodes))


# ── Ranking ───────────────────────────────────────────────────────────────

def test_rank_limits():
    scores = torch.tensor([[3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0], [5.0, 5.0, 0.0, 0.0]])
    ranks = rank_of_answer(scores, torch.zeros(4, dtype=torch.long))
    assert ranks.tolist() == [1.0, 4.0, 2.5, 1.5]   # best, worst, all tied (no credit), tie of two


def test_ranking_metrics():
    m = ranking_metrics(torch.tensor([1.0, 2.0, 4.0, 20.0]))
    assert m["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 4 + 1 / 20) / 4)
    assert (m["hits@1"], m["hits@3"], m["hits@10"], m["num_queries"]) == (0.25, 0.5, 0.75, 4)
    assert ranking_metrics(torch.ones(5))["mrr"] == 1.0


# ── Training on a planted link ────────────────────────────────────────────

def _planted(shuffle, n=600, seed=0):
    """
    n heads (key 0), each linked to cue node 0 or 1 (keys 1, 2) by relation 0 and to its answer
    (candidate node a0 or a1, keys 3, 4) by relation 1. Planted: answer = cue; shuffled: answers
    permuted across heads, so the cue says nothing. One race per head, split 80/10/10.
    """
    gen = torch.Generator().manual_seed(seed)
    cue = torch.randint(0, 2, (n,), generator=gen)
    answer = cue[torch.randperm(n, generator=gen)] if shuffle else cue
    heads, cues, answers = torch.arange(n), n + cue, n + 2 + answer
    src = torch.cat([heads, heads])
    dst = torch.cat([cues, answers])
    edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
    edge_type = torch.cat([torch.zeros(n), torch.ones(n), torch.full((n,), 2), torch.full((n,), 3)]).long()
    key_id = torch.cat([torch.zeros(n), torch.tensor([1, 2, 3, 4])]).long()
    data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=n + 4, key_id=key_id, value=torch.zeros(n + 4))
    answer_cols = torch.cat([torch.arange(n, 2 * n), torch.arange(3 * n, 4 * n)])
    target = TargetData(head=heads, answer=answer, race=heads, candidates=torch.tensor([n + 2, n + 3]),
                        edge_fact=torch.stack([answer_cols, torch.cat([heads, heads])]))
    split = torch.zeros(n, dtype=torch.long)
    split[torch.randperm(n, generator=gen)[: n // 5]] = torch.tensor([1, 2]).repeat(n // 10)
    empty = torch.zeros(2, 0, dtype=torch.long)
    return LPData(data, {"team": target}, split, empty, empty, 4, ["h", "c0", "c1", "a0", "a1"], {}, list(range(n)))


def _fit(model_cls, lp, seed=0, epochs=30):
    gnn_tools.set_seed(seed)
    encoder = model_cls(num_keys=5, feature_dim=8, hidden_channels_list=[8], num_relations=4, out_channels=8)
    with contextlib.redirect_stdout(io.StringIO()):
        model, history = train_lp_model(LinkPredictor(encoder, 1, 8), lp, epochs=epochs, lr=0.05, weight_decay=0,
                                        patience=epochs, disjoint_ratio=0.3, seed=seed)
    return model, history, evaluate_lp(model, lp, "test")


@pytest.mark.parametrize("model_cls", [RGCNModel, GraphSAGEModel])
def test_learns_a_planted_link(cpu, model_cls):
    _, _, test = _fit(model_cls, _planted(shuffle=False))
    assert test["team"]["mrr"] > 0.95


@pytest.mark.parametrize("model_cls", [RGCNModel, GraphSAGEModel])
def test_shuffled_answers_stay_at_chance(cpu, model_cls):
    """With shuffled answers the best a model can do is rank by answer frequency (two candidates: MRR ~0.75)."""
    lp = _planted(shuffle=True)
    _, _, test = _fit(model_cls, lp)
    assert test["team"]["mrr"] <= lpt.frequency_baseline(lp, "test")["team"]["mrr"] + 0.03


def test_warm_up_must_leave_an_epoch_to_select():
    with pytest.raises(ValueError):
        train_lp_model(None, _planted(shuffle=False), epochs=5, lr=0.01, weight_decay=0, patience=5,
                       disjoint_ratio=0.3, start_from_epoch=5)


@pytest.mark.parametrize("start_from_epoch", [0, 10])
def test_early_stopping_restores_the_best_epoch_after_the_warm_up(cpu, start_from_epoch):
    lp = _planted(shuffle=True)        # val MRR wanders, so patience triggers before the cap
    gnn_tools.set_seed(0)
    encoder = RGCNModel(num_keys=5, feature_dim=8, hidden_channels_list=[8], num_relations=4, out_channels=8)
    with contextlib.redirect_stdout(io.StringIO()):
        model, history = train_lp_model(LinkPredictor(encoder, 1, 8), lp, epochs=200, lr=0.05, weight_decay=0,
                                        patience=5, disjoint_ratio=0.3, start_from_epoch=start_from_epoch)
        restored = evaluate_lp(model, lp, "val")["pooled"]["mrr"]
    best = history["best_epoch"]
    selectable = history["val_mrr"][start_from_epoch:]
    assert best >= start_from_epoch and history["val_mrr"][best] == max(selectable)
    assert len(history["val_mrr"]) == best + 5 + 1 < 200     # stopped `patience` epochs after the best
    assert restored == pytest.approx(history["val_mrr"][best])


def test_an_oom_fails_the_config_and_skips_its_remaining_seeds(monkeypatch):
    calls = []

    def oom(*args, seed, **kwargs):
        calls.append(seed)
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1 GiB")

    monkeypatch.setattr(lpt, "train_lp_model", oom)
    factory = lambda hidden_list, bases: LinkPredictor(torch.nn.Identity(), 1, 8)
    row = lpt.run_lp_config_over_seeds(factory, None, [8], None, 0.01, 0, 10, 5, 0, 0.3, [0, 1, 2], "x.pt")
    assert calls == [0] and row["status"] == "oom" and row["val_mrr"] is None and row["best_epoch"] is None


def test_other_runtime_errors_are_raised(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("shape mismatch")

    monkeypatch.setattr(lpt, "train_lp_model", broken)
    factory = lambda hidden_list, bases: LinkPredictor(torch.nn.Identity(), 1, 8)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        lpt.run_lp_config_over_seeds(factory, None, [8], None, 0.01, 0, 10, 5, 0, 0.3, [0], "x.pt")


def test_training_is_reproducible_per_seed(cpu):
    lp = _planted(shuffle=False)
    _, first, _ = _fit(RGCNModel, lp, epochs=5)
    _, second, _ = _fit(RGCNModel, lp, epochs=5)
    assert first == second


def test_each_step_hides_and_supervises_the_same_share_of_training_races(cpu, monkeypatch):
    lp = _planted(shuffle=False)
    seen = []
    real_visible, real_loss = lpt.visible_graph, lpt._query_loss
    monkeypatch.setattr(lpt, "visible_graph", lambda d, hidden: seen.append(["graph", hidden.clone()]) or real_visible(d, hidden))
    monkeypatch.setattr(lpt, "_query_loss", lambda m, e, d, q: seen.append(["loss", q.clone()]) or real_loss(m, e, d, q))
    _fit(RGCNModel, lp, epochs=3)
    train = lp.race_split == 0
    steps = [(seen[i][1], seen[i + 1][1]) for i in range(len(seen) - 1) if seen[i][0] == "graph" and seen[i + 1][0] == "loss"]
    assert len(steps) == 3
    for hidden, queried in steps:
        assert int(queried.sum()) == round(0.3 * int(train.sum()))
        assert not (queried & ~train).any()          # only training races are supervised
        assert hidden.equal(queried | ~train)        # ... and exactly they and val/test are hidden
    evals = [h for kind, h in seen if kind == "graph"][1::2]
    assert all(h.equal(lp.race_split >= 1) for h in evals)   # val scored with val + test hidden


def test_only_supervised_queries_reach_the_gradients(cpu):
    lp = _planted(shuffle=False)
    gnn_tools.set_seed(0)
    model = LinkPredictor(RGCNModel(num_keys=5, feature_dim=8, hidden_channels_list=[8], num_relations=4,
                                    out_channels=8), 1, 8)
    queried = torch.zeros(len(lp.races), dtype=torch.bool)
    queried[:3] = True
    emb = model(visible_graph(lp, queried | (lp.race_split > 0)))
    loss = lpt._query_loss(model, emb, lp, queried)
    expected = torch.nn.functional.cross_entropy(model.scores(emb, 0, torch.arange(3), lp.targets["team"].candidates),
                                                 lp.targets["team"].answer[:3])
    assert loss.item() == pytest.approx(expected.item())


# ── Baselines ─────────────────────────────────────────────────────────────

def test_majority_breaks_ties_by_the_smallest_value():
    known = pd.DataFrame({"k": ["a", "a", "a", "a", "b"], "v": ["y", "x", "y", "x", "z"]})
    assert _majority(known, ["k"], "v").to_dict() == {"a": "x", "b": "z"}


def test_cascade_uses_the_first_level_with_a_known_value():
    known = pd.DataFrame({"d": ["1", "1", "2"], "s": ["s1", "s2", "s1"], "v": ["A", "B", "C"]})
    queries = pd.DataFrame({"d": ["1", "2", "3"], "s": ["s2", "s9", "s1"]})
    assert _cascade(queries, known, [["d", "s"], ["d"]], "v").tolist() == ["B", "C", "A"]


def test_baselines_on_the_toy_population(built):
    _, assertions = built
    hits = baseline_hits_at_1(assertions, SPLIT)
    # Race 11 is the test race. Team: the only team, right. Circuit: its name is unseen, so the overall
    # majority Circuit/1, wrong. Grid: no qualifying in race 11; entry 103 gets driver 1's 2010 slot P1
    # (was P2), entry 104 driver 2 has no known slot, so the overall P1 (was P3).
    assert hits == {"team": 1.0, "position": 0.0, "circuit": 0.0}


# ── Search ────────────────────────────────────────────────────────────────

def test_search_runs_every_phase_and_its_csv_resumes(cpu, tmp_path):
    lp = _planted(shuffle=False, n=200)

    def factory(hidden_list, bases):
        dim = hidden_list[-1] if hidden_list else 8
        return LinkPredictor(RGCNModel(num_keys=5, feature_dim=8, hidden_channels_list=hidden_list, num_relations=4,
                                       num_bases=bases, out_channels=dim), 1, dim)

    csv = tmp_path / "search.csv"
    with contextlib.redirect_stdout(io.StringIO()):
        results = lpt.run_lp_search(
            factory, lp, "toy", f"{tmp_path}/", csv, grid_dim=[4, 8], grid_depth=[1, 2], grid_bases=[None, 2],
            grid_lr=[0.05, 0.01], grid_wd=[0, 1e-4], default_bases=None, default_lr=0.05, default_wd=0,
            epochs=6, patience=6, start_from_epoch=1, disjoint_ratio=0.3, tuning_seeds=[0, 1], final_seeds=[0, 1])
    phases = results["phase"].tolist()
    assert phases[:3] == ["1_Architecture"] * 3                      # depth 1 once, depth 2 per dim
    assert {"2a_Bases", "2b_LR", "2c_WD"} <= set(phases)
    assert any(p.startswith("3_") for p in phases) and phases[-2:] == ["4_Final_Test_Evaluation"] * 2
    # the phase-1 winner's dim with phase 2's best regularisation was trained in phase 2: reused, not re-run
    cached = results[results["phase"] == "3_Verification_Cached"]
    assert len(cached) >= 1 and (cached["runtime_sec"] == 0).all()
    trained = results[results["runtime_sec"] > 0]
    assert trained["val_mrr_seeds"].map(lambda s: len(s.split(";"))).eq(2).all()
    assert trained["best_epoch_seeds"].map(lambda s: min(int(e) for e in s.split(";"))).ge(1).all()   # warm-up
    assert gnn_tools.finished_tuning_results(csv, [4, 8], [1, 2], [0, 1], [0, 1], metric="val_mrr") is not None
    assert gnn_tools.finished_tuning_results(csv, [4, 8, 16], [1, 2], [0, 1], [0, 1], metric="val_mrr") is None
    # epochs 6 = patience 6: every seed's best epoch is >= EPOCHS - PATIENCE, so every trained run is capped
    assert len(gnn_tools.runs_ended_by_epoch_cap(pd.read_csv(csv), 6, 6)) == len(trained)
    final = results[results["phase"] == "4_Final_Test_Evaluation"]
    assert final[["test_mrr", "test_team_mrr"]].notna().all().all()
    # every tuning row (wd 0 read back from the CSV as 0.0) finds its per-epoch histories
    for row in pd.read_csv(csv)[lambda df: df["phase"].str.match(r"[123]_")].to_dict("records"):
        conv = lpt.target_convergence(f"{tmp_path}/", "toy", row, [0, 1], start_from_epoch=1)
        assert len(conv) == 2 and (conv["peak_val_mrr"] >= conv["val_mrr_at_selected"]).all()
    assert (final["test_mrr"] == final["test_team_mrr"]).all()      # one target: pooled = per target


@pytest.mark.parametrize("kg", KGS)
def test_leak_check_finds_nothing_after_hiding_and_everything_without(lp, kg):
    d = lp[kg]
    hide = _hide(d, {R(10), R(11)})
    assert set(lpt.leaked_facts(d, hide).values()) == {0}
    unhidden = replace(d, targets={n: replace(t, edge_fact=t.edge_fact[:, :0]) for n, t in d.targets.items()},
                       member_edges=d.member_edges[:, :0])
    on = {n: int(hide[t.race].sum()) for n, t in d.targets.items()}
    # with no edges known to hide, each hidden fact is found once (Pure's team through its Role node)
    assert lpt.leaked_facts(unhidden, hide) == {**on, "team_via_member": 0}


@pytest.mark.parametrize("kg", ["pure_sulo", "hybrid_sulo"])
def test_leak_check_finds_a_member_pair_only_the_hidden_entries_support(lp, kg):
    """Driver 2 drove for Team 1 only in races 10 and 11 (entries 102, 104): with member pairs never
    hidden, hiding both races leaves Team 1 - Driver 2 visible, two hops from both hidden entries.
    Driver 1's pair stays explained by entry 100 (race 1)."""
    d = lp[kg]
    hide = _hide(d, {R(10), R(11)})
    assert lpt.leaked_facts(d, hide)["team_via_member"] == 0
    never_hidden = replace(d, member_edges=d.member_edges[:, :0])
    assert lpt.leaked_facts(never_hidden, hide)["team_via_member"] == 2


def test_frequency_baseline_ranks_by_earlier_answers():
    lp = _planted(shuffle=False)
    m = lpt.frequency_baseline(lp, "test")["team"]
    t = lp.targets["team"]
    counts = torch.bincount(t.answer[lp.race_split[t.race] < 2], minlength=2)
    test = t.answer[lp.race_split[t.race] == 2]
    majority = int(counts.argmax())
    assert m["hits@1"] == pytest.approx(float((test == majority).float().mean()))


# ── Enrichment ablation ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ablation_lp(built, schemas):
    entities, assertions = built
    return {variant: build_lp_data(base, entities, assertions, SPLIT, ablation_schema(variant, schemas))
            for variant, (base, _, _) in ABLATION_VARIANTS.items()}


def iris(lp, ids):
    iri_of = {i: iri for iri, i in lp.node_ids.items()}
    return [iri_of[i] for i in ids.tolist()]


@pytest.mark.parametrize("variant", ABLATION_VARIANTS)
def test_every_ablation_variant_asks_the_same_queries_and_hides_them(ablation_lp, schemas, variant):
    d, ref = ablation_lp[variant], ablation_lp["adhoc"]
    assert d.races == ref.races and torch.equal(d.race_split, ref.race_split)
    for name, t in d.targets.items():          # node ids differ between graphs, IRIs must not
        r = ref.targets[name]
        assert iris(d, t.head) == iris(ref, r.head) and iris(d, t.candidates) == iris(ref, r.candidates)
        assert torch.equal(t.answer, r.answer) and torch.equal(t.race, r.race)
    has_members = "hasTeamMember" in ablation_schema(variant, schemas).properties
    assert (d.member_edges.size(1) > 0) == has_members
    for hidden in (d.race_split >= 1, d.race_split == 2):   # training and test-time hiding
        assert set(lpt.leaked_facts(d, hidden).values()) == {0}


def _planted_factory(hidden_list, bases):
    dim = hidden_list[-1] if hidden_list else 8
    return LinkPredictor(RGCNModel(num_keys=5, feature_dim=8, hidden_channels_list=hidden_list, num_relations=4,
                                   num_bases=bases, out_channels=dim), 1, dim)


def _run_lp_ablation(tmp_path, lp, seeds=(0, 1), lr=0.05):
    with contextlib.redirect_stdout(io.StringIO()):
        return lpt.run_lp_ablation_variant(
            _planted_factory, lp, "v", [8], None, lr, 0.0, epochs=6, patience=6, start_from_epoch=1,
            disjoint_ratio=0.3, seeds=seeds, save_dir=f"{tmp_path}/", csv_path=tmp_path / "ablation.csv")


def _stub_lp_runs(monkeypatch, status_of_seed):
    """run_lp_config_over_seeds stubbed: saves a fresh model as the best checkpoint (or fails with the
    given status). Returns the seeds run."""
    runs = []

    def fake(model_factory, lp, hidden_list, bases, lr, wd, epochs, patience, start_from_epoch, disjoint_ratio,
             seeds, save_path):
        (seed,) = seeds
        runs.append(seed)
        status = status_of_seed.get(seed, "ok")
        if status == "ok":
            torch.save({"model_state_dict": model_factory(hidden_list, bases).state_dict()},
                       f"{save_path.removesuffix('.pt')}_seed{seed}.pt")
        return {"val_mrr": 0.5 if status == "ok" else None, "best_epoch": 3 if status == "ok" else None,
                "max_grad_norm": None, "runtime_sec": 1.0, "status": status}

    monkeypatch.setattr(lpt, "run_lp_config_over_seeds", fake)
    return runs


def test_lp_ablation_scores_the_best_val_checkpoint_on_test_once_per_seed(cpu, tmp_path):
    lp = _planted(shuffle=False, n=200)
    df = _run_lp_ablation(tmp_path, lp)
    assert list(df["seed"]) == [0, 1] and (df["status"] == "ok").all()
    assert (df["layers"] == 2).all() and (df["num_edges"] == lp.data.edge_index.size(1)).all()
    for row in df.to_dict("records"):
        checkpoint = torch.load(f"{tmp_path}/v_seed{row['seed']}.pt", weights_only=False)
        model = _planted_factory([8], None)
        model.load_state_dict(checkpoint["model_state_dict"])
        test = evaluate_lp(model, lp, "test")
        assert row["test_mrr"] == pytest.approx(test["pooled"]["mrr"])
        assert row["test_team_mrr"] == pytest.approx(test["team"]["mrr"])
        assert row["best_epoch"] >= 1                                         # warm-up
        assert row["val_mrr"] == pytest.approx(checkpoint["history"]["val_mrr"][row["best_epoch"]])
    saved = pd.read_csv(tmp_path / "ablation.csv")
    assert list(saved["seed"]) == [0, 1] and saved["test_mrr"].tolist() == pytest.approx(df["test_mrr"].tolist())


def test_lp_ablation_resume_skips_finished_seeds_and_retries_failed_ones(monkeypatch, tmp_path):
    lp = _planted(shuffle=False, n=200)
    _stub_lp_runs(monkeypatch, {1: "oom"})
    first = _run_lp_ablation(tmp_path, lp)
    assert first["status"].tolist() == ["ok", "oom"]
    assert first["test_mrr"].isna().tolist() == [False, True]               # an OOM is logged, not scored
    runs = _stub_lp_runs(monkeypatch, {})
    second = _run_lp_ablation(tmp_path, lp)
    assert runs == [1]
    assert second["status"].tolist() == ["ok", "ok"] and list(second["seed"]) == [0, 1]


def test_lp_ablation_refuses_to_resume_rows_of_another_configuration(monkeypatch, tmp_path):
    lp = _planted(shuffle=False, n=200)
    _stub_lp_runs(monkeypatch, {})
    _run_lp_ablation(tmp_path, lp)
    with pytest.raises(ValueError, match="another configuration"):
        _run_lp_ablation(tmp_path, lp, lr=0.01)


def test_lp_ablation_summary_deltas_per_target_against_the_control():
    base = [0.30, 0.32, 0.25, 0.27, 0.40, 0.42]     # ctrl, ctrl, minus, minus, plus, plus
    rows = pd.DataFrame({"variant": ["ctrl", "ctrl", "minus", "minus", "plus", "plus"],
                         **{m: [v + i / 10 for v in base] for i, m in enumerate(LP_ABLATION_METRICS)}})
    summary = gnn_tools.summarize_ablation(rows, {"ctrl": "ctrl", "minus": "ctrl", "plus": "ctrl"},
                                           metrics=LP_ABLATION_METRICS, delta_metrics=LP_ABLATION_METRICS)
    for i, m in enumerate(LP_ABLATION_METRICS):
        assert summary[f"delta_{m}"].tolist() == pytest.approx([0.0, -0.05, 0.10])
        assert summary.loc["ctrl", f"{m}_mean"] == pytest.approx(0.31 + i / 10)


# ── Mechanism checks ──────────────────────────────────────────────────────

@pytest.mark.parametrize("split", ["val", "test"])
def test_query_ranks_are_evaluate_lps_queries_and_metrics(cpu, lp, split):
    d = lp["hybrid_sulo"]
    gnn_tools.set_seed(0)
    model = LinkPredictor(RGCNModel(num_keys=len(d.vocabulary), feature_dim=8, hidden_channels_list=[8],
                                    num_relations=d.num_relations, out_channels=8), len(TARGETS), 8)
    ranks, metrics = query_ranks(model, d, split), evaluate_lp(model, d, split)
    for name, t in d.targets.items():
        head, rank = ranks[name]
        assert torch.equal(head, t.head[d.race_split[t.race] == SPLITS.index(split)])
        assert ranking_metrics(rank) == pytest.approx(metrics[name])
    assert ranking_metrics(torch.cat([r for _, r in ranks.values()])) == pytest.approx(metrics["pooled"])


def test_evidence_hops_on_the_three_schemas(schemas):
    """Direct properties are 1 hop, SULO paths 2: Pure spells every literal and Role as a path, Hybrid only the rank."""
    expected = {                                   # (Ad-hoc, Hybrid, Pure)
        "LP team: the driver's literals, from the entry": (2, 2, 4),
        "LP team: the team's literals, from a candidate": (1, 1, 2),
        "LP position: a session's rank, from the entry": (3, 4, 4),
        "LP position: the rank, from a candidate": (1, 2, 2),
        "LP circuit: the race's literals, from the race": (1, 1, 2),
        "LP circuit: the circuit's literals, from a candidate": (1, 1, 2),
        "NP: the entry's own grid rank": (2, 3, 3),
    }
    assert set(EVIDENCE_CHAINS) == set(expected)
    for name, chain in EVIDENCE_CHAINS.items():
        assert tuple(evidence_hops(schemas[kg], chain) for kg in ("adhoc", "hybrid_sulo", "pure_sulo")) == expected[name]
    assert evidence_hops(schemas["pure_sulo"], ()) == 0


def test_session_position_entries_need_a_session_with_a_position():
    S = lambda i: NS + f"SessionResult/{i}"
    rows = [("E1", "hasSessionResult", S(1)), (S(1), "hasSessionType", "Qualifying"), (S(1), "hasPosition", "P3"),
            ("E2", "hasSessionResult", S(2)), (S(2), "hasSessionType", "Sprint"), (S(2), "hasPosition", "P1"),
            ("E3", "hasSessionResult", S(3)), (S(3), "hasSessionType", "Qualifying"),        # no position
            ("E4", "hasDriver", "D1")]                                                       # no session
    assertions = pd.DataFrame(rows, columns=["subject", "property", "value"])
    assert session_position_entries(assertions) == {"E1", "E2"}
    assert session_position_entries(assertions, "Qualifying") == {"E1"}
    assert session_position_entries(assertions, "FP1") == set()
