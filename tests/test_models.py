import contextlib
import io

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from node_prediction import gnn_tools
from node_prediction.models import GraphSAGEModel, RGCNModel

MODELS = {"rgcn": RGCNModel, "graphsage": GraphSAGEModel}


@pytest.fixture
def cpu(monkeypatch):
    """Train on CPU: GPU scatter ops are not bit-reproducible, CPU ones are."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _toy(planted, n_targets=2500, seed=0):
    """
    Target nodes (all with feature key 0) each link to one of two hub nodes
    (keys 1 and 2) through relation 0 / inverse 1. Planted: the label is which
    hub; otherwise labels are random. Split 2000 / 250 / 250.
    """
    gen = torch.Generator().manual_seed(seed)
    hub_of = torch.randint(0, 2, (n_targets,), generator=gen)
    targets, hubs = torch.arange(n_targets), n_targets + hub_of
    edge_index = torch.cat([torch.stack([targets, hubs]), torch.stack([hubs, targets])], dim=1)
    edge_type = torch.cat([torch.zeros(n_targets), torch.ones(n_targets)]).long()
    y = torch.zeros(n_targets + 2)
    y[:n_targets] = hub_of.float() if planted else torch.randint(0, 2, (n_targets,), generator=gen).float()
    key_id = torch.zeros(n_targets + 2, dtype=torch.long)
    key_id[n_targets:] = torch.tensor([1, 2])
    data = Data(edge_index=edge_index, edge_type=edge_type, y=y, num_nodes=n_targets + 2, key_id=key_id,
                value=torch.zeros(n_targets + 2), degree_feat=torch.zeros(n_targets + 2))
    perm = torch.randperm(n_targets, generator=gen)
    return data, perm[:2000], perm[2000:2250], perm[2250:]


def _train_and_test(model_cls, data, train, val, test, seed, epochs=8):
    gnn_tools.set_seed(seed)
    model = model_cls(num_keys=3, feature_dim=8, hidden_channels_list=[8], num_relations=2)
    with contextlib.redirect_stdout(io.StringIO()):
        model, history = gnn_tools.train_model(model, data, train, val, epochs=epochs, lr=0.01, patience=epochs)
        metrics = gnn_tools.evaluate_test_set(model, data, test)
    return model, history, metrics


@pytest.mark.parametrize("model_cls", MODELS.values(), ids=MODELS.keys())
def test_forward_gives_one_logit_per_node(model_cls):
    data, *_ = _toy(planted=True, n_targets=10)
    model = model_cls(num_keys=3, feature_dim=4, hidden_channels_list=[4, 4], num_relations=2)
    assert model(data).shape == (12, 1)


@pytest.mark.parametrize("model_cls", MODELS.values(), ids=MODELS.keys())
def test_learns_a_planted_signal(cpu, model_cls):
    _, _, metrics = _train_and_test(model_cls, *_toy(planted=True), seed=0)
    assert metrics["roc_auc"] > 0.95


@pytest.mark.parametrize("model_cls", MODELS.values(), ids=MODELS.keys())
def test_random_labels_stay_at_chance(cpu, model_cls):
    # Leakage canary: if labels reached the model's inputs, test AUC would rise above chance.
    # Null AUC sd with 250 test nodes is ~0.037, so 0.15 is a 4-sd bound.
    _, _, metrics = _train_and_test(model_cls, *_toy(planted=False), seed=0)
    assert abs(metrics["roc_auc"] - 0.5) < 0.15


@pytest.mark.parametrize("model_cls", MODELS.values(), ids=MODELS.keys())
def test_training_is_reproducible_per_seed(cpu, model_cls):
    toy = _toy(planted=False)
    _, history_a, metrics_a = _train_and_test(model_cls, *toy, seed=1)
    _, history_b, metrics_b = _train_and_test(model_cls, *toy, seed=1)
    _, history_c, _ = _train_and_test(model_cls, *toy, seed=2)
    assert history_a == history_b and metrics_a == metrics_b
    assert history_a["train_loss"] != history_c["train_loss"]


@pytest.mark.parametrize("model_cls", MODELS.values(), ids=MODELS.keys())
def test_only_training_labels_reach_the_gradients(cpu, model_cls):
    # Full-batch forward passes cover every node, so the loss must still read the train labels only:
    # flipping every val/test label leaves the training trajectory and the final weights bit-identical.
    data, train, val, test = _toy(planted=True)
    flipped = data.clone()
    held_out = torch.cat([val, test])
    flipped.y[held_out] = 1 - flipped.y[held_out]
    _, history, _ = _train_and_test(model_cls, data, train, val, test, seed=0, epochs=3)
    _, history_flipped, _ = _train_and_test(model_cls, flipped, train, val, test, seed=0, epochs=3)
    assert history["train_loss"] == history_flipped["train_loss"]
    assert history["grad_norm"] == history_flipped["grad_norm"]

    # Weights are compared via train_final_model: train_model restores the best-val-epoch weights,
    # and which epoch that is legitimately depends on the (flipped) val labels.
    weights = []
    for graph in (data, flipped):
        gnn_tools.set_seed(0)
        model = model_cls(num_keys=3, feature_dim=8, hidden_channels_list=[8], num_relations=2)
        with contextlib.redirect_stdout(io.StringIO()):
            weights.append(gnn_tools.train_final_model(model, graph, train, epochs=3, lr=0.01).state_dict())
    assert all(torch.equal(weights[0][k], weights[1][k]) for k in weights[0])


@pytest.mark.parametrize("model_cls", MODELS.values(), ids=MODELS.keys())
def test_early_stopping_returns_the_best_epochs_weights(cpu, model_cls):
    # Random labels: val ROC-AUC peaks early, so patience stops long before the epoch cap. The
    # returned weights must be those after best_epoch + 1 steps -- which train_final_model replays
    # bit-exactly from the same seed (evaluation draws no random numbers) -- not the last epoch's.
    data, train, val, _ = _toy(planted=False)
    patience = 3

    def fresh():
        gnn_tools.set_seed(0)
        return model_cls(num_keys=3, feature_dim=8, hidden_channels_list=[8], num_relations=2)

    with contextlib.redirect_stdout(io.StringIO()):
        restored, history = gnn_tools.train_model(fresh(), data, train, val, epochs=100, lr=0.01,
                                                  weight_decay=1e-4, patience=patience)
        replayed = gnn_tools.train_final_model(fresh(), data, train, epochs=history["best_epoch"] + 1,
                                               lr=0.01, weight_decay=1e-4)
        untrained = gnn_tools.train_final_model(fresh(), data, train, epochs=0, lr=0.01)
    assert len(history["train_loss"]) == history["best_epoch"] + 1 + patience < 100
    assert all(torch.equal(restored.state_dict()[k], v) for k, v in replayed.state_dict().items())
    assert all(torch.equal(untrained.state_dict()[k], v) for k, v in fresh().state_dict().items())


@pytest.mark.parametrize("model_cls", MODELS.values(), ids=MODELS.keys())
def test_evaluation_is_deterministic(model_cls):
    # Full-batch evaluation in eval mode: dropout is off and nothing is sampled, so evaluations under
    # different RNG states score identically even with a high dropout rate.
    data, _, _, test = _toy(planted=False)
    model = model_cls(num_keys=3, feature_dim=8, hidden_channels_list=[8], num_relations=2, dropout=0.9)
    scores = []
    for seed in (0, 1):
        gnn_tools.set_seed(seed)
        scores.append(gnn_tools.evaluate_test_set(model, data, test))
    assert scores[0] == scores[1]


def _hubs_told_apart_by_one_edge(with_rank_edges, n_targets=2500, seed=0):
    """
    Like Position/<n>: two hubs with the same feature key, 1,250 target neighbours each, and one
    edge each (relation 2 / inverse 3) to a rank literal (key 2, value +1 or -1). The label is
    which hub a target links to, so only the rank edge can tell the classes apart. The hubs have
    exactly equal degrees: means over different counts of identical vectors round differently in
    the last bit, which a rank metric like ROC-AUC would pick up as a perfect signal.
    """
    gen = torch.Generator().manual_seed(seed)
    hub_of = torch.arange(n_targets) % 2
    targets, hubs = torch.arange(n_targets), n_targets + hub_of
    edge_index = torch.cat([torch.stack([targets, hubs]), torch.stack([hubs, targets])], dim=1)
    edge_type = torch.cat([torch.zeros(n_targets), torch.ones(n_targets)]).long()
    if with_rank_edges:
        hub_ids, literal_ids = torch.tensor([n_targets, n_targets + 1]), torch.tensor([n_targets + 2, n_targets + 3])
        edge_index = torch.cat([edge_index, torch.stack([hub_ids, literal_ids]), torch.stack([literal_ids, hub_ids])], dim=1)
        edge_type = torch.cat([edge_type, torch.tensor([2, 2, 3, 3])])
    data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=n_targets + 4,
                y=torch.cat([hub_of.float(), torch.zeros(4)]),
                key_id=torch.cat([torch.zeros(n_targets, dtype=torch.long), torch.tensor([1, 1, 2, 2])]),
                value=torch.cat([torch.zeros(n_targets + 2), torch.tensor([1.0, -1.0])]),
                degree_feat=torch.zeros(n_targets + 4))
    perm = torch.randperm(n_targets, generator=gen)
    return data, perm[:2000], perm[2000:2250], perm[2250:]


def test_rgcn_reads_a_hub_identity_behind_one_rare_edge(cpu):
    # Regression test for neighbour sampling: a fanout of 10 at the hub almost never drew the rank
    # edge. Full-batch R-GCN averages per relation, so the single rank message reaches the hub
    # undiluted. GraphSAGE is left out on purpose: its one mean over all neighbours dilutes it ~1/1250.
    def auc(with_rank_edges):
        data, train, val, test = _hubs_told_apart_by_one_edge(with_rank_edges)
        gnn_tools.set_seed(0)
        model = RGCNModel(num_keys=3, feature_dim=8, hidden_channels_list=[8], num_relations=4)
        with contextlib.redirect_stdout(io.StringIO()):
            model, _ = gnn_tools.train_model(model, data, train, val, epochs=50, lr=0.01, patience=50)
            return gnn_tools.evaluate_test_set(model, data, test)["roc_auc"]
    assert auc(with_rank_edges=True) > 0.95
    assert abs(auc(with_rank_edges=False) - 0.5) < 0.15   # identical hubs: no signal without the edge


def test_graphsage_layer_is_the_raw_mean_aggregator():
    # Hamilton et al.'s GraphSAGE-mean: h'_v = W_l * mean_{u in N(v)} h_u + b + W_r * h_v, with no learned
    # layer or ReLU on the neighbours before the mean (that would be project=True / mean-pooling).
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(4, 5, generator=gen) - 1.0                # mostly negative: a ReLU before the mean would show
    edge_index = torch.tensor([[1, 2, 3, 0], [0, 0, 0, 1]])   # 0 <- {1, 2, 3}, 1 <- {0}, nodes 2 and 3 get nothing
    conv = GraphSAGEModel(num_keys=1, feature_dim=4, hidden_channels_list=[3], num_relations=2).convs[0]
    expected = torch.stack([
        conv.lin_l(x[[1, 2, 3]].mean(0)) + conv.lin_r(x[0]),
        conv.lin_l(x[0]) + conv.lin_r(x[1]),
        conv.lin_l(torch.zeros(5)) + conv.lin_r(x[2]),        # no neighbours: the mean is 0, only the bias remains
        conv.lin_l(torch.zeros(5)) + conv.lin_r(x[3]),
    ])
    assert torch.allclose(conv(x, edge_index), expected, atol=1e-6)


def test_training_logs_gradient_norms_without_clipping(cpu, monkeypatch):
    def no_clipping(*args, **kwargs):
        raise AssertionError("gradients must not be clipped")
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", no_clipping)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_value_", no_clipping)
    _, history, _ = _train_and_test(RGCNModel, *_toy(planted=True), seed=0, epochs=3)
    assert len(history["grad_norm"]) == 3
    assert all(norm > 0 for norm in history["grad_norm"])


def _rgcn_without_checkpointing(model, data):
    """RGCNModel.forward with every activation kept for backward: the reference the checkpointed model must match."""
    x = model.encoder(data)
    for conv in model.convs[:-1]:
        x = F.dropout(F.relu(conv(x, data.edge_index, data.edge_type)), p=model.dropout, training=model.training)
    return model.convs[-1](x, data.edge_index, data.edge_type)


def _logits_and_grads(model, data, forward, seed):
    model.zero_grad()
    torch.manual_seed(seed)                                  # same dropout masks for both forwards
    out = forward(model, data)
    out.square().sum().backward()
    return out.detach(), [p.grad.clone() for p in model.parameters()]


@pytest.mark.parametrize("num_bases", [None, 2])
def test_rgcn_checkpointing_changes_neither_logits_nor_gradients(num_bases):
    data, *_ = _toy(planted=True, n_targets=200)
    data.value = torch.randn(data.num_nodes, generator=torch.Generator().manual_seed(1))
    gnn_tools.set_seed(0)
    model = RGCNModel(num_keys=3, feature_dim=8, hidden_channels_list=[8, 8], num_relations=2,
                      num_bases=num_bases, dropout=0.5).train()
    out, grads = _logits_and_grads(model, data, RGCNModel.forward, seed=3)
    ref_out, ref_grads = _logits_and_grads(model, data, _rgcn_without_checkpointing, seed=3)
    assert torch.equal(out, ref_out)
    assert all(torch.allclose(g, r, rtol=0, atol=1e-6) for g, r in zip(grads, ref_grads))
    assert any(g.abs().sum() > 0 for g in grads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_rgcn_checkpointing_lowers_peak_training_memory():
    # With bases, RGCNConv keeps one N x d activation per relation for backward; checkpointing keeps only
    # each layer's input and recomputes one layer at a time.
    gen = torch.Generator().manual_seed(0)
    n, e, r = 20_000, 200_000, 8
    data = Data(edge_index=torch.randint(0, n, (2, e), generator=gen),
                edge_type=torch.randint(0, r, (e,), generator=gen), num_nodes=n,
                key_id=torch.zeros(n, dtype=torch.long), value=torch.zeros(n), degree_feat=torch.zeros(n)).to("cuda")
    model = RGCNModel(num_keys=1, feature_dim=64, hidden_channels_list=[64, 64], num_relations=r,
                      num_bases=2).to("cuda").train()

    def peak(forward):
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        _logits_and_grads(model, data, forward, seed=0)
        return torch.cuda.max_memory_allocated()
    peak(RGCNModel.forward)                                   # warm-up: CUDA/cuBLAS workspaces
    assert peak(RGCNModel.forward) < 0.6 * peak(_rgcn_without_checkpointing)
