import contextlib
import io

import numpy as np
import pandas as pd
import pytest
import torch

from node_prediction import gnn_tools

DEPTH_SCORE = {1: 0.60, 2: 0.70, 3: 0.80}


def _search(monkeypatch, tmp_path, score, grid_dim=(32, 64), grid_bases=(None, 30), default_bases=30,
            grid_lr=(1e-3,), grid_wd=(1e-4, 1e-3), default_lr=1e-3, tuning_seeds=(0,), best_epoch=None,
            final_row=lambda kw: {'phase': '4'}, start_from_epoch=0, warmups=None):
    """
    Runs the sequential search with training stubbed out: `score(dim, depth, bases, lr, wd, seed)`
    is the validation ROC-AUC of a config (or None for a failed/OOM run); its best epoch is
    `best_epoch(seed)` if given, else the depth. `final_row(kwargs)` is the stubbed phase-4 row.
    `warmups`, if given, collects the warm-up every run was trained with.
    Returns (results DataFrame, list of trained configs, kwargs passed to the final evaluation);
    each config appears once in the list however many seeds it was trained with.
    """
    trained, final = [], {}

    def fake_run_one_config(model_factory, graph_data, train_indices, val_indices, hidden_list, bases,
                            lr, wd, epochs, patience, save_path, seed=0, start_from_epoch=0):
        if warmups is not None:
            warmups.append(start_from_epoch)
        depth = len(hidden_list) + 1
        dim = hidden_list[0] if hidden_list else None
        if seed == tuning_seeds[0]:
            trained.append((dim, depth, bases, lr, wd))
        auc = score(dim, depth, bases, lr, wd, seed)
        epoch = depth if best_epoch is None else best_epoch(seed)
        return {'val_roc_auc': auc, 'best_epoch': None if auc is None else epoch, 'max_grad_norm': None,
                'runtime_sec': 1.0, 'status': 'ok' if auc is not None else 'oom'}

    monkeypatch.setattr(gnn_tools, "run_one_config", fake_run_one_config)
    monkeypatch.setattr(gnn_tools, "evaluate_final_test_row", lambda **kw: final.update(kw) or final_row(kw))
    with contextlib.redirect_stdout(io.StringIO()):
        df = gnn_tools.run_sequential_optimization_with_verification(
            model_factory=None, graph_data=None, train_indices=None, val_indices=None, test_indices=None,
            schema_label='t', save_dir=f"{tmp_path}/", csv_path=tmp_path / "log.csv",
            grid_dim=list(grid_dim), grid_depth=[1, 2, 3],
            grid_bases=list(grid_bases), grid_lr=list(grid_lr), grid_wd=list(grid_wd),
            default_bases=default_bases, default_lr=default_lr, default_wd=1e-4,
            epochs=1, patience=1, tuning_seeds=tuning_seeds, start_from_epoch=start_from_epoch,
        )
    return df, trained, final


def _depth3_wins(dim, depth, bases, lr, wd, seed=0):
    return DEPTH_SCORE[depth] + (0.01 if dim == 64 else 0) - (0.05 if bases is None else 0) + (0.001 if wd == 1e-4 else 0)


def test_phase3_cache_only_reuses_runs_of_the_same_depth(monkeypatch, tmp_path):
    df, _, final = _search(monkeypatch, tmp_path, _depth3_wins)
    phase3 = df[df["phase"].str.startswith("3_")]
    assert set(phase3["layers"]) == {3}
    for _, row in phase3.iterrows():                       # every phase-3 score is a depth-3 score
        expected = _depth3_wins(row["hidden_dim"], 3, 30, row["lr"], row["weight_decay"])
        assert row["val_roc_auc"] == pytest.approx(expected)
        assert row["best_epoch"] == 3
    assert (final["hidden_dim"], final["depth"], final["weight_decay"]) == (64, 3, 1e-4)
    assert final["best_epoch"] == 3


def test_failed_configs_are_logged_without_aborting_the_search(monkeypatch, tmp_path):
    def fail_depth3(dim, depth, bases, lr, wd, seed=0):
        return None if depth == 3 else _depth3_wins(dim, depth, bases, lr, wd)
    df, _, final = _search(monkeypatch, tmp_path, fail_depth3)
    assert (df["status"] == "oom").any()
    assert final["depth"] == 2


def test_failed_run_with_zero_auc_is_not_mistaken_for_a_failure(monkeypatch, tmp_path):
    df, _, _ = _search(monkeypatch, tmp_path, lambda *config: 0.0)
    assert (df.loc[df["phase"] != "4", "status"] == "ok").all()   # phase 4 is the stubbed final eval


def test_phase3_verifies_regularisation_combos_from_all_phase2_sweeps(monkeypatch, tmp_path):
    # lr 5e-3 (phase 2b) and bases None (phase 2a) are the best regularisation settings, wd never matters
    def score(dim, depth, bases, lr, wd, seed=0):
        return DEPTH_SCORE[depth] + (0.02 if bases is None else 0) + (0.01 if lr == 5e-3 else 0)
    df, _, _ = _search(monkeypatch, tmp_path, score, grid_lr=(1e-3, 5e-3), grid_wd=(1e-4,))
    phase3 = df[df["phase"].str.startswith("3_")]
    combos = {(None if b != b else b, lr) for b, lr in zip(phase3["num_bases"], phase3["lr"])}
    assert len(combos) == 2                                # not only weight-decay variants


def test_architectures_without_bases_spend_no_run_on_bases(monkeypatch, tmp_path):
    _, trained, _ = _search(monkeypatch, tmp_path, _depth3_wins, grid_bases=[None], default_bases=None)
    assert all(bases is None for *_, bases, lr, wd in trained)
    phase1_and_2_runs = len(trained) - 4                   # minus the (at most) four phase-3 runs
    # 2 dims x 2 depths + one 1-layer run + 0 bases + 0 lr + 1 wd = 6 training runs before phase 3
    assert phase1_and_2_runs <= 6


def test_one_layer_winner_verifies_each_regularisation_combo_once(monkeypatch, tmp_path):
    # A 1-layer model has no hidden layer, so the dim axis of the 2x2 grid would train identical models.
    def depth1_wins(dim, depth, bases, lr, wd, seed=0):
        return {1: 0.80, 2: 0.70, 3: 0.60}[depth] + (0.01 if dim == 64 else 0) + (0.001 if wd == 1e-4 else 0)
    df, trained, final = _search(monkeypatch, tmp_path, depth1_wins)
    assert len(trained) == len(set(trained))               # no configuration is trained twice
    phase3 = df[df["phase"].str.startswith("3_")]
    assert len(phase3) == 2                                # one row per top regularisation combo
    assert set(phase3["layers"]) == {1}
    assert (final["depth"], final["weight_decay"]) == (1, 1e-4)


def test_grad_norm_is_the_l2_norm_over_all_parameter_gradients():
    model = torch.nn.Linear(1, 1)
    model.weight.grad = torch.tensor([[3.0]])
    model.bias.grad = torch.tensor([4.0])
    assert gnn_tools._grad_norm(model) == pytest.approx(5.0)
    model.bias.grad = None                                  # parameters without a gradient are skipped
    assert gnn_tools._grad_norm(model) == pytest.approx(3.0)
    model.weight.grad = torch.zeros(1, 1)
    assert gnn_tools._grad_norm(model) == 0.0


def test_binary_metrics_at_their_bounds():
    targets = np.array([0, 0, 1, 1])
    perfect = gnn_tools._binary_metrics(targets, np.array([0.1, 0.2, 0.8, 0.9]))
    assert perfect == pytest.approx(dict.fromkeys(["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"], 1.0))
    assert gnn_tools._binary_metrics(targets, np.array([0.9, 0.8, 0.2, 0.1]))["roc_auc"] == 0.0
    constant = gnn_tools._binary_metrics(targets, np.full(4, 0.3))   # no ranking; all predicted negative
    assert constant["roc_auc"] == 0.5
    assert constant["pr_auc"] == 0.5                                   # = the positive rate
    assert constant["accuracy"] == 0.5
    assert constant["precision"] == constant["recall"] == constant["f1"] == 0.0


def test_dims_are_compared_at_every_depth(monkeypatch, tmp_path):
    # Regression test: dims used to be ranked at depth 2 only, so a dim that wins only at depth 3
    # (where Pure/Hybrid SULO first see the grid rank) could never be chosen.
    def score(dim, depth, bases, lr, wd, seed=0):
        by_dim = {2: {32: 0.03, 64: 0.02, 128: 0.0}, 3: {32: 0.0, 64: 0.0, 128: 0.03}}
        return DEPTH_SCORE[depth] + by_dim.get(depth, {}).get(dim, 0)
    df, trained, final = _search(monkeypatch, tmp_path, score, grid_dim=(32, 64, 128))
    phase1 = df[df["phase"] == "1_Architecture"]
    assert len(phase1) == 3 * 2 + 1                        # every dim at depths 2 and 3, one 1-layer run
    assert (final["hidden_dim"], final["depth"]) == (128, 3)


def test_configs_are_scored_by_their_mean_over_seeds(monkeypatch, tmp_path):
    # Depth 3 has the best single seed but collapses on seed 2; depth 2 is stable and wins on the mean.
    def score(dim, depth, bases, lr, wd, seed=0):
        if depth == 3:
            return 0.90 if seed < 2 else 0.15
        return DEPTH_SCORE[depth]
    df, _, final = _search(monkeypatch, tmp_path, score, tuning_seeds=(0, 1, 2), best_epoch=lambda s: [300, 7, 320][s])
    depth3 = df[(df["phase"] == "1_Architecture") & (df["layers"] == 3)].iloc[0]
    assert depth3["val_roc_auc"] == pytest.approx(0.65)
    assert depth3["val_roc_auc_seeds"] == "0.9000;0.9000;0.1500"
    assert final["depth"] == 2
    assert final["best_epoch"] == 300                      # median of 300, 7, 320, not the mean (209)


def _over_seeds(monkeypatch, outcomes, seeds=(0, 1, 2)):
    """run_config_over_seeds with run_one_config stubbed: outcomes[seed] = (val ROC-AUC, best epoch) or None (OOM)."""
    calls = []
    def fake(*args, save_path, seed, **kwargs):
        calls.append((seed, save_path))
        auc, epoch = outcomes[seed] or (None, None)
        return {'val_roc_auc': auc, 'best_epoch': epoch, 'max_grad_norm': None if auc is None else float(seed),
                'runtime_sec': 2.0, 'status': 'ok' if auc is not None else 'oom'}
    monkeypatch.setattr(gnn_tools, "run_one_config", fake)
    row = gnn_tools.run_config_over_seeds(None, None, None, None, [8], None, 1e-3, 0.0, epochs=1, patience=1,
                                          save_path="dir/kg_1_L2_dim8.pt", seeds=seeds)
    return row, calls


def test_seed_aggregation(monkeypatch):
    row, calls = _over_seeds(monkeypatch, {0: (0.8, 300), 1: (0.7, 7), 2: (0.9, 320)})
    assert row["status"] == "ok"
    assert row["val_roc_auc"] == pytest.approx(0.8)
    assert row["val_roc_auc_std"] == pytest.approx(0.1)    # sample std, as in the final summary
    assert row["best_epoch"] == 300
    assert row["best_epoch_seeds"] == "300;7;320"
    assert row["max_grad_norm"] == 2.0 and row["runtime_sec"] == 6.0
    assert [p for _, p in calls] == [f"dir/kg_1_L2_dim8_seed{s}.pt" for s in range(3)]   # one checkpoint per seed


def test_even_seed_count_takes_an_epoch_a_seed_reached(monkeypatch):
    row, _ = _over_seeds(monkeypatch, {0: (0.8, 100), 1: (0.8, 201)}, seeds=(0, 1))
    assert row["best_epoch"] == 100                       # median_low, not 150.5


def test_one_seed_has_no_std(monkeypatch):
    row, _ = _over_seeds(monkeypatch, {0: (0.8, 10)}, seeds=(0,))
    assert row["val_roc_auc"] == 0.8 and row["val_roc_auc_std"] is None


def test_a_failed_seed_fails_the_config_and_stops_its_seeds(monkeypatch):
    row, calls = _over_seeds(monkeypatch, {0: (0.8, 10), 1: None, 2: (0.9, 10)})
    assert row["status"] == "oom"
    assert row["val_roc_auc"] is None and row["best_epoch"] is None
    assert [s for s, _ in calls] == [0, 1]


def test_runs_ended_by_epoch_cap():
    # epochs 400, patience 50: a seed whose best epoch is >= 350 cannot have stopped early.
    df = pd.DataFrame({
        'phase': ['1', '1', '1', '2a', '1'],
        'best_epoch_seeds': ['349;100', '350;100', '399', '399', None],
        'runtime_sec': [1.0, 1.0, 1.0, 0.0, 1.0],          # the 2a row reuses a phase-1 run
        'status': ['ok', 'ok', 'ok', 'ok', 'oom'],
    })
    capped = gnn_tools.runs_ended_by_epoch_cap(df, epochs=400, patience=50)
    assert list(capped.index) == [1, 2]


def _one_config(monkeypatch, tmp_path, error):
    def fail(**kwargs):
        raise error
    monkeypatch.setattr(gnn_tools, "train_model", fail)
    return gnn_tools.run_one_config(
        lambda hidden_list, bases: torch.nn.Linear(1, 1), None, None, None, hidden_list=[8], bases=None,
        lr=1e-3, wd=0.0, epochs=1, patience=1, save_path=str(tmp_path / "m.pt"),
    )


def test_cuda_oom_marks_the_config_and_returns(monkeypatch, tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        row = _one_config(monkeypatch, tmp_path, torch.cuda.OutOfMemoryError("CUDA out of memory."))
    assert row["status"] == "oom"
    assert row["val_roc_auc"] is None and row["best_epoch"] is None


def test_errors_other_than_oom_still_abort(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="device not ready"):
        _one_config(monkeypatch, tmp_path, RuntimeError("CUDA driver error: device not ready"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_cuda_memory_cap_turns_overcommit_into_oom():
    torch.cuda.empty_cache()
    try:
        free = gnn_tools.limit_cuda_memory_to_free_vram()
        _, total = torch.cuda.mem_get_info()
        assert 0 < free <= total
        with pytest.raises(torch.cuda.OutOfMemoryError):   # WSL2 would otherwise page this into RAM
            torch.empty(free + 2**30, dtype=torch.uint8, device="cuda")
        assert torch.ones(1, device="cuda").item() == 1.0   # the device is still usable afterwards
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)
        torch.cuda.empty_cache()


def test_phase2_reuses_the_winning_row_of_the_previous_sweep(monkeypatch, tmp_path):
    # lr 5e-3 beats the default 1e-3, so 2c's default-wd row must be the 2b lr-5e-3 run, not the default-lr one.
    def score(dim, depth, bases, lr, wd, seed=0):
        return DEPTH_SCORE[depth] + (0.05 if lr == 5e-3 else 0) + (0.01 if bases == 30 else 0)
    df, _, _ = _search(monkeypatch, tmp_path, score, grid_lr=(1e-3, 5e-3), grid_wd=(1e-4, 1e-3))
    lr_winner = df[(df["phase"] == "2b_LR") & (df["lr"] == 5e-3)].iloc[0]
    reused = df[(df["phase"] == "2c_WD") & (df["weight_decay"] == 1e-4)].iloc[0]
    assert reused["lr"] == 5e-3 and reused["num_bases"] == 30
    assert reused["val_roc_auc"] == lr_winner["val_roc_auc"] and reused["runtime_sec"] == 0.0


def test_finished_tuning_results_only_accepts_a_complete_run_of_this_protocol(monkeypatch, tmp_path):
    csv = tmp_path / "log.csv"
    args = dict(grid_dim=[32, 64], grid_depth=[1, 2, 3], tuning_seeds=(0,), final_seeds=(0,))
    assert gnn_tools.finished_tuning_results(csv, **args) is None                  # no file yet
    _search(monkeypatch, tmp_path, _depth3_wins, final_row=lambda kw: {
        'phase': '4_Final_Test_Evaluation', 'seed': kw['seed'], 'status': 'ok', 'runtime_sec': 0.0})
    assert gnn_tools.finished_tuning_results(csv, **args) is not None
    assert gnn_tools.finished_tuning_results(csv, **dict(args, grid_dim=[32, 64, 128])) is None   # other grid
    assert gnn_tools.finished_tuning_results(csv, **dict(args, tuning_seeds=(0, 1, 2))) is None   # other seed count
    assert gnn_tools.finished_tuning_results(csv, **dict(args, final_seeds=(0, 1))) is None       # phase 4 incomplete
    pd.read_csv(csv).drop(columns="val_roc_auc_seeds").to_csv(csv, index=False)                    # single-seed protocol
    assert gnn_tools.finished_tuning_results(csv, **args) is None



# ── Enrichment ablation ───────────────────────────────────────────────────

def _ablation_stub(monkeypatch, status_of_seed):
    """run_one_config stubbed: saves a Linear's weights as the 'best' checkpoint (or fails with the
    given status); evaluate_test_set returns the first weight as the test ROC-AUC. Returns the seeds run."""
    runs = []
    def fake_run(model_factory, graph_data, train, val, hidden_list, bases, lr, wd, epochs, patience, save_path, seed,
                 start_from_epoch):
        runs.append(seed)
        status = status_of_seed.get(seed, 'ok')
        if status == 'ok':
            best = torch.nn.Linear(1, 1)
            torch.nn.init.constant_(best.weight, 0.5 + seed / 10)
            torch.save({'model_state_dict': best.state_dict()}, save_path)
        return {'num_bases': bases, 'lr': lr, 'weight_decay': wd, 'val_roc_auc': 0.8 if status == 'ok' else None,
                'best_epoch': 3 if status == 'ok' else None, 'max_grad_norm': None, 'runtime_sec': 1.0, 'status': status}
    monkeypatch.setattr(gnn_tools, "run_one_config", fake_run)
    monkeypatch.setattr(gnn_tools, "evaluate_test_set",
                        lambda model, graph, test: {'roc_auc': model.weight.item(), 'f1': 0.0})
    return runs


def _ablation_kg():
    from f1_node_data import KGData
    from torch_geometric.data import Data
    data = Data(edge_index=torch.tensor([[0, 1], [1, 0]]), num_nodes=2)
    return KGData(data, torch.tensor([0]), torch.tensor([1]), torch.tensor([1]), 2, ['a'], {})


def _run_ablation(tmp_path, seeds=(0, 1)):
    with contextlib.redirect_stdout(io.StringIO()):
        return gnn_tools.run_ablation_variant(
            lambda hidden_list, bases: torch.nn.Linear(1, 1), _ablation_kg(), 'v', [8], None, 1e-3, 0.0,
            epochs=1, patience=1, start_from_epoch=0, seeds=seeds, save_dir=f"{tmp_path}/",
            csv_path=tmp_path / "ablation.csv")


def test_ablation_scores_the_best_val_checkpoint_on_test_once_per_seed(monkeypatch, tmp_path):
    runs = _ablation_stub(monkeypatch, {})
    df = _run_ablation(tmp_path)
    assert runs == [0, 1]
    assert list(df['seed']) == [0, 1]
    assert df['test_roc_auc'].tolist() == pytest.approx([0.5, 0.6])   # the checkpoint's weights, not a fresh init
    assert (df['layers'] == 2).all() and (df['num_edges'] == 2).all()
    saved = pd.read_csv(tmp_path / "ablation.csv")
    assert list(saved['seed']) == [0, 1] and saved['test_roc_auc'].tolist() == pytest.approx([0.5, 0.6])


def test_ablation_resume_skips_finished_seeds_and_retries_failed_ones(monkeypatch, tmp_path):
    _ablation_stub(monkeypatch, {1: 'oom'})
    first = _run_ablation(tmp_path)
    assert first['status'].tolist() == ['ok', 'oom']
    assert first['test_roc_auc'].isna().tolist() == [False, True]       # an OOM is logged, not scored
    runs = _ablation_stub(monkeypatch, {})
    second = _run_ablation(tmp_path)
    assert runs == [1]
    assert second['status'].tolist() == ['ok', 'ok'] and list(second['seed']) == [0, 1]


def test_ablation_summary_deltas_against_the_control():
    rows = pd.DataFrame({
        'variant': ['ctrl', 'ctrl', 'minus', 'minus', 'plus', 'plus'],
        **{m: [0.80, 0.82, 0.75, 0.77, 0.90, 0.92] for m in gnn_tools.ABLATION_METRICS},
    })
    summary = gnn_tools.summarize_ablation(rows, {'ctrl': 'ctrl', 'minus': 'ctrl', 'plus': 'ctrl'})
    assert list(summary.index) == ['ctrl', 'minus', 'plus']
    assert summary['delta_test_roc_auc'].tolist() == pytest.approx([0.0, -0.05, 0.10])
    assert summary['delta_val_roc_auc'].tolist() == pytest.approx([0.0, -0.05, 0.10])
    assert summary['test_roc_auc_std'].tolist() == pytest.approx([0.01414] * 3, abs=1e-5)
    assert summary['n_seeds'].tolist() == [2, 2, 2]


# ── Early-stopping warm-up (start_from_epoch) ─────────────────────────────

def _scripted_training(monkeypatch, val_aucs, epochs, patience, start_from_epoch, save_path=None):
    """train_model on a one-weight model whose val ROC-AUC at epoch e is val_aucs[e] and whose weight
    after epoch e's step is e, so the restored weight names the restored epoch."""
    model = torch.nn.Linear(1, 1, bias=False)
    def step(model, graph, indices, optimizer, loss_fn):
        with torch.no_grad():
            model.weight.fill_(len(history_epochs))
        history_epochs.append(None)
        return 0.0, 0.0
    history_epochs = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(gnn_tools, "_train_step", step)
    monkeypatch.setattr(gnn_tools, "_predict", lambda model, graph, idx: torch.zeros(len(idx)))
    monkeypatch.setattr(gnn_tools, "_binary_metrics", lambda targets, probs: {
        **dict.fromkeys(["accuracy", "precision", "recall", "f1", "pr_auc"], 0.0),
        "roc_auc": val_aucs[len(history_epochs) - 1]})
    from torch_geometric.data import Data
    graph = Data(y=torch.zeros(2), num_nodes=2)
    with contextlib.redirect_stdout(io.StringIO()):
        return gnn_tools.train_model(model, graph, torch.tensor([0]), torch.tensor([1]), epochs=epochs,
                                     patience=patience, start_from_epoch=start_from_epoch, save_path=save_path)


# A spike at epoch 2 that the run only passes at epoch 30, more than `patience` (25) epochs later: the
# ablation's false stops, scaled down.
SPIKE = [0.5, 0.6, 0.9] + [0.7] * 27 + [0.95] * 30


def test_without_warmup_an_early_spike_stops_the_run(monkeypatch):
    model, history = _scripted_training(monkeypatch, SPIKE, epochs=60, patience=25, start_from_epoch=0)
    assert history["best_epoch"] == 2
    assert len(history["train_loss"]) == 2 + 1 + 25           # stopped patience epochs after the spike
    assert model.weight.item() == 2


def test_warmup_epochs_are_neither_checkpointed_nor_counted_for_patience(monkeypatch):
    model, history = _scripted_training(monkeypatch, SPIKE, epochs=60, patience=25, start_from_epoch=10)
    assert history["best_epoch"] == 30                         # the spike at 2 is not the score to beat
    assert model.weight.item() == 30                           # and its weights are not restored
    assert len(history["train_loss"]) == 30 + 1 + 25
    assert len(history["val_roc_auc"]) == len(history["train_loss"])   # warm-up epochs are still logged


def test_warmup_bounds_the_earliest_stop_and_best_epoch(monkeypatch):
    # A flat val curve: the first selectable epoch is best, so the run stops at start_from_epoch + patience.
    model, history = _scripted_training(monkeypatch, [0.8] * 40, epochs=40, patience=5, start_from_epoch=10)
    assert history["best_epoch"] == 10 and model.weight.item() == 10
    assert len(history["train_loss"]) == 10 + 1 + 5


def test_warmup_leaves_a_run_whose_best_epoch_is_after_it_unchanged(monkeypatch):
    # Why only the ablation seeds with best_epoch < 50 needed re-running: for every other seed the
    # trajectory, best epoch, stop epoch and restored weights are the same with or without warm-up.
    curve = [0.5, 0.6, 0.7] + [0.65] * 12 + [0.9] + [0.85] * 44
    runs = [_scripted_training(monkeypatch, curve, epochs=60, patience=25, start_from_epoch=s) for s in (0, 10)]
    (m0, h0), (m10, h10) = runs
    assert h0["best_epoch"] == h10["best_epoch"] == 15
    assert len(h0["train_loss"]) == len(h10["train_loss"]) == 15 + 1 + 25
    assert h0["val_roc_auc"] == h10["val_roc_auc"] and m0.weight.item() == m10.weight.item() == 15


def test_warmup_must_leave_epochs_to_select_from(monkeypatch):
    with pytest.raises(ValueError):
        _scripted_training(monkeypatch, [0.8] * 10, epochs=10, patience=5, start_from_epoch=10)


def test_run_one_config_reports_the_val_score_of_the_restored_checkpoint(monkeypatch, tmp_path):
    def fake_train(model, start_from_epoch, **kwargs):
        return model, {"val_roc_auc": [0.5, 0.9, 0.6, 0.7, np.nan, 0.65], "best_epoch": 3, "grad_norm": [1.0]}
    monkeypatch.setattr(gnn_tools, "train_model", fake_train)
    with contextlib.redirect_stdout(io.StringIO()):
        warm = gnn_tools.run_one_config(lambda h, b: torch.nn.Linear(1, 1), None, None, None, [8], None, 1e-3, 0.0,
                                        epochs=6, patience=2, save_path=str(tmp_path / "m.pt"), start_from_epoch=2)
        cold = gnn_tools.run_one_config(lambda h, b: torch.nn.Linear(1, 1), None, None, None, [8], None, 1e-3, 0.0,
                                        epochs=6, patience=2, save_path=str(tmp_path / "m.pt"))
    assert warm["val_roc_auc"] == 0.7                          # not the warm-up spike 0.9
    assert cold["val_roc_auc"] == 0.9


# ── K-sweep ───────────────────────────────────────────────────────────────

def _tuning_csv(final_seeds=(0, 1), val_seeds='0.8;0.8', layers=3):
    """A finished tuning run whose winner is 64-d x `layers`, 10 bases, lr 5e-3, wd 1e-5 (best epoch 200)."""
    winner = dict(hidden_dim=64.0, layers=layers, num_bases=10.0, lr=5e-3, weight_decay=1e-5)
    rows = [
        dict(phase='3_Verification', status='ok', **dict(winner, hidden_dim=32.0), val_roc_auc=0.79,
             val_roc_auc_seeds='0.79;0.79', best_epoch=150, best_epoch_seeds='150;160'),
        dict(phase='3_Verification_Cached', status='ok', **winner, val_roc_auc=0.8, val_roc_auc_std=0.0,
             val_roc_auc_seeds=val_seeds, best_epoch=200, best_epoch_seeds='200;360', max_grad_norm=1.0,
             runtime_sec=0.0),
    ]
    rows += [dict(phase='4_Final_Test_Evaluation', status='ok', seed=float(s), **winner, val_roc_auc=0.8,
                  best_epoch=200, num_parameters=7, **{c: 0.9 + s / 100 for c in gnn_tools.FINAL_COLUMNS
                                                       if c.startswith('test_')})
             for s in final_seeds]
    return pd.DataFrame(rows)


def _k_sweep_stub(monkeypatch, fail_at=()):
    """run_config_over_seeds / evaluate_final_test_row stubbed: the tune row's val ROC-AUC is 0.7 + K/100 and
    its best epoch 10 * K (an OOM for K in fail_at); a final row's test ROC-AUC is its val + seed/1000.
    Returns the logged calls."""
    calls = []
    def fake_tune(model_factory, graph, train, val, hidden_list, bases, lr, wd, epochs, patience, save_path, seeds,
                  start_from_epoch):
        k = len(hidden_list) + 1
        calls.append(('tune', k, list(hidden_list), bases, lr, wd, start_from_epoch))
        ok = k not in fail_at
        return {'num_bases': bases, 'lr': lr, 'weight_decay': wd, 'val_roc_auc': 0.7 + k / 100 if ok else None,
                'val_roc_auc_std': 0.0 if ok else None, 'val_roc_auc_seeds': '0.7;0.7',
                'best_epoch': 10 * k if ok else None, 'best_epoch_seeds': f'{10 * k};{10 * k}' if ok else '',
                'max_grad_norm': 1.0, 'runtime_sec': 1.0, 'status': 'ok' if ok else 'oom'}
    def fake_final(model_factory, graph_data, train_indices, val_indices, test_indices, schema_label, save_dir,
                   hidden_dim, depth, bases, lr, weight_decay, best_epoch, val_roc_auc, seed):
        calls.append(('final', depth, hidden_dim, best_epoch, seed))
        return {'seed': seed, 'val_roc_auc': val_roc_auc, 'best_epoch': best_epoch, 'num_parameters': depth,
                'status': 'ok', **{c: val_roc_auc + seed / 1000 for c in gnn_tools.FINAL_COLUMNS
                                   if c.startswith('test_')}}
    monkeypatch.setattr(gnn_tools, "run_config_over_seeds", fake_tune)
    monkeypatch.setattr(gnn_tools, "evaluate_final_test_row", fake_final)
    return calls


def _run_k_sweep(tmp_path, tuning, k_values=(1, 2, 3, 4), final_seeds=(0, 1), kg='pure', start_from_epoch=50):
    with contextlib.redirect_stdout(io.StringIO()):
        return gnn_tools.run_k_sweep(
            lambda hidden_list, bases: torch.nn.Linear(1, 1), _ablation_kg(), kg, tuning, list(k_values),
            epochs=400, patience=50, start_from_epoch=start_from_epoch, tuning_seeds=(0, 1), final_seeds=final_seeds,
            save_dir=f"{tmp_path}/", csv_path=tmp_path / "k_sweep.csv")


def test_k_sweep_varies_only_the_depth_and_reuses_the_tuned_depth(monkeypatch, tmp_path):
    calls = _k_sweep_stub(monkeypatch)
    df = _run_k_sweep(tmp_path, _tuning_csv())
    tunes = [c for c in calls if c[0] == 'tune']
    assert tunes == [('tune', 1, [], 10, 5e-3, 1e-5, 50), ('tune', 2, [64], 10, 5e-3, 1e-5, 50),
                     ('tune', 4, [64, 64, 64], 10, 5e-3, 1e-5, 50)]   # K=3, the tuned depth, is not trained
    finals = [c for c in calls if c[0] == 'final']
    assert finals == [('final', k, None if k == 1 else 64, 10 * k, s) for k in (1, 2, 4) for s in (0, 1)]

    tune = df[df['phase'] == 'tune'].set_index('K')
    assert tune['reused'].tolist() == [False, False, True, False]
    assert pd.isna(tune.loc[1, 'hidden_dim']) and tune.loc[4, 'hidden_dim'] == 64
    assert tune.loc[3, 'val_roc_auc'] == 0.8 and tune.loc[3, 'best_epoch'] == 200   # the phase-3 row
    final = df[df['phase'] == 'final']
    assert len(final) == 4 * 2 and set(final['seed']) == {0, 1}
    assert final[final['K'] == 3]['test_roc_auc'].tolist() == pytest.approx([0.90, 0.91])   # the phase-4 rows
    assert final[final['K'] == 4]['test_roc_auc'].tolist() == pytest.approx([0.74, 0.741])


def test_k_sweep_flags_seeds_cut_off_by_the_epoch_cap(monkeypatch, tmp_path):
    _k_sweep_stub(monkeypatch)
    df = _run_k_sweep(tmp_path, _tuning_csv())
    tune = df[df['phase'] == 'tune'].set_index('K')
    assert tune['ended_by_epoch_cap'].tolist() == [False, False, True, False]    # the winner's seed peaked at 360


def test_k_sweep_failed_depth_logs_no_final_rows_and_the_rest_still_run(monkeypatch, tmp_path):
    _k_sweep_stub(monkeypatch, fail_at=(4,))
    df = _run_k_sweep(tmp_path, _tuning_csv())
    assert df[df['K'] == 4]['phase'].tolist() == ['tune']
    assert df[df['K'] == 4]['status'].tolist() == ['oom']
    assert not df[df['K'] == 4]['ended_by_epoch_cap'].any()
    assert sorted(df.loc[df['phase'] == 'final', 'K'].unique()) == [1, 2, 3]


def test_k_sweep_resume_skips_finished_depths_and_reruns_failed_ones(monkeypatch, tmp_path):
    _k_sweep_stub(monkeypatch, fail_at=(4,))
    _run_k_sweep(tmp_path, _tuning_csv())
    _run_k_sweep(tmp_path, _tuning_csv(), kg='adhoc', k_values=(1,))   # another KG shares the CSV
    calls = _k_sweep_stub(monkeypatch)
    df = _run_k_sweep(tmp_path, _tuning_csv())
    assert {c[1] for c in calls} == {4}
    pure = df[df['kg'] == 'pure']
    assert len(pure) == 4 * (1 + 2) and (pure['status'] == 'ok').all()
    assert len(df[df['kg'] == 'adhoc']) == 1 + 2
    assert not pd.read_csv(tmp_path / "k_sweep.csv").duplicated(['kg', 'K', 'phase', 'seed']).any()


@pytest.mark.parametrize("tuning, error", [
    (_tuning_csv(final_seeds=(0,)), "phase-4 seeds"),                 # finished with other final seeds
    (_tuning_csv(val_seeds='0.8'), "tuned with seeds"),               # tuned with another seed count
    (_tuning_csv().query("phase != '3_Verification_Cached'"), "no phase-3 row"),
    (_tuning_csv().query("not phase.str.startswith('3')", engine="python"), "no phase-3 row"),   # none at all
    (pd.concat([_tuning_csv(), _tuning_csv(layers=2).tail(1)]), "disagree"),   # a phase-4 row at another depth
])
def test_k_sweep_refuses_a_tuning_run_of_another_protocol(monkeypatch, tmp_path, tuning, error):
    _k_sweep_stub(monkeypatch)
    with pytest.raises(ValueError, match=error):
        _run_k_sweep(tmp_path, tuning)


def test_k_sweep_refuses_to_reuse_a_winner_that_the_warmup_would_change(monkeypatch, tmp_path):
    # The winner's tuning seeds peaked at epochs 200 and 360: reusable under a warm-up of up to 200 epochs.
    _k_sweep_stub(monkeypatch)
    assert len(_run_k_sweep(tmp_path, _tuning_csv(), k_values=(3,), start_from_epoch=200)) == 1 + 2
    with pytest.raises(ValueError, match="warm-up"):
        _run_k_sweep(tmp_path, _tuning_csv(), k_values=(3,), start_from_epoch=201)


def test_run_config_over_seeds_passes_the_warmup_to_every_seed(monkeypatch):
    seen = []
    def fake(*args, start_from_epoch, **kwargs):
        seen.append(start_from_epoch)
        return {'val_roc_auc': 0.8, 'best_epoch': 60, 'max_grad_norm': 1.0, 'runtime_sec': 1.0, 'status': 'ok'}
    monkeypatch.setattr(gnn_tools, "run_one_config", fake)
    gnn_tools.run_config_over_seeds(None, None, None, None, [8], None, 1e-3, 0.0, epochs=1, patience=1,
                                    save_path="dir/x.pt", seeds=(0, 1, 2), start_from_epoch=50)
    assert seen == [50, 50, 50]


def test_search_trains_every_tuning_run_with_the_warmup(monkeypatch, tmp_path):
    warmups = []
    df, trained, _ = _search(monkeypatch, tmp_path, _depth3_wins, start_from_epoch=50, warmups=warmups)
    assert len(warmups) == len(trained) > 0 and set(warmups) == {50}


def test_k_sweep_refuses_to_resume_rows_of_another_configuration(monkeypatch, tmp_path):
    _k_sweep_stub(monkeypatch)
    _run_k_sweep(tmp_path, _tuning_csv(), k_values=(1, 2))
    retuned = _tuning_csv().assign(lr=1e-2)                  # the same KG after a re-tuning picked another lr
    with pytest.raises(ValueError, match="another configuration"):
        _run_k_sweep(tmp_path, retuned, k_values=(1, 2))
    rows = _run_k_sweep(tmp_path, retuned, k_values=(1,), kg='other')   # another KG in the same CSV is free
    assert len(rows[rows['kg'] == 'other']) == 1 + 2 and len(rows[rows['kg'] == 'pure']) == 2 * (1 + 2)


def test_ablation_refuses_to_resume_rows_of_another_configuration(monkeypatch, tmp_path):
    _ablation_stub(monkeypatch, {})
    _run_ablation(tmp_path)
    with pytest.raises(ValueError, match="another configuration"):
        with contextlib.redirect_stdout(io.StringIO()):
            gnn_tools.run_ablation_variant(
                lambda hidden_list, bases: torch.nn.Linear(1, 1), _ablation_kg(), 'v', [8], None, 1e-2, 0.0,
                epochs=1, patience=1, start_from_epoch=0, seeds=(0, 1), save_dir=f"{tmp_path}/",
                csv_path=tmp_path / "ablation.csv")


def test_k_sweep_summary():
    df = pd.DataFrame([
        dict(kg='a', K=1, phase='tune', val_roc_auc=0.7, val_roc_auc_std=0.01, best_epoch=10, reused=False,
             ended_by_epoch_cap=False),
        *[dict(kg='a', K=1, phase='final', seed=s, num_parameters=5, **{m: v for m in gnn_tools.TEST_METRICS})
          for s, v in [(0, 0.80), (1, 0.82)]],
        dict(kg='a', K=2, phase='tune', val_roc_auc=0.8, val_roc_auc_std=0.0, best_epoch=20, reused=True,
             ended_by_epoch_cap=True),
        *[dict(kg='a', K=2, phase='final', seed=s, num_parameters=9, **{m: 0.9 for m in gnn_tools.TEST_METRICS})
          for s in (0, 1)],
    ])
    summary = gnn_tools.summarize_k_sweep(df)
    assert list(summary.index) == [('a', 1), ('a', 2)]
    assert summary['test_roc_auc_mean'].tolist() == pytest.approx([0.81, 0.9])
    assert summary['test_roc_auc_std'].tolist() == pytest.approx([0.01414, 0.0], abs=1e-5)
    assert summary['n_seeds'].tolist() == [2, 2] and summary['num_parameters'].tolist() == [5, 9]
    assert summary['val_roc_auc'].tolist() == [0.7, 0.8] and summary['reused'].tolist() == [False, True]
