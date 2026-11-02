from dataclasses import replace
import csv
import json

import pytest
import torch

from lambdadesign import GenerationOptions, LambdaDesign
from lambdadesign.chemistry import (MoleculeError, aliphatic_component_sizes,
                                   has_tertiary_amine, parse_smiles, passes_tail_filter,
                                   smiles_to_graph)
from lambdadesign.checkpoint import (VERIFIED_ACTIVITY_TRANSFORMS, file_sha256,
                                    load_design_checkpoint, resolve_device)
from lambdadesign.generation import resolve_activity_threshold, sample_tokens
from lambdadesign.vocabulary import SelfiesVocabulary
from lambdadesign.cli import main
from lambdadesign.molclr import MolecularMessageLayer


def test_scoring_preserves_failed_rows_and_is_repeatable(checkpoints):
    model = LambdaDesign.from_checkpoint(*checkpoints, device="cpu")
    inputs = ["CCO", "invalid", "", "*", "C"*21, "CCO", "[He]"]
    result = model.score(inputs, batch_size=2)
    assert [r["status"] for r in result] == ["ok", "invalid_smiles", "invalid_smiles",
                                               "unsupported_atom", "too_many_atoms", "ok", "ok"]
    assert all(r["activity_score"] is None for r in result[1:5])
    assert result[0]["activity_score"] == pytest.approx(result[5]["activity_score"], abs=1e-6)
    again = model.score(inputs, batch_size=1)
    for left,right in zip(result, again):
        assert left["activity_score"] == pytest.approx(right["activity_score"], abs=1e-6)


def test_canonicalization_and_no_bond_graph(checkpoints):
    model = LambdaDesign.from_checkpoint(*checkpoints, device="cpu")
    rows = model.score(["OCC", "CCO"], canonicalize=True)
    assert rows[0]["activity_score"] == rows[1]["activity_score"]
    graph = smiles_to_graph("[He]", 20)
    assert graph.edges.shape == (2,0) and graph.bonds.shape == (0,2)


def test_required_weights_and_strict_state(checkpoints, tmp_path):
    with pytest.raises(FileNotFoundError):
        LambdaDesign.from_checkpoint(tmp_path / "missing.pt")
    model = LambdaDesign.from_checkpoint(checkpoints[0], device="cpu")
    with pytest.raises(ValueError, match="molclr-checkpoint"):
        model.score(["CCO"])
    ck = torch.load(checkpoints[0], weights_only=True)
    del ck["model_state"]["activity_predictor.net.0.weight"]
    bad = tmp_path / "bad.pt"; torch.save(ck,bad)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_design_checkpoint(bad, "cpu")


def test_unsupported_chirality_is_explicit():
    with pytest.raises(MoleculeError, match="Chirality"):
        smiles_to_graph("[Pt@SP1](Cl)(Br)(I)F", 20)


def test_original_chemical_filter_definition():
    eligible = parse_smiles("CCCCCCCCCCCCN(C)CCCCCCCCCCCC")
    sizes = aliphatic_component_sizes(eligible)
    assert sizes == [12,12,1]
    assert passes_tail_filter(sizes) and has_tertiary_amine(eligible)
    assert not passes_tail_filter([19,12])
    assert not passes_tail_filter([12])
    assert not has_tertiary_amine(parse_smiles("CC(=O)N(C)C"))


def test_message_sum_preserves_negative_features():
    layer = MolecularMessageLayer(2)
    # Observe aggregation before the learned update, including both directed
    # bonds and the two added self loops. The original MolCLR message is x+e.
    layer.mlp = torch.nn.Identity()
    with torch.no_grad():
        layer.edge_embedding1.weight.zero_(); layer.edge_embedding2.weight.zero_()
    nodes = torch.tensor([[-3., 1.], [1., -4.]])
    edges = torch.tensor([[0,1],[1,0]])
    bonds = torch.zeros(2,2,dtype=torch.long)
    torch.testing.assert_close(layer(nodes,edges,bonds), torch.tensor([[-2.,-3.],[-2.,-3.]]))


@pytest.mark.parametrize("value", ["p99", "garbage", float("nan"), float("inf")])
def test_unknown_or_nonfinite_threshold_raises(value):
    with pytest.raises(ValueError):resolve_activity_threshold(value, {"p75":0.4})


def test_threshold_is_saved_metadata():
    assert resolve_activity_threshold("p75", {"p75":0.4093}) == 0.4093
    assert resolve_activity_threshold("0.7", {}) == 0.7
    assert resolve_activity_threshold("none", {}) is None
    with pytest.raises(ValueError):resolve_activity_threshold("p90", {})


def test_budget_counts_exactly_without_decoding(checkpoints):
    model = LambdaDesign.from_checkpoint(checkpoints[0], device="cpu")
    options = GenerationOptions(num_molecules=2, max_latents=7, latent_batch_size=4,
                                max_tokens=12, activity_threshold=1e8)
    result = model.generate(options)
    assert result.molecules == []
    assert result.summary["sampled_latents"] == 7
    assert result.summary["decoded_attempts"] == 0
    assert not result.summary["complete"]


def test_per_sequence_eos_and_padding():
    vocabulary = SelfiesVocabulary({s:i for i,s in enumerate(["<pad>","<bos>","<eos>","<unk>","[C]"])})
    class Decoder:
        def decode_logits(self, latent, tokens):
            out = torch.full((2,tokens.size(1),5), -100.0)
            out[0,-1,2] = 100
            out[1,-1,4 if tokens.size(1)==1 else 2] = 100
            return out
    ids, ended = sample_tokens(Decoder(), vocabulary, torch.zeros(2,4),
                              GenerationOptions(max_tokens=8, top_k=1), torch.Generator().manual_seed(1))
    assert ids == [[1,2,0],[1,4,2]]
    assert ended == [True,True]
    assert vocabulary.decode(ids[1]) == "[C]"


@pytest.mark.parametrize("changes", [{"max_tokens":13}, {"temperature":0}, {"latent_batch_size":0},
                                     {"top_k":-1}, {"filter_tails":"false"}, {"seed":-1}])
def test_generation_config_errors(checkpoints, changes):
    model = LambdaDesign.from_checkpoint(checkpoints[0], device="cpu")
    options = replace(GenerationOptions(max_tokens=12), **changes)
    with pytest.raises(ValueError):model.generate(options)


def test_cli_score_summary_overwrite_and_partial_generation(checkpoints, tmp_path):
    design, encoder = map(str, checkpoints)
    source = tmp_path / "molecules.csv"
    source.write_text("id,smiles\na,CCO\nb,bad\n")
    output = tmp_path / "scores.csv"
    args = ["score","--checkpoint",design,"--molclr-checkpoint",encoder,"--input",str(source),
            "--output",str(output),"--device","cpu"]
    assert main(args) == 0
    with output.open() as f: rows=list(csv.DictReader(f))
    assert [r["id"] for r in rows] == ["a","b"]
    assert rows[1]["activity_score"] == ""
    assert json.loads(output.with_suffix('.json').read_text())["status_counts"] == {"ok":1,"invalid_smiles":1}
    assert main(args) == 2
    generated = tmp_path / "generated.csv"
    assert main(["generate","--checkpoint",design,"--output",str(generated),"--device","cpu",
                 "--max-latents","3","--activity-threshold","100000","--max-tokens","12"]) == 3
    assert json.loads(generated.with_suffix('.json').read_text())["sampled_latents"] == 3


def test_invalid_device_and_vocabulary():
    with pytest.raises(ValueError):resolve_device("mps")
    with pytest.raises(ValueError):SelfiesVocabulary({"<pad>":0,"<bos>":0})


def register_fixture_scale(checkpoints, monkeypatch):
    transform = dict(scale=7.826270046, offset=-2.621690952,
                     output_scale="original HeLa/mRNA Experiment_value",
                     source="test fixture")
    monkeypatch.setitem(VERIFIED_ACTIVITY_TRANSFORMS, file_sha256(checkpoints[0]), transform)
    return transform


def test_activity_scale_requires_verified_checkpoint_identity(checkpoints, monkeypatch, tmp_path):
    unknown = LambdaDesign.from_checkpoint(*checkpoints, device="cpu")
    row = unknown.score(["CCO"])[0]
    assert row["activity_score"] is not None
    assert row["predicted_activity"] is None
    assert unknown.describe()["activity_transform"] is None
    transform = register_fixture_scale(checkpoints, monkeypatch)
    renamed = tmp_path / "renamed_model.pt"
    renamed.write_bytes(checkpoints[0].read_bytes())
    loaded = load_design_checkpoint(renamed, "cpu")
    assert loaded.activity_transform == transform


def test_activity_inverse_endpoints_no_clipping_and_failed_rows(checkpoints, monkeypatch):
    register_fixture_scale(checkpoints, monkeypatch)
    model = LambdaDesign.from_checkpoint(*checkpoints, device="cpu")
    class FixedPredictor(torch.nn.Module):
        def forward(self, mean):
            return mean.new_tensor([-0.25, 0.0, 0.5, 1.0, 1.25])
    model.checkpoint.model.activity_predictor = FixedPredictor()
    rows = model.score(["CCO", "CCN", "CCC", "CC", "CCCC", "bad"])
    assert [r["activity_score"] for r in rows[:5]] == [-0.25, 0.0, 0.5, 1.0, 1.25]
    assert [r["predicted_activity"] for r in rows[:5]] == pytest.approx(
        [-4.5782584635, -2.621690952, 1.291444071, 5.204579094, 7.1611466055])
    assert rows[5]["predicted_activity"] is None
    assert rows[5]["status"] == "invalid_smiles"


def test_cli_original_activity_is_independent_of_reference_labels(checkpoints, monkeypatch, tmp_path):
    transform = register_fixture_scale(checkpoints, monkeypatch)
    source = tmp_path / "input.csv"
    source.write_text("ID,smiles,activity\na,CCO,999\nb,bad,-999\n")
    output = tmp_path / "output.csv"
    args = ["score", "--checkpoint", str(checkpoints[0]), "--molclr-checkpoint", str(checkpoints[1]),
            "--input", str(source), "--output", str(output), "--device", "cpu"]
    assert main(args) == 0
    with output.open() as handle: rows = list(csv.DictReader(handle))
    assert rows[0]["activity"] == "999"
    assert float(rows[0]["predicted_activity"]) == pytest.approx(
        float(rows[0]["activity_score"]) * 7.826270046 - 2.621690952)
    assert rows[1]["activity_score"] == rows[1]["predicted_activity"] == ""
    assert json.loads(output.with_suffix(".json").read_text())["activity_transform"] == transform
    source.write_text("ID,smiles,activity\na,CCO,-123456\nb,bad,123456\n")
    assert main(args + ["--overwrite"]) == 0
    with output.open() as handle: changed = list(csv.DictReader(handle))
    assert rows[0]["predicted_activity"] == changed[0]["predicted_activity"]
    source.write_text("ID,smiles,predicted_activity\na,CCO,0\n")
    assert main(args + ["--overwrite"]) == 2
