"""Public scoring and generation interface."""

import torch

from .checkpoint import (file_sha256, load_design_checkpoint, load_molclr_checkpoint,
                         resolve_device)
from .chemistry import MoleculeError, batch_graphs, smiles_to_graph
from .generation import GenerationOptions, generate_molecules


class LambdaDesign:
    def __init__(self, checkpoint, device, molclr_encoder=None, molclr_sha256=None):
        self.checkpoint = checkpoint
        self.device = device
        self.molclr_encoder = molclr_encoder
        self.molclr_sha256 = molclr_sha256

    @classmethod
    def from_checkpoint(cls, checkpoint, molclr_checkpoint=None, device="auto"):
        device = resolve_device(device)
        design = load_design_checkpoint(checkpoint, device)
        encoder, digest = None, None
        if molclr_checkpoint is not None:
            encoder = load_molclr_checkpoint(molclr_checkpoint, device, design.model.node_dim)
            digest = file_sha256(molclr_checkpoint)
        return cls(design, device, encoder, digest)

    def describe(self):
        return {"context":"HeLa/mRNA", "architecture":self.checkpoint.architecture,
                "vocabulary_size":len(self.checkpoint.vocabulary),
                "activity_statistics":self.checkpoint.activity_statistics,
                "activity_transform":self.checkpoint.activity_transform,
                "checkpoint_sha256":self.checkpoint.sha256,
                "molclr_sha256":self.molclr_sha256, "device":str(self.device),
                "scoring_ready":self.molclr_encoder is not None}

    @torch.inference_mode()
    def score(self, smiles, batch_size=32, canonicalize=False):
        """Return one result per input, using activity_predictor(posterior_mean).

        activity_score retains the normalized model output. predicted_activity
        uses the verified checkpoint-specific inverse transformation, when
        available. Invalid molecules and unverified scales return None rather
        than a fabricated original-scale prediction.
        """
        if self.molclr_encoder is None:
            raise ValueError("Scoring requires --molclr-checkpoint in addition to the design checkpoint")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(smiles, str):
            raise TypeError("Pass a sequence of SMILES, not a single string")
        rows, valid = [], []
        for i, value in enumerate(smiles):
            row = {"input_smiles":value, "canonical_smiles":"", "activity_score":None,
                   "predicted_activity":None, "status":"ok", "error":""}
            try:
                graph = smiles_to_graph(value, self.checkpoint.model.max_atoms, canonicalize=canonicalize)
                row["canonical_smiles"] = graph.canonical_smiles
                valid.append((i, graph))
            except MoleculeError as error:
                row.update(status=error.status, error=str(error))
            rows.append(row)
        for start in range(0, len(valid), batch_size):
            subset = valid[start:start+batch_size]
            atoms, edges, bonds, sizes = batch_graphs([g for _,g in subset], self.device)
            nodes = self.molclr_encoder(atoms, edges, bonds)
            features = torch.zeros(len(subset), max(sizes), nodes.size(1), device=self.device)
            mask = torch.ones(len(subset), max(sizes), dtype=torch.bool, device=self.device)
            for i, piece in enumerate(nodes.split(sizes)):
                features[i, :len(piece)] = piece
                mask[i, :len(piece)] = False
            mean, _ = self.checkpoint.model.encoder(features, mask)
            predictions = self.checkpoint.model.activity_predictor(mean)
            if not torch.isfinite(predictions).all():
                raise RuntimeError("Model produced non-finite activity predictions")
            for (index,_), prediction in zip(subset, predictions.cpu().tolist()):
                rows[index]["activity_score"] = float(prediction)
                transform = self.checkpoint.activity_transform
                if transform is not None:
                    rows[index]["predicted_activity"] = (
                        float(prediction) * transform["scale"] + transform["offset"])
        return rows

    def generate(self, options=None, progress=None):
        return generate_molecules(self.checkpoint, options or GenerationOptions(), self.device, progress)
