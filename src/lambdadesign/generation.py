"""Latent activity filtering followed by autoregressive SELFIES generation."""

from dataclasses import asdict, dataclass
import math

import selfies
import torch

from .chemistry import (MoleculeError, aliphatic_component_sizes, has_tertiary_amine,
                        parse_smiles, passes_tail_filter)


@dataclass(frozen=True)
class GenerationOptions:
    num_molecules: int = 10
    max_latents: int = 10000
    latent_batch_size: int = 128
    decode_batch_size: int = 16
    samples_per_latent: int = 5
    max_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 10
    activity_threshold: str | float | None = "p75"
    filter_tails: bool = True
    tail_min_carbons: int = 10
    tail_max_carbons: int = 18
    required_tail_components: int = 2
    require_tertiary_amine: bool = True
    seed: int = 42

    def validate(self, model):
        integer_keys = ("num_molecules", "max_latents", "latent_batch_size", "decode_batch_size",
                        "samples_per_latent", "max_tokens", "tail_min_carbons",
                        "tail_max_carbons", "required_tail_components")
        for key in integer_keys:
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer between 0 and 2**63-1")
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError("top_k must be a nonnegative integer; zero disables top-k filtering")
        if not isinstance(self.temperature, (float, int)) or not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and greater than zero")
        if not 2 <= self.max_tokens <= model.max_seq_len:
            raise ValueError(f"max_tokens must be between 2 and {model.max_seq_len}, including BOS")
        if self.tail_min_carbons > self.tail_max_carbons:
            raise ValueError("tail_min_carbons exceeds tail_max_carbons")
        if type(self.filter_tails) is not bool or type(self.require_tertiary_amine) is not bool:
            raise ValueError("Chemical filter switches must be booleans")


@dataclass
class GenerationResult:
    molecules: list[dict]
    summary: dict


def resolve_activity_threshold(value, statistics):
    if value is None or (isinstance(value, str) and value.lower() in ("none", "off")):
        return None
    if isinstance(value, str) and value.lower() in ("p75", "p90", "p95", "mean"):
        key = value.lower()
        if key not in statistics:
            raise ValueError(f"Checkpoint does not contain activity statistic {key}; supply a numeric threshold")
        return statistics[key]
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unknown activity threshold: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError("Activity threshold must be finite")
    return result


@torch.inference_mode()
def sample_tokens(model, vocabulary, latent, options, rng):
    tokens = torch.full((len(latent), 1), vocabulary.bos_id, dtype=torch.long, device=latent.device)
    finished = torch.zeros(len(latent), dtype=torch.bool, device=latent.device)
    for _ in range(options.max_tokens - 1):
        logits = model.decode_logits(latent, tokens)[:, -1] / options.temperature
        if options.top_k > 0:
            cutoff = logits.topk(min(options.top_k, logits.size(-1)), dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < cutoff, -torch.inf)
        probabilities = logits.softmax(dim=-1)
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("Decoder produced non-finite sampling probabilities")
        next_token = torch.multinomial(probabilities, 1, generator=rng).squeeze(-1)
        next_token = next_token.masked_fill(finished, vocabulary.pad_id)
        tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
        finished |= next_token.eq(vocabulary.eos_id)
        if finished.all():
            break
    return tokens.cpu().tolist(), finished.cpu().tolist()


@torch.inference_mode()
def generate_molecules(checkpoint, options, device, progress=None):
    from rdkit import Chem

    model, vocabulary = checkpoint.model, checkpoint.vocabulary
    options.validate(model)
    threshold = resolve_activity_threshold(options.activity_threshold, checkpoint.activity_statistics)
    rng = torch.Generator(device=device).manual_seed(options.seed)
    counts = {key:0 for key in ("sampled_latents", "activity_passing_latents", "decoded_attempts",
                                "eos_terminated_attempts", "rdkit_valid_attempts", "tail_passing_attempts",
                                "amine_passing_attempts", "duplicate_eligible_attempts")}
    accepted, seen = [], set()
    while len(accepted) < options.num_molecules and counts["sampled_latents"] < options.max_latents:
        size = min(options.latent_batch_size, options.max_latents - counts["sampled_latents"])
        first_index = counts["sampled_latents"]
        latent = torch.randn(size, model.latent_dim, device=device, generator=rng)
        scores = model.activity_predictor(latent)
        if not torch.isfinite(scores).all():
            raise RuntimeError("Activity head produced non-finite scores")
        passed = torch.ones(size, dtype=torch.bool, device=device) if threshold is None else scores >= threshold
        selected = torch.nonzero(passed).flatten()
        counts["sampled_latents"] += size
        counts["activity_passing_latents"] += len(selected)
        for draw in range(options.samples_per_latent):
            for start in range(0, len(selected), options.decode_batch_size):
                subset = selected[start:start+options.decode_batch_size]
                generated, terminated = sample_tokens(model, vocabulary, latent[subset], options, rng)
                counts["decoded_attempts"] += len(generated)
                counts["eos_terminated_attempts"] += sum(terminated)
                ids, values = subset.cpu().tolist(), scores[subset].cpu().tolist()
                for token_ids, ended, index, score in zip(generated, terminated, ids, values):
                    sequence = vocabulary.decode(token_ids)
                    try:
                        smiles = selfies.decoder(sequence)
                        molecule = parse_smiles(smiles)
                    except (selfies.DecoderError, MoleculeError):
                        continue
                    counts["rdkit_valid_attempts"] += 1
                    sizes = aliphatic_component_sizes(molecule)
                    if options.filter_tails and not passes_tail_filter(sizes, options.tail_min_carbons,
                                                                      options.tail_max_carbons, options.required_tail_components):
                        continue
                    counts["tail_passing_attempts"] += 1
                    if options.require_tertiary_amine and not has_tertiary_amine(molecule):
                        continue
                    counts["amine_passing_attempts"] += 1
                    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
                    if canonical in seen:
                        counts["duplicate_eligible_attempts"] += 1
                        continue
                    seen.add(canonical)
                    accepted.append({"smiles":canonical, "selfies":sequence,
                                     "latent_activity_score":float(score),
                                     "latent_index":first_index+index, "draw_index":draw,
                                     "eos_terminated":ended,
                                     "aliphatic_carbon_counts":";".join(map(str, sizes))})
                if len(accepted) >= options.num_molecules:
                    break
            if len(accepted) >= options.num_molecules:
                break
        if progress:
            progress(dict(counts, unique_eligible_molecules=len(accepted)))
    # Preserve first-seen deduplication and selection, then rank by latent score.
    selected_molecules = sorted(accepted[:options.num_molecules],
                                key=lambda r:r["latent_activity_score"], reverse=True)
    rows = [dict(rank=i+1, **row) for i,row in enumerate(selected_molecules)]
    summary = dict(counts, unique_eligible_molecules=len(accepted), retained_molecules=len(rows),
                   complete=len(rows)==options.num_molecules,
                   activity_threshold_value=threshold, checkpoint_sha256=checkpoint.sha256,
                   score_definition="activity_predictor(z), z sampled from N(0,I)",
                   device=str(device), options=asdict(options))
    return GenerationResult(rows, summary)
