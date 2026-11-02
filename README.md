<p align="center">

  <h3 align="center">Ionizable lipid design for RNA delivery with a generative model</h3>

  <p align="center">
    Supporting code for the paper
  </p>
</p>

## Table of contents
* [About this repository](#about-this-repository)
* [Quick start](#quick-start)
* [License](#license)
* [Contact](#contact)

<!-- ABOUT THE PROJECT -->
## About this repository

This codebase contains the inference code and example data associated with the paper titled "Ionizable lipid design for RNA delivery with a generative model". LambdaDesign is a conditional variational framework for ionizable lipid design in which activity estimation and molecular generation share a single latent representation, so that every candidate is scored and optimized before it is decoded into a molecule.

## Quick start

Here, we briefly outline (a) how to install the dependencies for this repository, and (b) how to score a set of ionizable lipids and generate new candidates with the working examples given in this repository.

**Note**: The instructions below are based on a Linux operating system with CUDA 12.8. The code also runs on CPU, including macOS arm64.

```bash
git clone https://github.com/WangJiuming/LambdaDesign.git
cd LambdaDesign
```
Then create the conda environment from the `.yml` configuration file.
```bash
conda env create -f environment.yml
```
This should install all the dependencies necessary for this repository. To activate the environment, simply run the following.
```bash
conda activate lambdadesign
```
The environment file pins the CPU-validated runtime. **If you are running on an NVIDIA GPU, replace PyTorch with a CUDA build that matches your hardware.** Recent data-centre and consumer cards (Hopper `sm_90`, Blackwell `sm_100`/`sm_120`, including the RTX 50 series) are not supported by the pinned `torch==2.6.0` and require CUDA 12.8 wheels.
```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
```
You can confirm that your GPU architecture is supported before going further.
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())"
```
Then install this package itself.
```bash
pip install --no-deps .
lambdadesign --help
```

The pre-trained model checkpoints used in this repository can be downloaded from the huggingface repository via the following. The estimated download time is one minute.
```bash
wget https://huggingface.co/cuhkaih/lambdadesign/resolve/main/lambdadesign_hela_mrna.pt -O checkpoints/lambdadesign_hela_mrna.pt
wget https://huggingface.co/cuhkaih/lambdadesign/resolve/main/molclr_lipid_encoder.pth -O checkpoints/molclr_lipid_encoder.pth
```
In case the above link is unavailable, the checkpoints can also be downloaded manually using the <a href="https://drive.google.com/drive/folders/1A0c5mYZozb5J7Jt6pDsn4HhJqzaVw5la">alternate link</a>.

Two files are required. Scoring uses both; generation uses the design checkpoint only.

Now LambdaDesign is ready. Users may first inspect the checkpoint to confirm that both weight files load and that the intended device is selected. Note that the model is configured to use GPU device `cuda:0` by setting the argument `--device cuda:0`.
```bash
lambdadesign inspect \
  --checkpoint checkpoints/lambdadesign_hela_mrna.pt \
  --molclr-checkpoint checkpoints/molclr_lipid_encoder.pth \
  --device cuda:0
```
This prints the architecture, the saved activity statistics, the checkpoint hashes and the resolved device, ending with `"scoring_ready": true`.

To score a set of ionizable lipid structures, execute the following. The input is a CSV with one SMILES per row.
```bash
lambdadesign score \
  --checkpoint checkpoints/lambdadesign_hela_mrna.pt \
  --molclr-checkpoint checkpoints/molclr_lipid_encoder.pth \
  --input examples/molecules.csv \
  --smiles-column smiles \
  --output results/activity_scores.csv \
  --device cuda:0
```
Then, users may check the outputs located at the directory specified by the `--output` argument, which should contain the following files.
```bash
- results/
     |- activity_scores.csv   # one row per input molecule, with scores and per-row status
     |- activity_scores.json  # run summary: counts, checkpoint hashes, settings, versions
```
The scores will be written alongside the original columns like below.
```
molecule_id,canonical_smiles,activity_score,predicted_activity
example_001,CC(C)CCCCCCCOC(=O)CCCCCN(CCCCCC(=O)OCCCCCCCC(C)C)CCN1CCN(C)CC1,0.3624,0.2143
example_002,CC(C)=CCCC(C)CCOC(=O)CCCCCCCN(CCCCCCCC(=O)OCCC(C)CCC=C(C)C)CCN1CCN(C)CC1,0.4674,1.0363
```
`activity_score` is the normalized model output on the training scale, and `predicted_activity` is the same value converted back to the original HeLa/mRNA activity scale. Neither is a probability, a transfection fold change or a percentage, and neither is clipped.

To generate new candidate structures, execute the following. Generation requires the design checkpoint only.
```bash
lambdadesign generate \
  --checkpoint checkpoints/lambdadesign_hela_mrna.pt \
  --config configs/generation.json \
  --num-molecules 5 \
  --max-latents 512 \
  --output results/generated_molecules.csv \
  --device cuda:0
```
Progress is reported to the screen while sampling proceeds.
```
sampled=32 decoded=0 unique_eligible=0
sampled=64 decoded=5 unique_eligible=1
sampled=96 decoded=13 unique_eligible=5
```
The generated molecules are written with their SELFIES string, the activity score of the latent vector that produced them, and the aliphatic carbon component sizes used by the chemical filters.

To score the structures that were just generated, run `score` on the generated CSV.
```bash
lambdadesign score \
  --checkpoint checkpoints/lambdadesign_hela_mrna.pt \
  --molclr-checkpoint checkpoints/molclr_lipid_encoder.pth \
  --input results/generated_molecules.csv \
  --smiles-column smiles \
  --output results/generated_rescored.csv \
  --device cuda:0
```
Note that `latent_activity_score` in the generation output and `activity_score` in the scoring output are **distinct quantities**. The former belongs to the sampled latent vector before decoding; the latter is obtained by re-encoding the decoded structure.

To check the installation, users can run the test suite.
```bash
python -m pytest
```

## License

The codes for this repository are licensed under the MIT License (see `LICENSE`).

## Contact

For questions about this repository or the paper, please contact jmwang@link.cuhk.edu.hk, or open an issue in this repository.
