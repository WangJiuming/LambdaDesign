# Third-party notices

`src/lambdadesign/molclr.py` adapts the MolCLR GIN node encoder by Yuyang Wang
(MIT license), from upstream commit
`3d3bc1912be27b0c97435fd9134f4d4c73d4c5ab` at
https://github.com/yuyangw/MolCLR.

The upstream license is included at `src/lambdadesign/licenses/MOLCLR.txt` and
is also installed inside the Python package. The adaptation replaces PyG message
aggregation with the equivalent PyTorch indexed sum and omits unused graph-level
pooling/projection. The original atom and bond indexing and node-update
equations are retained. Numerical comparisons are summarized in `README.md`.

PyTorch, RDKit, NumPy and SELFIES are installed as dependencies under their own
licenses. They are not copied into this source archive. The MolCLR MIT notice
applies to that component and does not assign a new license to the rest of the
research code.
