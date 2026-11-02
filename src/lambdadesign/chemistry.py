"""RDKit featurization and the original generation eligibility filters."""

from dataclasses import dataclass

import torch
from rdkit import Chem, rdBase

CHIRALITY = (Chem.ChiralType.CHI_UNSPECIFIED, Chem.ChiralType.CHI_TETRAHEDRAL_CW,
             Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
BOND_TYPES = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
              Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)
BOND_DIRECTIONS = (Chem.BondDir.NONE, Chem.BondDir.ENDUPRIGHT, Chem.BondDir.ENDDOWNRIGHT)
TERTIARY_AMINE = Chem.MolFromSmarts("[NX3;H0;!$(N-C=[O,S,N]);!$(N=*);!$(N#*);!$(Nc)]")


class MoleculeError(ValueError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


@dataclass
class MolecularGraph:
    atoms: torch.Tensor
    edges: torch.Tensor
    bonds: torch.Tensor
    canonical_smiles: str


def parse_smiles(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        raise MoleculeError("invalid_smiles", "Empty SMILES")
    # RDKit diagnostics are captured as row-level status, not printed hundreds of times.
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise MoleculeError("invalid_smiles", "RDKit could not parse a nonempty molecule")
    return molecule


def smiles_to_graph(smiles, max_atoms, canonicalize=False):
    molecule = parse_smiles(smiles)
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    if canonicalize:
        molecule = parse_smiles(canonical)
    if molecule.GetNumAtoms() > max_atoms:
        raise MoleculeError("too_many_atoms", f"{molecule.GetNumAtoms()} atoms exceed the checkpoint limit {max_atoms}")
    atoms = []
    for atom in molecule.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        if not 1 <= atomic_number <= 118:
            raise MoleculeError("unsupported_atom", "Only atomic numbers 1-118 are supported")
        chirality = atom.GetChiralTag()
        if chirality not in CHIRALITY:
            raise MoleculeError("unsupported_chirality", "Chirality is outside the checkpoint's three-token vocabulary")
        atoms.append([atomic_number - 1, CHIRALITY.index(chirality)])
    edges, bonds = [], []
    for bond in molecule.GetBonds():
        if bond.GetBondType() not in BOND_TYPES:
            raise MoleculeError("unsupported_bond", f"Unsupported bond type: {bond.GetBondType()}")
        direction = bond.GetBondDir()
        direction_index = BOND_DIRECTIONS.index(direction) if direction in BOND_DIRECTIONS else 0
        feature = [BOND_TYPES.index(bond.GetBondType()), direction_index]
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([[start, end], [end, start]])
        bonds.extend([feature, feature])
    edge_tensor = torch.tensor(edges, dtype=torch.long).reshape(-1, 2).t().contiguous()
    bond_tensor = torch.tensor(bonds, dtype=torch.long).reshape(-1, 2)
    return MolecularGraph(torch.tensor(atoms, dtype=torch.long), edge_tensor, bond_tensor, canonical)


def batch_graphs(graphs, device):
    atoms, edges, bonds, sizes = [], [], [], []
    offset = 0
    for graph in graphs:
        atoms.append(graph.atoms)
        edges.append(graph.edges + offset)
        bonds.append(graph.bonds)
        sizes.append(len(graph.atoms))
        offset += sizes[-1]
    return (torch.cat(atoms).to(device), torch.cat(edges, dim=1).to(device),
            torch.cat(bonds).to(device), sizes)


def aliphatic_component_sizes(molecule):
    """Carbon counts of connected non-ring, non-aromatic carbon components."""
    remaining = {a.GetIdx() for a in molecule.GetAtoms()
                 if a.GetAtomicNum() == 6 and not a.GetIsAromatic() and not a.IsInRing()}
    sizes = []
    while remaining:
        todo = [min(remaining)]; size = 0
        while todo:
            atom = todo.pop()
            if atom not in remaining:
                continue
            remaining.remove(atom); size += 1
            todo.extend(n.GetIdx() for n in molecule.GetAtomWithIdx(atom).GetNeighbors()
                        if n.GetIdx() in remaining)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def passes_tail_filter(sizes, minimum=10, maximum=18, required=2):
    return len(sizes) >= required and all(minimum <= n <= maximum for n in sizes[:required])


def has_tertiary_amine(molecule):
    return molecule.HasSubstructMatch(TERTIARY_AMINE)
