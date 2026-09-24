import torch
import dgl
import numpy as np

SUPPORTED_ELEMENTS = [
    "H",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
]


def fp_rdkit(atom):
    from rdkit import Chem

    element = atom.GetSymbol()
    if element not in SUPPORTED_ELEMENTS:
        raise ValueError(f"Element {element} is not supported.")

    HYBRIDIZATION_RDKIT = {
        Chem.rdchem.HybridizationType.SP: torch.tensor(
            [1, 0, 0, 0, 0],
            dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.SP2: torch.tensor(
            [0, 1, 0, 0, 0],
            dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.SP3: torch.tensor(
            [0, 0, 1, 0, 0],
            dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.SP3D: torch.tensor(
            [0, 0, 0, 1, 0],
            dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.SP3D2: torch.tensor(
            [0, 0, 0, 0, 1],
            dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.S: torch.tensor(
            [0, 0, 0, 0, 0],
            dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.UNSPECIFIED: torch.tensor(
            [0, 0, 0, 0, 0],
            dtype=torch.get_default_dtype(),
        ),
    }
    return torch.cat(
        [
            torch.tensor(
                [
                    atom.GetTotalDegree(),
                    atom.GetTotalValence(),
                    atom.GetExplicitValence(),
                    # atom.GetFormalCharge(),
                    atom.GetIsAromatic() * 1.0,
                    atom.GetMass(),
                    atom.IsInRingSize(3) * 1.0,
                    atom.IsInRingSize(4) * 1.0,
                    atom.IsInRingSize(5) * 1.0,
                    atom.IsInRingSize(6) * 1.0,
                    atom.IsInRingSize(7) * 1.0,
                    atom.IsInRingSize(8) * 1.0,
                ],
                dtype=torch.get_default_dtype(),
            ),
            HYBRIDIZATION_RDKIT[atom.GetHybridization()],
        ],
        dim=0,
    )


# def from_rdkit_mol(mol, use_fp=False):
#     """Convert an RDKit molecule to a DGLGraph.
    
#     Parameters
#     ----------
#     mol : rdkit.Chem.Mol
#         Input molecule.
#     use_fp : bool, optional, default=False
#         Whether to use fingerprint features.
    
#     Returns
#     -------
#     dgl.DGLGraph
#         DGL graph with node features.
#     """
    
#     # Получаем количество атомов
#     n_atoms = mol.GetNumAtoms()
    
#     # Создаем граф с помощью dgl.graph (рекомендуемый способ)
#     # Сначала создаем список ребер
#     src = []
#     dst = []
    
#     for bond in mol.GetBonds():
#         u = bond.GetBeginAtomIdx()
#         v = bond.GetEndAtomIdx()
#         src.append(u)
#         dst.append(v)
#         src.append(v)
#         dst.append(u)
    
#     if src:  # если есть связи
#         # Создаем граф из ребер
#         g = dgl.graph((src, dst), num_nodes=n_atoms)
#     else:
#         # Если нет связей, создаем граф только с узлами
#         g = dgl.graph(([], []), num_nodes=n_atoms)
    
#     # Добавляем признаки узлов
#     atomic_numbers = [float(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
#     g.ndata["type"] = torch.tensor(atomic_numbers, dtype=torch.float32).view(-1, 1)
    
#     formal_charges = [float(atom.GetFormalCharge()) for atom in mol.GetAtoms()]
#     g.ndata["q_ref"] = torch.tensor(formal_charges, dtype=torch.float32).view(-1, 1)
    
#     # Добавляем признаки ребер
#     if src:  # если есть связи
#         bond_types = []
#         for bond in mol.GetBonds():
#             bond_type = float(bond.GetBondTypeAsDouble())
#             bond_types.extend([bond_type, bond_type])
#         g.edata["type"] = torch.tensor(bond_types, dtype=torch.float32).view(-1, 1)
    
#     return g


def from_rdkit_mol(mol, use_fp=True):
    import dgl
    from rdkit import Chem

    # initialize graph
    g = dgl.DGLGraph()

    # enter nodes
    n_atoms = int(mol.GetNumAtoms())
    g.add_nodes(n_atoms)
    g.ndata["type"] = torch.tensor(
    [[float(atom.GetAtomicNum())] for atom in mol.GetAtoms()], 
    dtype=torch.float32
    )
    g.ndata["q_ref"] = torch.tensor(
    [[float(atom.GetFormalCharge())] for atom in mol.GetAtoms()],
    dtype=torch.float32
    )
    # g.ndata["type"] = torch.Tensor(
    #     [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    # )
    # g.ndata["q_ref"] = torch.Tensor(
    #     [[atom.GetFormalCharge()] for atom in mol.GetAtoms()]
    # )
    h_v = torch.zeros(g.ndata["type"].shape[0], 100, dtype=torch.float32)

    h_v[
        torch.arange(g.ndata["type"].shape[0]),
        torch.squeeze(g.ndata["type"]).long(),
    ] = 1.0

    h_v_fp = torch.stack([fp_rdkit(atom) for atom in mol.GetAtoms()], axis=0)

    if use_fp == True:
        h_v = torch.cat([h_v, h_v_fp], dim=-1)  # (n_atoms, 117)

    g.ndata["h0"] = h_v

    # enter bonds
    bonds = list(mol.GetBonds())
    bonds_begin_idxs = [bond.GetBeginAtomIdx() for bond in bonds]
    bonds_end_idxs = [bond.GetEndAtomIdx() for bond in bonds]
    bonds_types = [bond.GetBondType().real for bond in bonds]

    # NOTE: dgl edges are directional
    g.add_edges(bonds_begin_idxs, bonds_end_idxs)
    g.add_edges(bonds_end_idxs, bonds_begin_idxs)

    # g.edata["type"] = torch.Tensor(bonds_types)[:, None].repeat(2, 1)

    return g
