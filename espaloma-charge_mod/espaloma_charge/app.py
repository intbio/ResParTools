import os
import tempfile
from urllib import request
from rdkit import Chem
import dgl
try:
    dgl.use_libxsmm(False)
except:
    pass
import torch
from torch.utils.model_zoo import load_url
import numpy as np
from .utils import from_rdkit_mol
from .models import ChargeEquilibrium_mod
from typing import Sequence, Dict


# TODO: Do we really want to define this at file level, 
# rather than within some kind of class?
MODEL_URL = " https://github.com/choderalab/espaloma_charge/releases/download/v0.0.8/model.pt"
MODEL_PATH = ".espaloma_charge_model.pt"


def charge(
        molecule,
        total_charge: float = None,
        model_url: str = None,
        constraints: Dict[int, float] = None
    ) -> np.ndarray:
    """Assign machine-learned AM1-BCC partial charges to a molecule.

    Parameters
    ----------
    molecule : rdkit.Chem.Mol
        Input molecule.

    total_charge : float = 0.0


    model_url : str, optional, default=None
        URL or filepath to retrieve the model from.
        If None, the default MODEL_URL (defined at file level) is used

    Returns
    -------
    np.ndarray : (n_atoms, ) array of partial charges.

    """
    if isinstance(molecule, Sequence):
        return charge_multiple(molecule)


    if model_url is None:
        model_url = MODEL_URL

    if not os.path.exists(MODEL_PATH):
        request.urlretrieve(model_url, MODEL_PATH)

    model = torch.load(MODEL_PATH)
    if constraints is not None:
        model[2] = torch.nn.Identity()
        out_block = ChargeEquilibrium_mod()

    if total_charge is None:
        total_charge = Chem.GetFormalCharge(molecule)
    graph = from_rdkit_mol(molecule)

    if torch.cuda.is_available():
        graph = graph.to("cuda:0")
        model = model.cuda()
        # print(model)
    
    graph = model(graph)
    if constraints:
        graph = out_block(graph, constraints=constraints)
    else:
        graph = model(graph)
    return graph.ndata["q"].cpu().detach().flatten().numpy()


def charge_multiple(
        molecules,
        model_url: str = None,
        constraints: Sequence[Dict[int, float]] = None
    ) -> np.ndarray:
    """Assign machine-learned AM1-BCC partial charges to a molecule.

    Parameters
    ----------
    molecule : rdkit.Chem.Mol
        Input molecule.

    model_url : str, optional, default=None
        URL or filepath to retrieve the model from.
        If None, the default MODEL_URL (defined at file level) is used

    Returns
    -------
    np.ndarray : (n_atoms, ) array of partial charges.

    """
    if model_url is None:
        model_url = MODEL_URL

    if not os.path.exists(MODEL_PATH):
        request.urlretrieve(model_url, MODEL_PATH)

    model = torch.load(MODEL_PATH)

    graphs = [from_rdkit_mol(molecule) for molecule in molecules]
    graph = dgl.batch(graphs)
    
    if torch.cuda.is_available():
        batched_graph = batched_graph.to("cuda:0")
        model = model.cuda()

    # Process constraints
    if constraints is None:
        constraints = [None] * len(molecules)
    
    # Forward pass with per-molecule constraints
    batched_graph = model(batched_graph, constraints=constraints)
    
    # Unbatch and return
    return [g.ndata["q"].detach().flatten().numpy() for g in dgl.unbatch(batched_graph)]



