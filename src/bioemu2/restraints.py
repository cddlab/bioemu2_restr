"""Thin BioEmu2 data extraction and sampling hooks for rgi_toolkit."""

import re
from pathlib import Path

import numpy as np
from rdkit import Chem

from bioemu2.data_interface.const import get_bondtype_from_pdbx_bondtype
from bioemu2.data_interface.structure_hierarchy import FRIENDLYNAME_PER_MOLTYPE, Ligand
from bioemu2.denoising import cp_sync
from bioemu2.structure.templates import _assign_mol_atom_names, _load_ccd_cifwrapper


def resolve_system_restraints(data: dict, path: str | Path) -> dict:
    """Resolve config-file references relative to the containing system YAML."""
    if data.get("restraints_config") is not None:
        from rgi_toolkit.config import resolve_restraints_config  # noqa: PLC0415

        data = {
            **data,
            "restraints_config": resolve_restraints_config(
                data["restraints_config"], base_dir=Path(path).parent
            ),
        }
    return data


def _inchi_atom_order(auxiliary: str, atom_count: int) -> list[int]:
    """Read original heavy-atom indices in canonical InChI order."""
    layer = re.search(r"(?:^|/)N:([^/]+)", auxiliary)
    if layer is None:
        raise ValueError("InChI auxiliary information has no atom correspondence")
    order = [int(value) - 1 for value in re.split("[,;]", layer.group(1))]
    if sorted(order) != list(range(atom_count)):
        raise ValueError("InChI atom correspondence is not a heavy-atom permutation")
    return order


def _source_ligand(entity: Ligand) -> Chem.Mol:
    """Preserve input chemistry while using the predictor's canonical atom names."""
    if entity.ccd_code is None:
        inchi = entity.inchi
        source = None
        source_order = None
        if entity.smiles is not None:
            source = Chem.RemoveAllHs(Chem.MolFromSmiles(entity.smiles))
            inchi, auxiliary = Chem.MolToInchiAndAuxInfo(source)
            source_order = _inchi_atom_order(auxiliary, source.GetNumAtoms())
        mol = Chem.MolFromInchi(inchi)
        if mol is None:
            raise ValueError("Cannot resolve BioEmu2 ligand InChI")
        mol = Chem.AddHs(mol)
        _assign_mol_atom_names(mol)
        for atom in mol.GetAtoms():
            atom.SetProp("atom_name", atom.GetProp("name"))
        mol = Chem.RemoveAllHs(mol)
        if source is None:
            return mol
        normalized_inchi, auxiliary = Chem.MolToInchiAndAuxInfo(mol)
        if normalized_inchi != inchi or source.GetNumAtoms() != mol.GetNumAtoms():
            raise ValueError("BioEmu2 InChI normalization changed the ligand identity")
        normalized_order = _inchi_atom_order(auxiliary, mol.GetNumAtoms())
        correspondence = dict(zip(normalized_order, source_order, strict=True))
        for normalized_index, source_index in correspondence.items():
            atom = source.GetAtomWithIdx(source_index)
            normalized_atom = mol.GetAtomWithIdx(normalized_index)
            if atom.GetAtomicNum() != normalized_atom.GetAtomicNum():
                raise ValueError("InChI atom correspondence changed an element")
            atom.SetProp("atom_name", normalized_atom.GetProp("atom_name"))
        normalized_edges = {
            tuple(sorted((correspondence[b.GetBeginAtomIdx()], correspondence[b.GetEndAtomIdx()])))
            for b in mol.GetBonds()
        }
        source_edges = {
            tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in source.GetBonds()
        }
        if normalized_edges != source_edges:
            raise ValueError("BioEmu2 InChI normalization changed ligand connectivity")
        return source

    component = _load_ccd_cifwrapper(entity.ccd_code)
    atoms = component._chem_comp_atom
    names = list(atoms.atom_id)
    lookup = {name: i for i, name in enumerate(names)}
    molecule = Chem.RWMol()
    for name, element, charge in zip(names, atoms.type_symbol, atoms.charge, strict=True):
        atom = Chem.Atom(0 if element == "X" else str(element).capitalize())
        atom.SetFormalCharge(0 if charge in {".", "?"} else int(float(charge)))
        atom.SetProp("atom_name", name)
        molecule.AddAtom(atom)
    bonds = component._chem_comp_bond
    if bonds is not None:
        for first, second, order, aromatic in zip(
            bonds.atom_id_1,
            bonds.atom_id_2,
            bonds.value_order,
            bonds.pdbx_aromatic_flag,
            strict=True,
        ):
            bond_type = (
                Chem.BondType.AROMATIC
                if aromatic == "Y" or aromatic is True
                else getattr(Chem.BondType, get_bondtype_from_pdbx_bondtype(order).name)
            )
            molecule.AddBond(lookup[first], lookup[second], bond_type)
    mol = molecule.GetMol()
    Chem.SanitizeMol(mol)
    try:
        coords = np.array(
            [
                atoms.pdbx_model_Cartn_x_ideal,
                atoms.pdbx_model_Cartn_y_ideal,
                atoms.pdbx_model_Cartn_z_ideal,
            ],
            dtype=float,
        ).T
    except ValueError:
        coords = np.array(
            [atoms.model_Cartn_x, atoms.model_Cartn_y, atoms.model_Cartn_z], dtype=float
        ).T
    conformer = Chem.Conformer(len(names))
    for index, xyz in enumerate(coords):
        conformer.SetAtomPosition(index, xyz)
    mol.AddConformer(conformer)
    Chem.AssignStereochemistryFrom3D(mol)
    return Chem.RemoveAllHs(mol)


def attach_restraints(batch, structure, system):
    """Attach plain adapter metadata to a padded batch, without changing features."""
    if system.restraints_config is None:
        return batch
    from rgi_toolkit.bioemu2 import BioEmu2Adapter  # noqa: PLC0415

    n = batch.coords.shape[1]
    real_atoms = int(structure.num_nodes)

    def padded(values, value=0):
        array = np.asarray(values)
        return np.pad(
            array,
            [(0, n - real_atoms), *[(0, 0)] * (array.ndim - 1)],
            constant_values=value,
        )

    types = {kind.value: name for kind, name in FRIENDLYNAME_PER_MOLTYPE.items()}
    ligand_mols = {}
    for entity in system.entities:
        if isinstance(entity, Ligand):
            source = _source_ligand(entity)
            ligand_mols.update(dict.fromkeys(entity.asym_ids, source))
    adapter = BioEmu2Adapter(
        chain_ids=system.chain_ids,
        atom_chain_indices=padded(structure.allatom_asym_id.cpu().numpy()),
        atom_to_token_map=padded(structure.token_idx_at_atom.cpu().numpy(), -1),
        atom_mask=batch.atom_pad_mask[0].cpu().numpy(),
        atom_name_codes=padded(structure.name_encoding.cpu().numpy()),
        residue_names=padded(np.array(structure.residue_labels, dtype=object), None),
        mol_types=padded(
            np.array([types[int(kind)] for kind in structure.mol_type.cpu()], dtype=object),
            None,
        ),
        elements=padded(structure.element.cpu().numpy()),
        ref_pos=padded(structure.template_coords.cpu().numpy()),
        ref_space_uid=padded(structure.residue_idx.cpu().numpy(), -1),
        conformer_chains=[
            chain
            for entity in system.entities
            if entity.conformer_restraints
            for chain in entity.asym_ids
        ],
        ligand_mols=ligand_mols,
    )
    return batch.replace(rgi_adapter=adapter, restraints_config=system.restraints_config)


def setup_restraints(batch, multiplicity):
    """Create fresh engine state for each structure and sampling batch."""
    config = getattr(batch, "restraints_config", None)
    if config is None:
        return None
    from rgi_toolkit.combined import CombinedRestraints  # noqa: PLC0415

    restraints = CombinedRestraints()
    restraints.setup(batch.rgi_adapter, nbatch=multiplicity, config=config)
    return restraints


def minimize_restraints(restraints, coords, step, sigma):
    """Keep context-parallel coordinates identical after rank-zero minimization."""
    if restraints is None:
        return coords
    if cp_sync.is_rank0():
        coords = restraints.minimize(coords, step, sigma)
    return cp_sync.broadcast_from_rank0(coords.contiguous())


def finalize_restraints(restraints, coords, step):
    if restraints is not None and cp_sync.is_rank0():
        restraints.finalize(coords, step)
