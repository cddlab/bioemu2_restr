"""Shared system and MSA preparation for inference."""

import mdtraj
import torch

from bioemu2.conditioning.method import SamplingMode
from bioemu2.data_interface import Contact, Protein, System
from bioemu2.msa.alignment import MSA
from bioemu2.msa.colabfold import tokenize_msa
from bioemu2.msa.features import NUM_MSA_TOKENS, process_msa_features
from bioemu2.structure import AllAtomStructure, BatchAllAtomStructure
from bioemu2.structure.batch import batch_collate
from bioemu2.structure.inference_inputs import topology_from_structure


def prepare_structure(
    system: System,
    *,
    msa: MSA,
    mode: SamplingMode,
    contacts: list[Contact],
) -> AllAtomStructure:
    """Build an atom-level structure with MSA features from a molecular system.

    ``system`` describes the molecular entities and bonds, not an existing
    ``AllAtomStructure``. Validate its contacts and MSA chain mapping, then return
    the newly constructed structure.
    """
    system.validate_contacts(contacts)
    protein_chain_ids = [
        chain_id
        for entity in system.entities
        if isinstance(entity, Protein)
        for chain_id in entity.asym_ids
    ]
    if set(msa.chain_ids) != set(protein_chain_ids):
        raise ValueError(
            f"MSA must contain chains {sorted(protein_chain_ids)}, got {sorted(msa.chain_ids)}."
        )
    structure = AllAtomStructure.from_system(
        system,
        mode=mode,
        contacts=contacts,
    )

    chain_indices = {chain_id: index for index, chain_id in enumerate(system.chain_ids)}
    unpaired = {chain_indices[chain_id]: a3m for chain_id, a3m in msa.unpaired_a3m.items()}
    paired = (
        None
        if msa.paired_a3m is None
        else {chain_indices[chain_id]: a3m for chain_id, a3m in msa.paired_a3m.items()}
    )
    msa_array, deletion, paired_array = tokenize_msa(
        structure=structure,
        unpaired_a3ms=unpaired,
        paired_a3ms=paired,
    )
    structure = structure.replace(
        **process_msa_features(
            msa=torch.as_tensor(msa_array, dtype=torch.long),
            deletion=torch.as_tensor(deletion, dtype=torch.float32),
            paired=torch.as_tensor(paired_array, dtype=torch.long),
            num_tokens_msa=NUM_MSA_TOKENS,
        )
    )
    return structure


def prepare_sampling_inputs(
    system: System,
    *,
    device: torch.device,
    window_size: int = 32,
) -> tuple[mdtraj.Topology, BatchAllAtomStructure]:
    """Build the unconditioned topology and model inputs used during sampling."""
    structure = AllAtomStructure.from_system(system)
    structure = structure.replace(coords=torch.zeros_like(structure.coords))
    topology = topology_from_structure(structure, renamed_asym_ids=system.chain_ids)
    batch = batch_collate([structure], window_size=window_size).to(device)
    if system.restraints_config is not None:
        from bioemu2.restraints import attach_restraints  # noqa: PLC0415

        batch = attach_restraints(batch, structure, system)
    return topology, batch
