"""Exercise RGI through real BioEmu2 input preparation and sampler loops."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rdkit import Chem
from rgi_toolkit.combined import CombinedRestraints

from bioemu2.biomoa.models.edm_schedule import EDMSchedule
from bioemu2.data_interface import Ligand, Protein, System
from bioemu2.data_interface.system import load_system_and_contacts
from bioemu2.denoising.edm import edm_sample_schedule, edm_sampler
from bioemu2.denoising.steering.edm_fkc import edm_fkc_sampler
from bioemu2.denoising.steering.edm_steered import edm_steered_sampler
from bioemu2.restraints import _source_ligand, setup_restraints
from bioemu2.structure.preparation import prepare_sampling_inputs


class GaussianModel:
    edm_schedule = EDMSchedule()

    def compute_static_context(self, batch, trunk_embeddings, multiplicity):
        return SimpleNamespace(coords_cond=None, is_dynamics_conditioned=None, dt=None)

    def denoise(self, context, trunk_embeddings, coords, sigmas):
        return SimpleNamespace(denoised_coords=coords / (sigmas[:, None, None] ** 2 + 1))


def config(target):
    return {
        "gpu": False,
        "max_iter": 100,
        "distance_restraints_config": [
            {
                "atom_selection1": "chain A and resid 1 and name CA",
                "atom_selection2": "chain A and resid 3 and name CA",
                "harmonic": {"target_distance": target},
            }
        ],
    }


def protein_batch(target=None):
    system = System(
        entities=[Protein(sequence="AAA", asym_ids="A")],
        restraints_config=None if target is None else config(target),
    )
    return prepare_sampling_inputs(system, device=torch.device("cpu"))[1]


def sample(batch, sampler=edm_sampler, **kwargs):
    torch.manual_seed(11)
    return sampler(
        batch=batch,
        score_model=GaussianModel(),
        trunk_embeddings=None,
        multiplicity=2,
        num_steps=8,
        step_scale=1.0,
        noise_scale=1.0,
        **kwargs,
    ).coords


@pytest.mark.parametrize("sampler", [edm_sampler, edm_steered_sampler])
def test_distance_reaches_each_target_and_fresh_inputs_do_not_leak(sampler, monkeypatch):
    monkeypatch.setenv("RGI_DISABLE_COMPILE", "1")
    baseline = sample(protein_batch(), sampler)
    for target in [10.0, 20.0]:
        batch = protein_batch(target)
        positions = sample(batch, sampler)
        ca = [r.index for r in batch.rgi_adapter.iter_atoms() if r.name == "CA"]
        distances = torch.linalg.vector_norm(positions[:, ca[0]] - positions[:, ca[2]], dim=-1)
        torch.testing.assert_close(distances, torch.full((2,), target), atol=0.005, rtol=0)
        assert torch.isfinite(positions).all()
        assert not torch.equal(positions, baseline)
    assert torch.equal(sample(protein_batch(), sampler), baseline)


def test_gate_uses_pre_churn_sigma(monkeypatch):
    seen = []
    original = CombinedRestraints.minimize

    def record(self, coords, step, sigma):
        seen.append((step, sigma))
        return original(self, coords, step, sigma)

    monkeypatch.setenv("RGI_DISABLE_COMPILE", "1")
    monkeypatch.setattr(CombinedRestraints, "minimize", record)
    sample(protein_batch(10.0))
    expected = edm_sample_schedule(8, device=torch.device("cpu"))
    assert [step for step, _ in seen] == list(range(8))
    np.testing.assert_allclose([sigma for _, sigma in seen], expected[:-1].numpy())


def test_smiles_chirality_cistrans_charge_and_opt_in(monkeypatch):
    monkeypatch.setenv("RGI_DISABLE_COMPILE", "1")
    system = System(
        entities=[
            Protein(sequence="AA", asym_ids="A"),
            Ligand(
                smiles="C[N+](C)(C)CC/C=C/[C@@H](O)C",
                asym_ids="L",
                conformer_restraints=True,
            ),
        ],
        restraints_config={"gpu": False, "conformer_restraints_config": {"vdw": {"weight": 0}}},
    )
    _, batch = prepare_sampling_inputs(system, device=torch.device("cpu"))
    (ligand,) = batch.rgi_adapter.iter_ligand_confs()
    assert sum(a.GetFormalCharge() for a in ligand.mol.GetAtoms()) == 1
    assert "@" in Chem.MolToSmiles(ligand.stereo_mol)
    assert "/" in Chem.MolToSmiles(ligand.stereo_mol)
    restraints = setup_restraints(batch, 2)
    assert restraints.spec.bond.mask.sum() > 0
    assert restraints.spec.angle.mask.sum() > 0
    assert restraints.spec.chiral.mask.sum() > 0
    assert restraints.spec.cistrans.mask.sum() > 0
    system.entities[1].conformer_restraints = False
    _, off = prepare_sampling_inputs(system, device=torch.device("cpu"))
    assert not next(off.rgi_adapter.iter_ligand_confs()).conformer_restraints


def test_ccd_names_and_full_chemistry_are_resolved_offline(monkeypatch):
    monkeypatch.setenv("BIOEMU2_CCD_CACHE", str(Path(__file__).parent / "structure" / "ccd"))
    mol = _source_ligand(Ligand(ccd_code="ATP", asym_ids="L"))
    assert mol.GetNumAtoms() == 31
    assert all(atom.HasProp("atom_name") for atom in mol.GetAtoms())
    assert len({a.GetProp("atom_name") for a in mol.GetAtoms()}) == 31
    assert "@" in Chem.MolToSmiles(mol)


@pytest.mark.parametrize("smiles", ["CC(=N)O", "[NH2+]=C(N)N", "CC(=N)O.[Cl-]"])
def test_smiles_chemistry_survives_inchi_normalization(smiles):
    source = Chem.RemoveAllHs(Chem.MolFromSmiles(smiles))
    ligand = Ligand(smiles=smiles, asym_ids="B", conformer_restraints=True)
    restored = _source_ligand(ligand)
    assert Chem.MolToSmiles(restored) == Chem.MolToSmiles(source)
    names = [atom.GetProp("atom_name") for atom in restored.GetAtoms()]
    assert len(set(names)) == source.GetNumAtoms()
    system = System(
        entities=[Protein(sequence="AAA", asym_ids="A"), ligand],
        restraints_config={"conformer_restraints_config": {"vdw": False}, "gpu": False},
    )
    _, batch = prepare_sampling_inputs(system, device=torch.device("cpu"))
    conformer = next(batch.rgi_adapter.iter_ligand_confs())
    assert Chem.MolToSmiles(conformer.mol) == Chem.MolToSmiles(source)


def test_both_yaml_loaders_resolve_relative_config(tmp_path):
    (tmp_path / "restraints.yaml").write_text("gpu: false\nmax_iter: 17\n")
    system_yaml = tmp_path / "system.yaml"
    system_yaml.write_text(
        "entities:\n- entity_type: protein\n  sequence: AAA\n  asym_ids: A\n"
        "restraints_config:\n  config_path: restraints.yaml\n"
    )
    for system in [System.from_yaml(system_yaml), load_system_and_contacts(system_yaml)[0]]:
        assert system.restraints_config == {"gpu": False, "max_iter": 17}


def test_fkc_rejects_unsupported_guidance_instead_of_ignoring_it():
    with pytest.raises(NotImplementedError, match="RGI supports"):
        edm_fkc_sampler(
            batch=protein_batch(10.0),
            score_model=GaussianModel(),
            trunk_embeddings=None,
            num_steps=2,
        )
