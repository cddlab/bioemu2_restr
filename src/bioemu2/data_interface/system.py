# Portions adapted from Boltz. See NOTICES.md and licenses/LICENSE.BOLTZ-MIT.

"""System specifies a molecular system composition in a both human- and machine-friendly way
for sampling with the model. It can be serialized to and from a yaml file. It implements thorough
validations to help users catch potential mistakes in the specification.
The schema design draws on PDBx/mmCIF. BioEmu2 extensions support
posttranslational modifications and noncanonical residues.
"""

import gzip
import logging
from collections.abc import Iterator
from itertools import count, product
from pathlib import Path
from string import ascii_uppercase
from tempfile import TemporaryDirectory
from typing import Any, Self

import mdtraj
import requests
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticUndefined, PydanticUndefinedType

from bioemu2.structure.all_atom_structure import AllAtomStructure
from bioemu2.structure.inference_inputs import topology_from_structure

from .cif import author_chain_ids_from_cif_file, system_entities_from_cif_file
from .structure_hierarchy import (
    DNA,
    RNA,
    Bond,
    Branched,
    Contact,
    Ligand,
    Protein,
    SystemEntity,
)

logger = logging.getLogger(__name__)

_RCSB_ASYMMETRIC_UNIT_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
_RCSB_BIOLOGICAL_ASSEMBLY_URL = (
    "https://files.rcsb.org/download/{pdb_id}-assembly{assembly_id}.cif.gz"
)


def _asym_ids() -> Iterator[str]:
    """A, B, ..., Z, AA, AB, ..."""
    for n in count(1):
        yield from ("".join(p) for p in product(ascii_uppercase, repeat=n))


class System(BaseModel):
    """Define the system to be sampled by entities and optional bonds that are not part of the
    standard polymer links. See the documentation of Polymer (Protein, RNA, DNA), Ligand, Branched,
    Atom and Bond for details. A System can be serialized to and from a yaml file, and can be
    converted by `AllAtomStructure.from_system` to an AllAtomStructure for inference.

    For the common case of protein sequences plus ligands, use
    `System.from_description(proteins=["MVLS..."], ligand_ccd_codes=["HEM"])`.

    Alternative to standard class instantiation, a System can be created from a dict via
    `System.model_validate(data: dict)`. Here is an example of the dict structure:
    ```python
    {
        "entities": [
            {
                "entity_type": "protein",
                "sequence": "MADCLTCEQXAEFKEAXSLF",
                "asym_ids": ["A", "B"],
                "mods": [
                    {"res_id": 10, "inchi": "InChI=1S/C6H13NO2/c1-2-3-4-5(7)6(8)9/h5H,2-4,7H2,1H3,(H,8,9)"},
                    {"res_id": 17, "ccd_code": "SEP"}
                ],
            },
            {"entity_type": "rna", "sequence": "GCAUAGC", "asym_ids": "C"},
            {"entity_type": "ligand", "smiles": "CC1=CC=CC=C1", "asym_ids": "lig1"},
        ],
        "bonds": [
            {
                "atom1": {"name": "SG", "res_name": "CYS", "res_id": 7, "asym_id": "A"},
                "atom2": {"name": "O2P", "res_name": "SEP", "res_id": 17, "asym_id": "B"},
            },
        ],
    }
    ```
    Different to the class signature, the dict representation requires `entity_type` field for
    distinguishing between Protein, RNA, DNA and Ligand. The dict version is compatible with the
    serialized yaml format.

    Fields
    ------
    entities: list[Polymer | Ligand | Branched] of length >= 1:
        A list of entities (Polymers, Ligands, or Branched) that make up the system. Each Entity
        defines the chemical composition for the chains in the system labelled by its asym_ids,
        which can be a single str or a list of them. Each asym_id should be unique.
        A Polymer can be Protein, RNA or DNA with a sequence given by one-letter code for canonical
        residues and "X" ("N" for nucleotide) placeholders. Noncanonical residues are specified in
        the `mods` field with a 1-based `res_id` and an InChI string or CCD code. A modification may
        use its biological sequence letter or a placeholder. Unannotated placeholders represent
        residues with unknown identity.
        A Ligand introduces a small molecule by either a SMILES string, an InChI string, or a CCD
        code for the corresponding asym_ids. The chemical string or CCD code will be validated.
        Repeated entities with the same definition will be merged by combining their asym_ids.
        A Branched entity defines branched polymers such as glycans. The mon_seq is a list of
        monomers, where each monomer is represented by a ModifiedResidue specified by a CCD code.
        The bonds and connectivity between monomers and with other entities must be defined in the
        `bonds` field.
    bonds: list[Bond], optional
        An optional list of Bonds to be added to the system. Each Bond connects two Atoms atom1 and
        atom2, each specified by an (atom) name, res_name, res_id (1-based index) and asym_id for
        its location in the system.
        The bond definitions will be validated against the underlying entities to ensure that the
        specified atoms exist in the corresponding entities, except for the atom names, which are
        not strictly checked. In case of modified residues, the res_name should be consistent with
        the provided CCD code if the residue is defined by a CCD, or the placeholder name typical
        to the PDB (i.e., UNK for Protein, DN for DNA and N for RNA) if defined by an InChI string.
        DO NOT include bonds that are part of the standard polymer links that can be inferred
        directly from the sequence, e.g., peptide bonds for Protein and phosphodiester bonds for
        RNA/DNA.
    asym_id_to_author_id: dict[str, str], optional
        Mapping from canonical label asym IDs to the corresponding author chain IDs. Imported
        systems populate this mapping in the same order as ``chain_ids``.
    """

    model_config = ConfigDict(extra="forbid")

    entities: list[SystemEntity] = Field(min_length=1)
    bonds: list[Bond] = Field(default_factory=list)  # An optional list of bonds
    asym_id_to_author_id: dict[str, str] = Field(default_factory=dict)
    restraints_config: dict[str, Any] | None = None

    def __init__(
        self,
        entities: list[SystemEntity],
        *,
        bonds: list[Bond] | PydanticUndefinedType = PydanticUndefined,
        asym_id_to_author_id: dict[str, str] | PydanticUndefinedType = PydanticUndefined,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            entities=entities,
            bonds=bonds,
            asym_id_to_author_id=asym_id_to_author_id,
            **kwargs,
        )

    @property
    def chain_ids(self) -> list[str]:
        return [asym_id for entity in self.entities for asym_id in entity.asym_ids]

    @property
    def topology(self) -> mdtraj.Topology:
        structure = AllAtomStructure.from_system(self, mode="structure")
        return topology_from_structure(structure, renamed_asym_ids=self.chain_ids)

    @field_validator("entities")
    @classmethod
    def _validate_entities(cls, v: list[SystemEntity]) -> list[SystemEntity]:
        """Validate entities' asym_ids are mutually exclusive."""
        asym_ids = []
        for entity in v:
            asym_ids.extend(entity.asym_ids)
        if len(asym_ids) != len(set(asym_ids)):
            raise ValueError("Entities contain repeated asym_ids.")

        for asym_id in asym_ids:
            if len(asym_id) > 4:
                raise ValueError(
                    f"asym_id '{asym_id}' is longer than 4 characters. biotite stores "
                    "AtomArray.chain_id with dtype U4, so longer asym_ids may be silently "
                    "truncated. Please use asym_ids with 4 or fewer characters."
                )

        # merge repeated entities with the same definition by combining their asym_ids
        unique_entities: list[SystemEntity] = []
        for entity in v:
            for index, unique_entity in enumerate(unique_entities):
                if entity == unique_entity:
                    unique_entities[index] = unique_entity.model_copy(
                        update={"asym_ids": [*unique_entity.asym_ids, *entity.asym_ids]},
                        deep=True,
                    )
                    break
            else:
                unique_entities.append(entity.model_copy(deep=True))
        return unique_entities

    @model_validator(mode="after")
    def _validate_bonds(self) -> Self:
        """Validate that bond definitions are consistent with the underlying entities."""
        asym_id_to_entity = {
            asym_id: entity for entity in self.entities for asym_id in entity.asym_ids
        }
        for bond in self.bonds:
            bond_message = f"according to the definition of {bond}"
            for atom in (bond.atom1, bond.atom2):
                if atom.asym_id not in asym_id_to_entity:
                    raise ValueError(
                        f"Atom {atom}'s asym_id '{atom.asym_id}' is not present in any entity.\n"
                        + bond_message
                    )
                entity = asym_id_to_entity[atom.asym_id]
                try:
                    entity.ensure_atom_in_entity(atom)
                except ValueError as e:
                    e.add_note(bond_message)  # Python 3.11+
                    raise e
        return self

    def validate_contacts(self, contacts: list[Contact]) -> None:
        """Validate that contact definitions are consistent with the underlying entities."""
        asym_id_to_entity = {
            asym_id: entity for entity in self.entities for asym_id in entity.asym_ids
        }
        for contact in contacts:
            contact_message = f"according to the definition of {contact}"
            for token in (contact.token1, contact.token2):
                if token.asym_id not in asym_id_to_entity:
                    raise ValueError(
                        f"ContactToken {token}'s asym_id '{token.asym_id}' is not present in any "
                        f"entity.\n" + contact_message
                    )
                entity = asym_id_to_entity[token.asym_id]
                if isinstance(entity, Ligand):
                    if not token.atom_name:
                        raise ValueError(
                            f"ContactToken {token} addresses a Ligand entity and must specify "
                            f"atom_name.\n" + contact_message
                        )
                    if token.res_id != 1:
                        raise ValueError(
                            f"ContactToken {token} addresses a Ligand entity with only one "
                            f"residue (res_id=1).\n" + contact_message
                        )
                else:
                    if isinstance(entity, Branched) and not token.atom_name:
                        raise ValueError(
                            f"ContactToken {token} addresses a branched entity and must specify "
                            f"atom_name.\n" + contact_message
                        )
                    num_residues = (
                        len(entity.mon_seq)
                        if isinstance(entity, Branched)
                        else len(entity.sequence)
                    )
                    if token.res_id > num_residues:
                        raise ValueError(
                            f"ContactToken {token}'s res_id '{token.res_id}' exceeds the "
                            f"{entity.entity_type} length {num_residues}.\n" + contact_message
                        )

    @model_validator(mode="after")
    def _validate_asym_id_to_author_id(self) -> Self:
        """Validate and order author IDs by the canonical label asym IDs."""
        if not self.asym_id_to_author_id:
            return self
        chain_ids = self.chain_ids
        if set(self.asym_id_to_author_id) != set(chain_ids):
            raise ValueError(
                "asym_id_to_author_id keys must exactly match the System's canonical asym IDs."
            )
        self.asym_id_to_author_id = {
            asym_id: self.asym_id_to_author_id[asym_id] for asym_id in chain_ids
        }
        return self

    def to_yaml(self, path: str | Path) -> None:
        """Write the system to a yaml file."""
        with open(path, "w") as file:
            yaml.safe_dump(
                self.model_dump(exclude_none=True),
                file,
                sort_keys=False,
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Read a system from YAML, ignoring top-level sampling conditions."""
        with open(path) as file:
            data = yaml.safe_load(file)
        if isinstance(data, dict):
            data.pop("mode", None)
            data.pop("contacts", None)
            if data.get("restraints_config") is not None:
                from bioemu2.restraints import resolve_system_restraints  # noqa: PLC0415

                data = resolve_system_restraints(data, path)
        return cls.model_validate(data)

    @classmethod
    def from_cif(
        cls,
        path: str | Path,
        *,
        include_ligands: bool = True,
        exclude_ligands: list[str] | None = None,
    ) -> Self:
        """Construct a system from a local mmCIF file.

        Waters are omitted. Polymer sequences include unresolved residues from the mmCIF entity
        tables, and supported modified residues and non-polymer entities are represented by CCD
        codes. Branched entities and explicit connections from ``_struct_conn`` are not imported;
        warnings are emitted when these unsupported records are present. Systems defined directly
        in Python or YAML can represent these entities and bonds. Label asym IDs are canonical;
        ``asym_id_to_author_id`` records the corresponding author chain IDs.

        Args:
          path: Path to a local, uncompressed mmCIF file.
          include_ligands: Whether to include non-water components.
          exclude_ligands: Optional ligand CCD codes to exclude. Cannot be combined with
            ``include_ligands=False``.

        Returns:
          A system describing the molecular system in the mmCIF file.
        """
        cif_path = Path(path)
        if not cif_path.is_file():
            raise FileNotFoundError(f"CIF file does not exist: {cif_path}")
        entities = system_entities_from_cif_file(cif_path)
        entities = cls._filter_entities(
            entities,
            include_ligands=include_ligands,
            exclude_ligands=exclude_ligands,
        )
        label_to_author = author_chain_ids_from_cif_file(cif_path)
        selected_asym_ids = [asym_id for entity in entities for asym_id in entity["asym_ids"]]
        return cls.model_validate(
            {
                "entities": entities,
                "asym_id_to_author_id": {
                    asym_id: label_to_author[asym_id] for asym_id in selected_asym_ids
                },
            }
        )

    @classmethod
    def from_pdb_id(
        cls,
        pdb_id: str,
        *,
        assembly_id: str | None = "1",
        auth_asym_ids: list[str] | None = None,
        label_asym_ids: list[str] | None = None,
        include_ligands: bool = True,
        exclude_ligands: list[str] | None = None,
    ) -> Self:
        """Download an RCSB mmCIF entry and construct a system.

        Returned systems use label asym IDs as their canonical chain IDs and ordering.
        ``asym_id_to_author_id`` records the corresponding author IDs. Branched entities and
        explicit connections from ``_struct_conn`` are not imported; warnings are emitted when
        these unsupported records are present. Systems defined directly in Python or YAML can
        represent these entities and bonds.

        Args:
          pdb_id: Four-character PDB identifier, such as ``"101m"``.
          assembly_id: Biological assembly ID, or ``None`` for the asymmetric unit.
          auth_asym_ids: Optional author chain IDs to include.
          label_asym_ids: Optional label chain IDs to include.
          include_ligands: Whether to include non-water components.
          exclude_ligands: Optional ligand CCD codes to exclude. Cannot be combined with
            ``include_ligands=False``.

        Returns:
          A system describing the downloaded RCSB entry.
        """
        if auth_asym_ids is not None and label_asym_ids is not None:
            raise ValueError("Provide either auth_asym_ids or label_asym_ids, not both.")

        normalized_pdb_id = pdb_id.upper()
        url_template = (
            _RCSB_BIOLOGICAL_ASSEMBLY_URL if assembly_id is not None else _RCSB_ASYMMETRIC_UNIT_URL
        )
        response = requests.get(
            url_template.format(pdb_id=normalized_pdb_id, assembly_id=assembly_id),
            timeout=60,
        )
        response.raise_for_status()

        with TemporaryDirectory() as tempdir:
            cif_path = Path(tempdir) / f"{normalized_pdb_id}.cif"
            content = (
                gzip.decompress(response.content) if assembly_id is not None else response.content
            )
            cif_path.write_bytes(content)
            entities = system_entities_from_cif_file(cif_path)
            label_to_author = author_chain_ids_from_cif_file(cif_path)
            available_label_asym_ids = list(label_to_author)
            available_auth_asym_ids = list(dict.fromkeys(label_to_author.values()))

            if label_asym_ids is not None:
                missing_label_asym_ids = [
                    asym_id for asym_id in label_asym_ids if asym_id not in available_label_asym_ids
                ]
                if missing_label_asym_ids:
                    raise ValueError(
                        f"Label asym IDs {missing_label_asym_ids} are not present. "
                        f"Available label asym IDs: {available_label_asym_ids}."
                    )
                selected_label_asym_ids = set(label_asym_ids)
            elif auth_asym_ids is not None:
                missing_auth_asym_ids = [
                    asym_id for asym_id in auth_asym_ids if asym_id not in available_auth_asym_ids
                ]
                if missing_auth_asym_ids:
                    raise ValueError(
                        f"Author asym IDs {missing_auth_asym_ids} are not present. "
                        f"Available author asym IDs: {available_auth_asym_ids}."
                    )
                selected_label_asym_ids = {
                    label_asym_id
                    for label_asym_id, auth_asym_id in label_to_author.items()
                    if auth_asym_id in auth_asym_ids
                }
            else:
                selected_label_asym_ids = set(available_label_asym_ids)

            chain_filtered_entities: list[dict[str, Any]] = []
            for entity in entities:
                entity_label_asym_ids = [
                    label_asym_id
                    for label_asym_id in entity["asym_ids"]
                    if label_asym_id in selected_label_asym_ids
                ]
                if entity_label_asym_ids:
                    chain_filtered_entities.append({**entity, "asym_ids": entity_label_asym_ids})
            selected_entities = cls._filter_entities(
                chain_filtered_entities,
                include_ligands=include_ligands,
                exclude_ligands=exclude_ligands,
            )
            selected_asym_ids = [
                asym_id for entity in selected_entities for asym_id in entity["asym_ids"]
            ]
            return cls.model_validate(
                {
                    "entities": selected_entities,
                    "asym_id_to_author_id": {
                        asym_id: label_to_author[asym_id] for asym_id in selected_asym_ids
                    },
                }
            )

    @classmethod
    def from_description(
        cls,
        *,
        proteins: list[str] | None = None,
        rna: list[str] | None = None,
        dna: list[str] | None = None,
        ligand_ccd_codes: list[str] | None = None,
        ligand_smiles: list[str] | None = None,
    ) -> Self:
        """Build a System from polymer sequences and ligands. Chain IDs are assigned A, B, C, ...
        in argument order (proteins, rna, dna, ligand_ccd_codes, ligand_smiles). Identical
        sequences are merged into one multi-chain entity. Use the class constructor for mods or
        bonds.
        """
        ids = _asym_ids()
        entities: list[SystemEntity] = [
            Protein(sequence=seq, asym_ids=[next(ids)]) for seq in proteins or []
        ]
        entities += [RNA(sequence=seq, asym_ids=[next(ids)]) for seq in rna or []]
        entities += [DNA(sequence=seq, asym_ids=[next(ids)]) for seq in dna or []]
        entities += [Ligand(ccd_code=c, asym_ids=[next(ids)]) for c in ligand_ccd_codes or []]
        entities += [Ligand(smiles=s, asym_ids=[next(ids)]) for s in ligand_smiles or []]
        return cls(entities=entities)

    @staticmethod
    def _filter_entities(
        entities: list[dict[str, Any]],
        *,
        include_ligands: bool,
        exclude_ligands: list[str] | None,
    ) -> list[dict[str, Any]]:
        if not include_ligands and exclude_ligands:
            raise ValueError("exclude_ligands cannot be combined with include_ligands=False.")
        if exclude_ligands is not None and any(
            not ccd_code.strip() for ccd_code in exclude_ligands
        ):
            raise ValueError("exclude_ligands must contain non-empty CCD code strings.")
        excluded_ligands = {ccd_code.strip().upper() for ccd_code in exclude_ligands or []}
        return [
            entity
            for entity in entities
            if entity["entity_type"] != "ligand"
            or (include_ligands and entity["ccd_code"].upper() not in excluded_ligands)
        ]

    def __str__(self) -> str:
        """Same YAML as `to_yaml` writes, so `print(system)` output can be pasted into a file."""
        return yaml.safe_dump(self.model_dump(exclude_none=True), sort_keys=False)


def load_system_and_contacts(
    path: str | Path,
) -> tuple[System, list[Contact] | None]:
    """Load a molecular system and optional contacts from a YAML file."""
    with open(path) as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("System YAML must contain a mapping.")
    if data.get("restraints_config") is not None:
        from bioemu2.restraints import resolve_system_restraints  # noqa: PLC0415

        data = resolve_system_restraints(data, path)
    # System forbids extra fields, so separate its fields from the optional contacts.
    system = System.model_validate({k: data.pop(k) for k in System.model_fields if k in data})
    raw_contacts = data.pop("contacts", None)
    if data:
        raise ValueError(f"Unknown fields in system YAML: {sorted(data)}")
    if raw_contacts is None:
        return system, None
    # Contact validates its own fields; the System checks that the chains exist.
    contacts = [Contact.model_validate(contact) for contact in raw_contacts]
    system.validate_contacts(contacts)
    return system, contacts
