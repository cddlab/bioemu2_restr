# Portions adapted from Boltz. See NOTICES.md and licenses/LICENSE.BOLTZ-MIT.

"""Classes to represent the hierarchy of a biomolecular structure following roughly the PDBx/mmCIF
data schema with automatic parsing and validation via Pydantic and can be exported to internal
AllAtomStructure representation.
"""

from typing import Annotated, Any, ClassVar, Final, Self, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    SerializerFunctionWrapHandler,
    Tag,
    computed_field,
    field_validator,
    model_serializer,
    model_validator,
)

from .const import (
    DNA_LETTER_TO_TOKEN,
    MOLTYPE_TO_UNKNOWN_CCD,
    MOLTYPE_TO_UNKTOKEN,
    PROT_LETTER_TO_TOKEN,
    RNA_LETTER_TO_TOKEN,
    BondType,
    MolType,
    Token,
)
from .ligand_validators import CCDCodeField, InChIField, SMILESField

# Residues, that are present in the PROT_LETTER_TO_TOKEN and are noncanonical.
NONCANONICAL_RESIDUES: Final[list[str]] = ["J", "B", "Z", "O", "U"]
# allowed residues
VALID_CHARS_PER_MOLTYPE: Final[dict[MolType, set[str]]] = {
    MolType.PROTEIN: set(PROT_LETTER_TO_TOKEN.keys()) - set(NONCANONICAL_RESIDUES),
    MolType.RNA: set(RNA_LETTER_TO_TOKEN.keys()),
    MolType.DNA: set(DNA_LETTER_TO_TOKEN.keys()),
}
MAP_LETTER_TO_TOKEN_PER_MOLTYPE: Final[dict[MolType, dict[str, Token]]] = {
    MolType.PROTEIN: PROT_LETTER_TO_TOKEN,
    MolType.RNA: RNA_LETTER_TO_TOKEN,
    MolType.DNA: DNA_LETTER_TO_TOKEN,
}
# Friendly names for the molecule types
FRIENDLYNAME_PER_MOLTYPE: Final[dict[MolType, str]] = {
    MolType.PROTEIN: "protein",
    MolType.RNA: "rna",
    MolType.DNA: "dna",
    MolType.NONPOLYMER: "ligand",
}


__all__ = [
    "Atom",
    "Bond",
    "Contact",
    "ContactToken",
    "Ligand",
    "ModifiedResidue",
    "Polymer",
    "SystemEntity",
    "Protein",
    "RNA",
    "DNA",
]


class Atom(BaseModel):
    """
    A class contains information to identify an atom in a structure.

    Fields
    ------
    name: str, atom name, e.g. "CA"
    res_name: str, residue name, e.g. "ALA"
    res_id: positive int, 1-based residue id, e.g. 123. Should only be 1 (default) for ligands
    asym_id: str, chain identifier, e.g. "A"
    """

    name: str  # e.g. "CA"
    res_name: str  # e.g. "ALA"
    res_id: int = Field(1, gt=0)  # positive number like 123. Should only be 1 (default) for ligands
    asym_id: str  # e.g. "A"

    model_config = ConfigDict(extra="forbid", frozen=True)

    def __str__(self) -> str:
        """String representation of Atom object with class name and round brackets."""
        return repr(self)


class Bond(BaseModel):
    """
    A class contains information to identify a bond in a structure.
    Bonds are defined by two atoms. The order of the atoms does not matter.
    ``bond_type`` is preserved in the schema but excluded from serialized model
    input, where every connection is treated as a single bond.

    Fields
    ------
    atom1: Atom, first atom of the bond
    atom2: Atom, second atom of the bond
    bond_type: BondType, the type of bond (default: SINGLE)
    """

    atom1: Atom
    atom2: Atom
    bond_type: BondType = Field(default=BondType.SINGLE, exclude=True)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("bond_type", mode="before")
    @classmethod
    def _coerce_bond_type(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return BondType[v.upper()]
            except KeyError:
                raise ValueError(
                    f"Invalid bond type name: '{v}'. Valid names: {[bt.name for bt in BondType]}"
                )
        return v

    @model_serializer(mode="wrap")
    def _serialize_model(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        if self.bond_type != BondType.SINGLE:
            data["bond_type"] = self.bond_type.name.lower()
        return data

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Bond):
            return NotImplemented
        atoms_match = {self.atom1, self.atom2} == {other.atom1, other.atom2}
        return atoms_match and self.bond_type == other.bond_type

    def __str__(self) -> str:
        """String representation of Bond object with class name and round brackets."""
        return repr(self)


class _Entity(BaseModel):
    """
    Base class of entities that can be either protein, RNA/DNA, or ligand. Should not be
    instantiated directly. Use subclasses instead for specific molecule types, e.g. Protein, RNA,
    DNA and Ligand.

    Fields
    ------
    ENTITY_TYPE: (bioemu2.data_interface.const.MolType) the type of the entity
    asym_ids: list[str]
        A list of chain identifier(s) for the entity. Can be provided as a single string, e.g. "A"
        (will be converted to ["A"] internally), comma or space separated string "A,B" (will be
        converted to ["A", "B"]) or ["B", "C"].
    entity_type: str
        A computed field to return the friendly name of the entity type, e.g. "protein", "rna",
        "dna" or "ligand". Will be ignored when set during initialization.
    """

    ENTITY_TYPE: ClassVar[MolType]
    asym_ids: list[str]
    conformer_restraints: bool = Field(default=False, strict=True)

    @field_validator("asym_ids", mode="before")
    def ensure_list(cls, value):
        if isinstance(value, str):
            my_str = value.replace(",", " ")
            return my_str.split()
        return value

    @computed_field  # type: ignore # incompatibility between Pydantic and mypy
    @property
    def entity_type(self) -> str:
        return FRIENDLYNAME_PER_MOLTYPE[self.ENTITY_TYPE]

    def ensure_atom_in_entity(self, atom: Atom) -> None:
        """Raise ValueError if the given atom does not belong to this entity."""
        if atom.asym_id not in self.asym_ids:
            raise ValueError(
                f"Atom {atom}'s asym_id '{atom.asym_id}' does not belong to entity {self}."
            )

    @model_serializer(mode="wrap")
    def serialize_model(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        """Custom serializer to include entity_type as the first item."""
        serialized = handler(self)
        serialized.pop("entity_type")
        if not self.conformer_restraints:
            serialized.pop("conformer_restraints", None)
        return {"entity_type": self.entity_type, **serialized}

    def __str__(self) -> str:
        """String representation with class name and round brackets."""
        return repr(self)

    def __eq__(self, other: object) -> bool:
        """Returns True if other and self are the same entity EXCEPT for the asym_ids."""
        if not isinstance(other, _Entity):
            return NotImplemented
        if self.ENTITY_TYPE != other.ENTITY_TYPE:
            return False
        self_fields = self.__class__.model_fields
        other_fields = other.__class__.model_fields
        common_fields = set(self_fields.keys()).intersection(set(other_fields.keys()))
        for field in common_fields:
            if field == "asym_ids":
                continue
            if getattr(self, field) != getattr(other, field):
                return False
        for field in self_fields:
            if field not in common_fields and getattr(self, field) is not None:
                return False
        for field in other_fields:
            if field not in common_fields and getattr(other, field) is not None:
                return False
        return True


class Ligand(_Entity):
    """
    A class to represent a ligand with SMILES/InChI/CCD code representation. Only one of the
    following should be set:
    - inchi
    - smiles
    - ccd_code
    If multiple are set, a ValueError will be raised.
    """

    ENTITY_TYPE = MolType.NONPOLYMER
    inchi: InChIField = None
    smiles: SMILESField = None
    ccd_code: CCDCodeField = None

    @model_validator(mode="after")
    def _check_one_of(self):
        if (
            sum(
                [
                    (self.inchi is not None),
                    (self.smiles is not None),
                    (self.ccd_code is not None),
                ]
            )
            != 1
        ):
            raise ValueError("One and only one of inchi, smiles or ccd_code should be provided.")
        return self

    def ensure_atom_in_entity(self, atom: Atom) -> None:
        """Raise ValueError if the given atom does not belong to this entity."""
        super().ensure_atom_in_entity(atom)
        if atom.res_id != 1:
            raise ValueError(
                f"Atom {atom}'s res_id '{atom.res_id}' must be 1 for {type(self)} entity {self}."
            )


class ContactToken(BaseModel):
    """
    Identify a contact endpoint by chain, residue, and optional name for atomistic token.

    Fields
    ------
    asym_id: Chain identifier, e.g. "A".
    res_id: 1-based residue index. Required for residue contacts; atom-named contacts
        default to residue 1.
    atom_name: Atom name for an atomistic token. If omitted for a modified
        polymer residue, its CA or C1' token is selected. Required for branched
        entities and ligands.
    """

    asym_id: str
    res_id: int = Field(default=1, gt=0)
    atom_name: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_explicit_res_id(self) -> Self:
        if self.atom_name is None and "res_id" not in self.model_fields_set:
            raise ValueError("res_id must be explicitly set when atom_name is None")
        return self

    @model_serializer(mode="wrap")
    def serialize_model(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        fields = handler(self)
        if self.atom_name is not None and self.res_id == 1:
            fields.pop("res_id")
        return fields

    def __str__(self) -> str:
        """String representation of ContactToken object with class name and round brackets."""
        return repr(self)


class Contact(BaseModel):
    """
    A class specifying a contact constraint between two tokens (residues or
    ligand atoms), used to condition inference on a desired contact. Analogous
    to Bond but does not imply a covalent connection: it only marks the token
    pair as being in contact within max_distance for the contact_conditioning /
    contact_threshold model features.

    Fields
    ------
    token1: First endpoint, specified as a ContactToken or a mapping with named
        fields. Use asym_id and res_id for a residue; add atom_name to select
        an atomistic token. Ligand atoms need asym_id and atom_name.
    token2: Second endpoint, in the same format.
    max_distance: Maximum distance in Angstrom between the two tokens (default: 6.0).
        Supported range is [4.0, 20.0], matching the cutoff range used at training time.
    """

    token1: ContactToken
    token2: ContactToken
    max_distance: float = Field(default=6.0, ge=4.0, le=20.0)

    model_config = ConfigDict(extra="forbid", frozen=True)

    def __str__(self) -> str:
        """String representation of Contact object with class name and round brackets."""
        return repr(self)


class ModifiedResidue(BaseModel):
    """
    A class to represent a chemically modified non-canonical residue (e.g., posttranslationally
    modified residue for protein) to represent with a 1-based polymer `res_id`, InChI string
    `inchi`, or CCD code `ccd_code`. `res_id` is required for polymer modifications and omitted
    for Branched monomers.
    """

    res_id: int | None = Field(default=None, gt=0)
    inchi: InChIField = None
    ccd_code: CCDCodeField = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _check_one_of(self):
        if not (self.inchi is None) ^ (self.ccd_code is None):
            raise ValueError("Only one of inchi and ccd_code should be provided.")
        return self

    @property
    def has_ccd(self) -> bool:
        """Return True if this ModifiedResidue is represented by a CCD code, False if represented
        by an InChI string.
        """
        return self.ccd_code is not None


class _Polymer(_Entity):
    """Base class for polymers (protein, RNA, DNA). A dummy letter marks an unidentified residue
    ("X" for protein, "N" for RNA or DNA). Explicit modifications are listed in `mods` with their
    1-based `res_id` and may retain their biological sequence letter. Should not be instantiated
    directly. Use subclasses instead for specific molecule types, e.g., Protein, RNA and DNA.
    """

    sequence: str = Field(min_length=1)  # e.g. "ACDEFGHIKLMNPQRSTVWY"
    mods: list[ModifiedResidue] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_mods(self) -> Self:
        """Validate that modifications address distinct positions in the polymer sequence."""
        mod_positions = []
        for mod in self.mods:
            if mod.res_id is None:
                placeholder = MOLTYPE_TO_UNKTOKEN[self.ENTITY_TYPE].singleletter
                raise ValueError(
                    f"Every {self.entity_type} modification must include a 1-based res_id. "
                    f"Use {placeholder} for an unidentified residue without a mods entry."
                )
            mod_positions.append(mod.res_id)
        if len(mod_positions) != len(set(mod_positions)):
            raise ValueError("Polymer modifications contain repeated res_id values.")
        invalid_positions = {
            position for position in mod_positions if position < 1 or position > len(self.sequence)
        }
        if invalid_positions:
            raise ValueError(
                f"Modification positions {sorted(invalid_positions)} are out of range for a "
                f"sequence of length {len(self.sequence)}."
            )
        return self

    @property
    def mods_by_res_id(self) -> dict[int, ModifiedResidue]:
        """Return explicit modifications keyed by their 1-based sequence position."""
        return {mod.res_id: mod for mod in self.mods if mod.res_id is not None}

    @field_validator("sequence")
    @classmethod
    def _validate_sequence(cls, v):
        """Validate sequence to contain only valid characters."""
        for char in v:
            if char not in VALID_CHARS_PER_MOLTYPE[cls.ENTITY_TYPE]:
                raise ValueError(f"Invalid character '{char}' in {cls.ENTITY_TYPE} sequence.")
        return v

    def get_token_sequence(self) -> list[Token]:
        """Map the polymer's one-letter sequence to model tokens."""
        letter_to_token_map = MAP_LETTER_TO_TOKEN_PER_MOLTYPE[self.ENTITY_TYPE]
        return [letter_to_token_map[res_code] for res_code in self.sequence]

    @property
    def residue_sequence(self) -> list[str]:
        """Return residue names resolved from the sequence and explicit modifications."""
        raise NotImplementedError("Use subclasses to implement residue_sequence property.")

    def ensure_atom_in_entity(self, atom: Atom) -> None:
        """Raise ValueError if the given atom does not belong to this entity."""
        super().ensure_atom_in_entity(atom)
        if atom.res_id > len(self.sequence):
            raise ValueError(
                f"Atom {atom}'s res_id '{atom.res_id}' exceeds the sequence length {len(self.sequence)} of {self.entity_type} {self}."
            )


class Protein(_Polymer):
    """A class for representing a protein chain. Posttranslational modifications are listed in
    `mods` as position-addressed `ModifiedResidue` values and may retain their biological sequence
    letter. The sequence letter is also used for MSA searches. An unmodified "X" marks an
    unidentified residue.
    """

    ENTITY_TYPE = MolType.PROTEIN

    @property
    def residue_sequence(self) -> list[str]:
        """Return residue names with explicit modifications and unknown residues "UNK"."""
        residues = [token.threeletter for token in self.get_token_sequence()]
        for position, modification in self.mods_by_res_id.items():
            residues[position - 1] = (
                modification.ccd_code or MOLTYPE_TO_UNKNOWN_CCD[self.ENTITY_TYPE]
            )
        unknown_residue = MOLTYPE_TO_UNKTOKEN[self.ENTITY_TYPE].threeletter
        return ["UNK" if residue == unknown_residue else residue for residue in residues]

    def ensure_atom_in_entity(self, atom: Atom) -> None:
        """Raise ValueError if the given atom does not belong to this entity."""
        super().ensure_atom_in_entity(atom)
        # check whether that residue is at the given res_id
        expected_res_name = self.residue_sequence[atom.res_id - 1]
        if atom.res_name.upper() != expected_res_name:
            raise ValueError(
                f"Atom {atom}'s res_name '{atom.res_name}' does not match the residue '{expected_res_name}' at res_id '{atom.res_id}' in {self.entity_type} sequence {self}."
            )


class RNA(_Polymer):
    """A class for representing an RNA chain. Chemical modifications are listed in `mods` as
    position-addressed `ModifiedResidue` values and may retain their biological sequence letter.
    An unmodified "N" marks an unidentified residue.
    """

    ENTITY_TYPE = MolType.RNA

    @property
    def residue_sequence(self) -> list[str]:
        """Return residue names with explicit modifications applied."""
        residues = [token.singleletter for token in self.get_token_sequence()]
        for position, modification in self.mods_by_res_id.items():
            residues[position - 1] = (
                modification.ccd_code or MOLTYPE_TO_UNKNOWN_CCD[self.ENTITY_TYPE]
            )
        return residues

    def ensure_atom_in_entity(self, atom: Atom) -> None:
        """Raise ValueError if the given atom does not belong to this entity."""
        super().ensure_atom_in_entity(atom)
        # check whether that residue is at the given res_id
        expected_res_name = self.residue_sequence[atom.res_id - 1]
        dummy_letter = MOLTYPE_TO_UNKTOKEN[self.ENTITY_TYPE].singleletter
        if atom.res_name.upper() != expected_res_name and not (
            atom.res_name.upper() == "UNK" and expected_res_name == dummy_letter
        ):
            raise ValueError(
                f"Atom {atom}'s res_name '{atom.res_name}' does not match the residue '{expected_res_name}' at res_id '{atom.res_id}' in {self.entity_type} sequence {self}."
            )


class DNA(_Polymer):
    """A class for representing a DNA chain. Chemical modifications are listed in `mods` as
    position-addressed `ModifiedResidue` values and may retain their biological sequence letter.
    An unmodified "N" marks an unidentified residue.
    """

    ENTITY_TYPE = MolType.DNA

    @property
    def residue_sequence(self) -> list[str]:
        """Return residue names with modifications applied. Unmodified DNA names begin with D.

        Canonical residues have names such as DA, DC, DG and DT; an unidentified residue is DN.
        """
        residues = ["D" + token.singleletter for token in self.get_token_sequence()]
        for position, modification in self.mods_by_res_id.items():
            residues[position - 1] = (
                modification.ccd_code or MOLTYPE_TO_UNKNOWN_CCD[self.ENTITY_TYPE]
            )
        return residues

    def ensure_atom_in_entity(self, atom: Atom) -> None:
        """Raise ValueError if the given atom does not belong to this entity."""
        super().ensure_atom_in_entity(atom)
        # check whether that residue is at the given res_id
        expected_res_name = self.residue_sequence[atom.res_id - 1]
        dummy_name = "D" + MOLTYPE_TO_UNKTOKEN[self.ENTITY_TYPE].singleletter
        if atom.res_name.upper() != expected_res_name and not (
            atom.res_name.upper() == "UNK" and expected_res_name == dummy_name
        ):
            raise ValueError(
                f"Atom {atom}'s res_name '{atom.res_name}' does not match the residue '{expected_res_name}' at res_id '{atom.res_id}' in {self.entity_type} sequence {self}."
            )


# Pydantic magic for handling serialization and deserialization of Polymer subclasses
def polymer_discriminator(v: _Polymer | dict) -> str | None:
    if isinstance(v, _Polymer):
        return v.entity_type
    elif isinstance(v, dict) and "entity_type" in v and isinstance(v["entity_type"], str):
        return v["entity_type"].lower()
    else:
        # return None if the discriminator value isn't found
        return None


# This is the type annotation for Polymer that can be used in other places
# to enable automatic (de)serialization of Polymer subclasses.
Polymer = Annotated[
    Union[
        Annotated[Protein, Tag(FRIENDLYNAME_PER_MOLTYPE[MolType.PROTEIN])],
        Annotated[RNA, Tag(FRIENDLYNAME_PER_MOLTYPE[MolType.RNA])],
        Annotated[DNA, Tag(FRIENDLYNAME_PER_MOLTYPE[MolType.DNA])],
    ],
    Discriminator(polymer_discriminator),
]


class Branched(_Entity):
    """A class for representing branched molecules that cannot be represented by a linear sequence,
    e.g. glycans. The monomer chemical structure should be listed in the `mon_seq` field as a list
    of `ModifiedResidue` encapsulating corresponding CCD codes. Most importantly, the bonds between
    the monomers and between this entity and other entities should be explicitly provided as a list
    of `Bond` when defining the system. Branched molecules are treated
    as non-polymers.
    """

    ENTITY_TYPE = MolType.NONPOLYMER
    mon_seq: list[ModifiedResidue] = Field(min_length=1, default_factory=list)

    @computed_field  # type: ignore[prop-decorator]  # incompatibility between Pydantic and mypy
    @property
    def entity_type(self) -> str:
        """Override the serialized entity discriminator for branched molecules."""
        return "branched"

    @field_validator("mon_seq")
    @classmethod
    def _mon_seq_uses_ccd(cls, mon_seq):
        """Validate that mon_seq uses CCD codes to represent the monomers."""
        for mon in mon_seq:
            if not mon.has_ccd:
                raise ValueError("mon_seq should use CCD codes to represent the monomers.")
        return mon_seq

    @property
    def residue_sequence(self) -> list[str]:
        """Return a list of monomer names."""
        residues: list[str] = []
        for mon in self.mon_seq:
            assert mon.ccd_code is not None, "Checked by validator."
            residues.append(mon.ccd_code)
        return residues

    def ensure_atom_in_entity(self, atom: Atom) -> None:
        """Raise ValueError if the given atom does not belong to this entity."""
        super().ensure_atom_in_entity(atom)
        if atom.res_id > len(self.mon_seq):
            raise ValueError(
                f"Atom {atom}'s res_id '{atom.res_id}' exceeds the monomer length "
                f"{len(self.mon_seq)} of {self.entity_type} {self}."
            )
        # check whether that residue is at the given res_id
        expected_res_name = self.residue_sequence[atom.res_id - 1]
        if atom.res_name.upper() != expected_res_name:
            raise ValueError(
                f"Atom {atom}'s res_name '{atom.res_name}' does not match the monomer "
                f"'{expected_res_name}' at res_id '{atom.res_id}' in {self.entity_type} mon_seq "
                f"{self}."
            )


def _system_entity_discriminator(value: object) -> str | None:
    """Return an entity's discriminator tag from YAML data or a model instance."""
    if isinstance(value, dict):
        entity_type = value.get("entity_type")
    else:
        entity_type = getattr(value, "entity_type", None)
    return entity_type.lower() if isinstance(entity_type, str) else None


SystemEntity = Annotated[
    Union[
        Annotated[Protein, Tag("protein")],
        Annotated[RNA, Tag("rna")],
        Annotated[DNA, Tag("dna")],
        Annotated[Ligand, Tag("ligand")],
        Annotated[Branched, Tag("branched")],
    ],
    Discriminator(_system_entity_discriminator),
]
