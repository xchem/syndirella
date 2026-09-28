#!/usr/bin/env python3
"""
Targeted tests for the selectivity-flag handling in
SlipperSynthesizer.filter_analogues_on_smarts (xchem/syndirella#17):

- A) selectivity_issue_contains_reaction_atoms_of_both_reactants must not be raised when the
     scaffold's own reactant for a role already exhibits that same ambiguity.
- B) more_reaction_centres_added must be raised (informationally) when an analogue matches its own
     reactant SMARTS more times than the scaffold's own reactant did.
"""

from types import SimpleNamespace

import pandas as pd
from rdkit import Chem

from syndirella.slipper.slipper_synthesizer.SlipperSynthesizer import SlipperSynthesizer

ACID_SMARTS = '[CX3;+0:2](=[O;H0;D1;+0:3])-[O;H1;D1;+0]'
AMINE_SMARTS = '[#7;H2;D1;+0:5]'
ACID_MOL = Chem.MolFromSmarts(ACID_SMARTS)
AMINE_MOL = Chem.MolFromSmarts(AMINE_SMARTS)

REACTION_NAME = 'dummy_amide'
R1_COL = f'r1_{REACTION_NAME}'
R2_COL = f'r2_{REACTION_NAME}'
R1_NUM_MATCHES_COL = f'{R1_COL}_num_matches'


def _check_other_reactant_smarts(analogues_mols, reactant_smarts, matched_smarts_to_reactant):
    """Faithful stand-in for Library.check_analogue_contains_other_reactant_smarts_pattern."""
    other_smarts = [s for s in matched_smarts_to_reactant if s != reactant_smarts][0]
    other_mol = Chem.MolFromSmarts(other_smarts)
    other_prefix = matched_smarts_to_reactant[other_smarts][2]
    return [bool(m.GetSubstructMatches(other_mol)) for m in analogues_mols], other_prefix


def _make_synthesizer(scaffold_r1_smiles: str) -> SlipperSynthesizer:
    """Build a SlipperSynthesizer with a mocked Library/Reaction whose r1 role is matched to
    scaffold_r1_smiles (r2 role is a fixed, unambiguous amine)."""
    scaffold_r1_mol = Chem.MolFromSmiles(scaffold_r1_smiles)
    scaffold_r2_mol = Chem.MolFromSmiles('CCCN')  # unambiguous primary amine, no acid group

    matched_smarts_to_reactant = {
        ACID_SMARTS: (scaffold_r1_mol, [], 'r1'),
        AMINE_SMARTS: (scaffold_r2_mol, [], 'r2'),
    }
    matched_smarts_index_to_reactant = {
        1: (scaffold_r1_mol, [], ACID_SMARTS),
        2: (scaffold_r2_mol, [], AMINE_SMARTS),
    }
    reaction = SimpleNamespace(
        scaffold=Chem.MolFromSmiles('CC'),
        reaction_name=REACTION_NAME,
        matched_smarts_to_reactant=matched_smarts_to_reactant,
        matched_smarts_index_to_reactant=matched_smarts_index_to_reactant,
    )
    library = SimpleNamespace(
        route_uuid='route-1',
        current_step=1,
        num_steps=1,
        atom_diff_min=-999,
        atom_diff_max=999,
        reaction=reaction,
        id='lib-1',
        elab_single_reactant_int=None,
        check_analogue_contains_other_reactant_smarts_pattern=(
            lambda analogues_mols, reactant_smarts: _check_other_reactant_smarts(
                analogues_mols, reactant_smarts, matched_smarts_to_reactant)),
    )
    return SlipperSynthesizer(library=library, output_dir='/tmp')


def _analogue_row(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    num_matches = len(mol.GetSubstructMatches(ACID_MOL))
    return {
        'r1_mol': mol,
        R1_COL: num_matches > 0,
        R1_NUM_MATCHES_COL: num_matches,
        R2_COL: bool(mol.GetSubstructMatches(AMINE_MOL)),
    }


def test_selectivity_flag_set_when_scaffold_reactant_is_unambiguous():
    """Scaffold's own r1 (plain benzoic acid) does not itself match the amine role, so an
    analogue that matches both roles is a genuine, new selectivity issue and should be flagged."""
    synth = _make_synthesizer(scaffold_r1_smiles='OC(=O)c1ccccc1')  # benzoic acid: acid only
    df = pd.DataFrame([_analogue_row('NCC(=O)O')])  # glycine: matches both acid and amine roles

    out = synth.filter_analogues_on_smarts(df, (R1_COL, R2_COL), 'r1')

    assert len(out) == 1
    assert out.loc[0, 'flag'] == ['selectivity_issue_contains_reaction_atoms_of_both_reactants']


def test_selectivity_flag_suppressed_when_scaffold_reactant_is_already_ambiguous():
    """Scaffold's own r1 (glycine) already matches both roles inherently -- an elaborated analogue
    that also matches both roles should NOT be flagged, since it's not a new issue."""
    synth = _make_synthesizer(scaffold_r1_smiles='NCC(=O)O')  # glycine: matches both roles
    df = pd.DataFrame([_analogue_row('NCCC(=O)O')])  # beta-alanine: also matches both roles

    out = synth.filter_analogues_on_smarts(df, (R1_COL, R2_COL), 'r1')

    assert len(out) == 1
    assert out.loc[0, 'flag'] is None


def test_more_reaction_centres_added_when_analogue_has_more_matches_than_scaffold():
    """Scaffold's own r1 (benzoic acid) has exactly one acid group. An analogue with two acid
    groups introduces an extra reaction centre and should be flagged for review, without being
    excluded."""
    synth = _make_synthesizer(scaffold_r1_smiles='OC(=O)c1ccccc1')  # benzoic acid: 1 acid group
    df = pd.DataFrame([_analogue_row('OC(=O)c1ccc(C(=O)O)cc1')])  # terephthalic acid: 2 acid groups

    out = synth.filter_analogues_on_smarts(df, (R1_COL, R2_COL), 'r1')

    assert len(out) == 1
    assert out.loc[0, 'flag'] == ['more_reaction_centres_added']


def test_no_extra_flag_when_analogue_has_same_number_of_matches_as_scaffold():
    """An analogue with the same number of reaction centres as the scaffold's own reactant should
    not be flagged for introducing more."""
    synth = _make_synthesizer(scaffold_r1_smiles='OC(=O)c1ccccc1')  # benzoic acid: 1 acid group
    df = pd.DataFrame([_analogue_row('OC(=O)c1ccc(C)cc1')])  # 4-methylbenzoic acid: 1 acid group

    out = synth.filter_analogues_on_smarts(df, (R1_COL, R2_COL), 'r1')

    assert len(out) == 1
    assert out.loc[0, 'flag'] is None


def test_both_flags_can_co_occur_on_the_same_row():
    """An analogue that is both ambiguous (matches both roles, and the scaffold isn't) and
    introduces more reaction centres than the scaffold should carry both flags."""
    synth = _make_synthesizer(scaffold_r1_smiles='OC(=O)c1ccccc1')  # benzoic acid: unambiguous, 1 match
    # glycine analogue with an extra acid group: matches both roles AND has 2 acid matches
    df = pd.DataFrame([_analogue_row('NC(C(=O)O)C(=O)O')])

    out = synth.filter_analogues_on_smarts(df, (R1_COL, R2_COL), 'r1')

    assert len(out) == 1
    assert set(out.loc[0, 'flag']) == {
        'selectivity_issue_contains_reaction_atoms_of_both_reactants',
        'more_reaction_centres_added',
    }
