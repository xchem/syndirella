#!/usr/bin/env python3
"""
Regression test for the --substructure_check_products filter in
SlipperSynthesizer._collect_unique_sanitized_products (xchem/syndirella#17).

This feature previously had no test coverage anywhere in the suite.
"""

from types import SimpleNamespace

from rdkit import Chem

from syndirella.slipper.slipper_synthesizer.SlipperSynthesizer import SlipperSynthesizer


def _make_synthesizer(scaffold_smiles: str, substructure_check_products: bool) -> SlipperSynthesizer:
    reaction = SimpleNamespace(
        reaction_pattern=None,
        scaffold=Chem.MolFromSmiles(scaffold_smiles),
        reaction_name='dummy_reaction',
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
    )
    return SlipperSynthesizer(library=library, output_dir='/tmp',
                              substructure_check_products=substructure_check_products)


def _product_sets(*smiles: str):
    return [(Chem.MolFromSmiles(s),) for s in smiles]


def test_flag_off_keeps_non_superstructure_products():
    scaffold = 'O=C(NC1CC(C(F)(F)F)C1)c1cc2ccsc2[nH]1'
    synth = _make_synthesizer(scaffold, substructure_check_products=False)
    unrelated_product = 'O=C(NC1CC(C(F)(F)F)C1)c1ccc(Cl)cc1'  # not a superstructure of scaffold

    unique = synth._collect_unique_sanitized_products(_product_sets(unrelated_product), calc_difference=False)

    assert [smi for smi, _ in unique] == [Chem.MolToSmiles(Chem.MolFromSmiles(unrelated_product), isomericSmiles=False)]


def test_flag_on_drops_non_superstructure_products():
    scaffold = 'O=C(NC1CC(C(F)(F)F)C1)c1cc2ccsc2[nH]1'
    synth = _make_synthesizer(scaffold, substructure_check_products=True)
    unrelated_product = 'O=C(NC1CC(C(F)(F)F)C1)c1ccc(Cl)cc1'  # not a superstructure of scaffold

    unique = synth._collect_unique_sanitized_products(_product_sets(unrelated_product), calc_difference=False)

    assert unique == []


def test_flag_on_keeps_genuine_superstructure_products():
    scaffold = 'O=C(NC1CC(C(F)(F)F)C1)c1cc2ccsc2[nH]1'
    synth = _make_synthesizer(scaffold, substructure_check_products=True)
    # methyl-substituted elaboration that genuinely contains the original scaffold as a substructure
    superstructure_product = 'Cc1csc2[nH]c(C(=O)NC3CC(C(F)(F)F)C3)cc12'

    unique = synth._collect_unique_sanitized_products(_product_sets(superstructure_product), calc_difference=False)

    assert [smi for smi, _ in unique] == [
        Chem.MolToSmiles(Chem.MolFromSmiles(superstructure_product), isomericSmiles=False)]
