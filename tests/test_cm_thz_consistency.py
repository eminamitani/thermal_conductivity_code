from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

import pyAF.thermal_conductivity_AF as af
from pyAF.constants import physical_constants


def _two_atom_mode_basis():
    translations = np.zeros((6, 3))
    differences = np.zeros((6, 3))
    for axis in range(3):
        translations[axis, axis] = 1.0 / np.sqrt(2.0)
        translations[axis + 3, axis] = 1.0 / np.sqrt(2.0)
        differences[axis, axis] = 1.0 / np.sqrt(2.0)
        differences[axis + 3, axis] = -1.0 / np.sqrt(2.0)
    return np.column_stack((translations, differences))


def _gate_setup(
    omega_threshould=0.01,
    two_dim=False,
    allow_2d_flexural_fail=False,
):
    return SimpleNamespace(
        omega_threshould=omega_threshould,
        two_dim=two_dim,
        translation_overlap_min=0.99,
        translation_capture_min=2.999,
        translation_leakage_max=0.001,
        asr_residual_max=1.0e-10,
        negative_mode_tolerance_cm=0.1,
        flexural_polarization_min=0.8,
        allow_2d_flexural_fail=allow_2d_flexural_fail,
    )


def test_mode_heat_capacity_has_finite_zero_frequency_limit():
    pc = physical_constants()

    assert af._mode_heat_capacity(0.0, pc.BOLTZMANN_CONSTANT) == pc.BOLTZMANN_CONSTANT
    assert np.isclose(
        af._mode_heat_capacity(1.0e-9, pc.BOLTZMANN_CONSTANT),
        pc.BOLTZMANN_CONSTANT,
    )


def test_cm_and_thz_routes_are_unit_invariant(tmp_path, monkeypatch):
    structure_file = tmp_path / "structure.vasp"
    dyn_file = tmp_path / "Dyn.form"
    atoms = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    write(structure_file, atoms, format="vasp")
    mode_basis = _two_atom_mode_basis()
    dyn = mode_basis @ np.diag([0.0, 0.0, 0.0, 1.0, 2.0, 3.0]) @ mode_basis.T
    np.savetxt(dyn_file, dyn)

    rng = np.random.default_rng(20260718)
    velocity = rng.normal(size=(6, 6))
    velocity = 0.5 * (velocity + velocity.T)
    np.fill_diagonal(velocity, 0.0)
    monkeypatch.setattr(
        af,
        "get_Vij_from_flat",
        lambda _structure_file, _dyn: (velocity, 0.7 * velocity, 1.3 * velocity),
    )
    monkeypatch.chdir(tmp_path)

    setup = SimpleNamespace(
        structure_file=str(structure_file),
        dyn_file=str(dyn_file),
        style="lammps-regular",
        temperature=300.0,
        broadening_factor=1.25,
        using_mean_spacing=False,
        omega_threshould=0.01,
        broadening_threshould=0.01,
        two_dim=False,
        symmetrize_fc=False,
        fix_diag=True,
    )

    cm = af.get_thermal_conductivity(setup)
    thz = af.get_thermal_conductivity_THz_unit(setup)
    pc = physical_constants()
    unit_factor = af._angular_thz_per_wavenumber(pc)

    np.testing.assert_allclose(
        cm["freq"] * unit_factor,
        thz["freq"] * 2.0 * np.pi,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        cm["diffusivity"],
        thz["diffusivity"],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        cm["thermal_conductivity"],
        thz["thermal_conductivity"],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert cm["mode_gate"]["status"] == "PASS"
    assert thz["mode_gate"]["status"] == "PASS"


def test_gate2_review_can_pass_when_cutoff_isolates_translation_subspace():
    basis = _two_atom_mode_basis()
    dyn = basis @ np.diag(
        [4.0e-8, 5.0e-8, 6.0e-8, 1.0e-8, 1.0, 2.0]
    ) @ basis.T
    eigenvalues, eigenvectors = np.linalg.eigh(dyn)
    pc = physical_constants()
    frequencies_cm = np.sqrt(eigenvalues) * pc.scale_cm

    diagnostics, _, _ = af._evaluate_mode_gates(
        _gate_setup(omega_threshould=1.0),
        eigenvalues,
        eigenvectors,
        np.array([28.0855, 28.0855]),
        frequencies_cm,
    )

    assert diagnostics["gate2_status"] == "PASS_WITH_CUTOFF"
    assert diagnostics["cutoff_excluded_nontranslation_count"] == 1
    assert diagnostics["active_translation_leakage"] < 0.001


def test_gate1_rejects_significant_negative_nontranslation_mode():
    basis = _two_atom_mode_basis()
    dyn = basis @ np.diag(
        [0.0, 0.0, 0.0, -0.01, 1.0, 2.0]
    ) @ basis.T
    eigenvalues, eigenvectors = np.linalg.eigh(dyn)
    pc = physical_constants()
    frequencies_cm = (
        np.sign(eigenvalues)
        * np.sqrt(np.abs(eigenvalues))
        * pc.scale_cm
    )

    with pytest.raises(ValueError, match="Mode Gate 1 failed"):
        af._evaluate_mode_gates(
            _gate_setup(),
            eigenvalues,
            eigenvectors,
            np.array([28.0855, 28.0855]),
            frequencies_cm,
        )


def test_3d_gate_reports_all_small_negative_nontranslation_modes():
    basis = _two_atom_mode_basis()
    pc = physical_constants()
    small_negative = -(0.05 / pc.scale_cm) ** 2
    dyn = basis @ np.diag(
        [0.0, 0.0, 0.0, small_negative, 1.0, 2.0]
    ) @ basis.T
    eigenvalues, eigenvectors = np.linalg.eigh(dyn)
    frequencies_cm = (
        np.sign(eigenvalues)
        * np.sqrt(np.abs(eigenvalues))
        * pc.scale_cm
    )

    diagnostics, _, _ = af._evaluate_mode_gates(
        _gate_setup(omega_threshould=1.0),
        eigenvalues,
        eigenvectors,
        np.array([28.0855, 28.0855]),
        frequencies_cm,
    )

    assert diagnostics["gate1_status"] == "REVIEW"
    assert diagnostics["small_negative_nontranslation_count"] == 1
    assert diagnostics["negative_za_like_count"] == 0
    assert any(
        "1 negative non-translation mode(s)" in warning
        for warning in diagnostics["warnings"]
    )


def test_gate2_review_stops_when_cutoff_does_not_isolate_translations():
    eigenvalues = np.arange(1.0, 7.0)
    eigenvectors = np.eye(6)
    pc = physical_constants()
    frequencies_cm = np.sqrt(eigenvalues) * pc.scale_cm

    with pytest.raises(ValueError, match="Mode Gate 2 requires review"):
        af._evaluate_mode_gates(
            _gate_setup(),
            eigenvalues,
            eigenvectors,
            np.array([28.0855, 28.0855]),
            frequencies_cm,
        )


def test_small_negative_za_mode_is_2d_review():
    basis = _two_atom_mode_basis()
    pc = physical_constants()
    small_negative = -(0.05 / pc.scale_cm) ** 2
    dyn = basis @ np.diag(
        [0.0, 0.0, 0.0, 1.0, 2.0, small_negative]
    ) @ basis.T
    eigenvalues, eigenvectors = np.linalg.eigh(dyn)
    frequencies_cm = (
        np.sign(eigenvalues)
        * np.sqrt(np.abs(eigenvalues))
        * pc.scale_cm
    )

    diagnostics, _, _ = af._evaluate_mode_gates(
        _gate_setup(omega_threshould=1.0, two_dim=True),
        eigenvalues,
        eigenvectors,
        np.array([12.011, 12.011]),
        frequencies_cm,
    )

    assert diagnostics["gate1_status"] == "REVIEW_2D_FLEXURAL"
    assert diagnostics["gate2_status"] == "PASS_WITH_CUTOFF_2D"
    assert diagnostics["negative_za_like_count"] == 1
    assert diagnostics["negative_non_za_count"] == 0


def test_small_negative_non_za_mode_is_2d_generic_review():
    basis = _two_atom_mode_basis()
    pc = physical_constants()
    small_negative = -(0.05 / pc.scale_cm) ** 2
    dyn = basis @ np.diag(
        [0.0, 0.0, 0.0, small_negative, 1.0, 2.0]
    ) @ basis.T
    eigenvalues, eigenvectors = np.linalg.eigh(dyn)
    frequencies_cm = (
        np.sign(eigenvalues)
        * np.sqrt(np.abs(eigenvalues))
        * pc.scale_cm
    )

    diagnostics, _, _ = af._evaluate_mode_gates(
        _gate_setup(omega_threshould=1.0, two_dim=True),
        eigenvalues,
        eigenvectors,
        np.array([12.011, 12.011]),
        frequencies_cm,
    )

    assert diagnostics["gate1_status"] == "REVIEW"
    assert diagnostics["negative_za_like_count"] == 0
    assert diagnostics["negative_non_za_count"] == 1
    assert any(
        "negative mode(s) are non-ZA" in warning
        for warning in diagnostics["warnings"]
    )


def test_mixed_small_2d_negative_modes_report_za_and_non_za():
    basis = _two_atom_mode_basis()
    pc = physical_constants()
    small_negative_x = -(0.04 / pc.scale_cm) ** 2
    small_negative_z = -(0.05 / pc.scale_cm) ** 2
    dyn = basis @ np.diag(
        [
            0.0,
            0.0,
            0.0,
            small_negative_x,
            1.0,
            small_negative_z,
        ]
    ) @ basis.T
    eigenvalues, eigenvectors = np.linalg.eigh(dyn)
    frequencies_cm = (
        np.sign(eigenvalues)
        * np.sqrt(np.abs(eigenvalues))
        * pc.scale_cm
    )

    diagnostics, _, _ = af._evaluate_mode_gates(
        _gate_setup(omega_threshould=1.0, two_dim=True),
        eigenvalues,
        eigenvectors,
        np.array([12.011, 12.011]),
        frequencies_cm,
    )

    assert diagnostics["gate1_status"] == "REVIEW"
    assert diagnostics["small_negative_nontranslation_count"] == 2
    assert diagnostics["negative_za_like_count"] == 1
    assert diagnostics["negative_non_za_count"] == 1
    assert any(
        "negative mode(s) are ZA-like" in warning
        for warning in diagnostics["warnings"]
    )
    assert any(
        "negative mode(s) are non-ZA" in warning
        for warning in diagnostics["warnings"]
    )


def test_significant_negative_za_mode_fails_by_default():
    basis = _two_atom_mode_basis()
    dyn = basis @ np.diag(
        [0.0, 0.0, 0.0, 1.0, 2.0, -0.01]
    ) @ basis.T
    eigenvalues, eigenvectors = np.linalg.eigh(dyn)
    pc = physical_constants()
    frequencies_cm = (
        np.sign(eigenvalues)
        * np.sqrt(np.abs(eigenvalues))
        * pc.scale_cm
    )

    with pytest.raises(ValueError, match="FAIL_2D_FLEXURAL"):
        af._evaluate_mode_gates(
            _gate_setup(omega_threshould=10.0, two_dim=True),
            eigenvalues,
            eigenvectors,
            np.array([12.011, 12.011]),
            frequencies_cm,
        )


def test_2d_flexural_fail_can_continue_only_with_explicit_test_override():
    basis = _two_atom_mode_basis()
    dyn = basis @ np.diag(
        [0.0, 0.0, 0.0, 1.0, 2.0, -0.01]
    ) @ basis.T
    eigenvalues, eigenvectors = np.linalg.eigh(dyn)
    pc = physical_constants()
    frequencies_cm = (
        np.sign(eigenvalues)
        * np.sqrt(np.abs(eigenvalues))
        * pc.scale_cm
    )

    diagnostics, _, _ = af._evaluate_mode_gates(
        _gate_setup(
            omega_threshould=10.0,
            two_dim=True,
            allow_2d_flexural_fail=True,
        ),
        eigenvalues,
        eigenvectors,
        np.array([12.011, 12.011]),
        frequencies_cm,
    )

    assert diagnostics["gate1_status"] == "FAIL_2D_FLEXURAL"
    assert diagnostics["status"] == "TEST_ONLY_FAIL_2D_FLEXURAL"


def test_2d_override_cannot_bypass_significant_non_za_instability():
    basis = _two_atom_mode_basis()
    dyn = basis @ np.diag(
        [0.0, 0.0, 0.0, -0.01, 1.0, -0.02]
    ) @ basis.T
    eigenvalues, eigenvectors = np.linalg.eigh(dyn)
    pc = physical_constants()
    frequencies_cm = (
        np.sign(eigenvalues)
        * np.sqrt(np.abs(eigenvalues))
        * pc.scale_cm
    )

    with pytest.raises(
        ValueError,
        match="significant negative non-translation",
    ):
        af._evaluate_mode_gates(
            _gate_setup(
                omega_threshould=10.0,
                two_dim=True,
                allow_2d_flexural_fail=True,
            ),
            eigenvalues,
            eigenvectors,
            np.array([12.011, 12.011]),
            frequencies_cm,
        )


def test_2d_gate_counts_positive_za_modes_removed_by_cutoff():
    basis = _two_atom_mode_basis()
    pc = physical_constants()
    low_positive_za = (1.0 / pc.scale_cm) ** 2
    dyn = basis @ np.diag(
        [0.0, 0.0, 0.0, 1.0, 2.0, low_positive_za]
    ) @ basis.T
    eigenvalues, eigenvectors = np.linalg.eigh(dyn)
    frequencies_cm = (
        np.sign(eigenvalues)
        * np.sqrt(np.abs(eigenvalues))
        * pc.scale_cm
    )

    diagnostics, _, _ = af._evaluate_mode_gates(
        _gate_setup(omega_threshould=10.0, two_dim=True),
        eigenvalues,
        eigenvectors,
        np.array([12.011, 12.011]),
        frequencies_cm,
    )

    assert diagnostics["cutoff_excluded_positive_nontranslation_count"] == 1
    assert diagnostics["cutoff_excluded_za_mode_count"] == 1
