"""Tests for the Result tab's loader, driven by the synthetic result fixtures.

Gradio is imported (the handler returns ``gr.update`` objects), but no Psi4 and no
browser: these pin what each kind of result file makes visible and what lands in each
table.
"""
import os

import pytest

pytest.importorskip("gradio")
pytest.importorskip("plotly")

from psi4_webui.result import RESULT_OUTPUT_COUNT, on_load_result_file, on_show_ecd_spectrum  # noqa: E402
from psi4_webui.utils import RESULT_SUFFIX, write_json  # noqa: E402


def _load(tmp_path, result, name="job"):
    """Write ``result`` into ``tmp_path`` and run it through the loader."""
    file_name = name + RESULT_SUFFIX
    write_json(str(tmp_path / file_name), result)
    return on_load_result_file(str(tmp_path), file_name)


# Positional indices into the loader's return tuple. Named so the assertions below read
# as intent rather than as magic numbers.
STATUS, RESULT_STATE = 0, 1
ENERGY_ACCORDION, ENERGY_TEXT, DIPOLE_TEXT, MO_TABLE = 2, 3, 4, 5
OPT_ACCORDION, OPT_PLOT = 6, 7
FREQ_ACCORDION, FREQ_TABLE, IR_PLOT, THERMO_TABLE, FREQ_NOTE = 8, 9, 10, 11, 12
EXC_ACCORDION, EXC_TABLE, ABS_PLOT, ECD_BUTTON = 13, 14, 15, 16
ORBITAL_ACCORDION, VIZ_DROPDOWN, VIZ_HTML = 17, 18, 19
EMISSION_ACCORDION, EMISSION_TABLE, EMISSION_PLOT = 20, 21, 22


def _visible(update):
    """Read the ``visible`` flag out of a ``gr.update`` result."""
    return dict(update).get("visible")


def test_return_tuple_matches_the_declared_output_count(tmp_path, single_point_result):
    # The loader returns one big positional tuple that must stay index-aligned with the
    # outputs list in result_tab_content. Asserting the length here is what turns a
    # silent misalignment -- which shows up as values landing in the wrong components --
    # into a failing test.
    assert len(_load(tmp_path, single_point_result)) == RESULT_OUTPUT_COUNT


class TestSinglePoint:
    def test_reveals_only_the_energy_accordion(self, tmp_path, single_point_result):
        outputs = _load(tmp_path, single_point_result)
        assert _visible(outputs[ENERGY_ACCORDION]) is True
        assert _visible(outputs[OPT_ACCORDION]) is False
        assert _visible(outputs[FREQ_ACCORDION]) is False
        assert _visible(outputs[EXC_ACCORDION]) is False

    def test_reports_energy_and_dipole(self, tmp_path, single_point_result):
        outputs = _load(tmp_path, single_point_result)
        assert "-74.96294665" in outputs[ENERGY_TEXT]
        assert "1.7257" in outputs[DIPOLE_TEXT]

    def test_mo_table_labels_the_frontier_orbitals(self, tmp_path, single_point_result):
        labels = outputs_labels = list(_load(tmp_path, single_point_result)[MO_TABLE]["Molecular orbital"])
        # homo_index 4 is 0-based, so MO 5 is the HOMO and MO 6 the LUMO.
        assert labels[4] == "MO 5 (HOMO)"
        assert labels[5] == "MO 6 (LUMO)"
        assert len(outputs_labels) == len(single_point_result["mo_energies_hartree"])

    def test_orbital_viewer_appears_when_a_wavefunction_was_saved(self, tmp_path, single_point_result):
        outputs = _load(tmp_path, single_point_result)
        assert _visible(outputs[ORBITAL_ACCORDION]) is True
        assert "Electron density" in dict(outputs[VIZ_DROPDOWN])["choices"]

    def test_orbital_viewer_is_hidden_without_a_wavefunction(self, tmp_path, single_point_result):
        # to_file() refuses custom basis sets, and a calculation that otherwise succeeded
        # should still load -- just without the viewer.
        single_point_result["wavefunction_file"] = None
        assert _visible(_load(tmp_path, single_point_result)[ORBITAL_ACCORDION]) is False

    def test_spin_density_is_offered_only_for_open_shell_results(self, tmp_path, single_point_result):
        # A closed-shell spin density is identically zero; offering it would render a
        # blank surface and look like a bug.
        closed = dict(_load(tmp_path, single_point_result)[VIZ_DROPDOWN])["choices"]
        assert "Spin density" not in closed

        single_point_result["n_beta"] = 4
        open_shell = dict(_load(tmp_path, single_point_result)[VIZ_DROPDOWN])["choices"]
        assert "Spin density" in open_shell


class TestOptimization:
    def test_reveals_the_optimization_accordion_with_a_plot(self, tmp_path, optimization_result):
        outputs = _load(tmp_path, optimization_result)
        assert _visible(outputs[OPT_ACCORDION]) is True
        assert outputs[OPT_PLOT] is not None

    def test_plots_every_step(self, tmp_path, optimization_result):
        figure = _load(tmp_path, optimization_result)[OPT_PLOT]
        assert len(figure.data[0].y) == optimization_result["optimization"]["n_steps"]


class TestFrequency:
    def test_table_pairs_each_frequency_with_its_intensity(self, tmp_path, frequency_result):
        table = _load(tmp_path, frequency_result)[FREQ_TABLE]
        assert len(table) == 3
        assert table["Frequency (cm-1)"][0] == "2170.79"
        assert table["IR intensity (km/mol)"][0] == "7.2100"

    def test_thermochemistry_is_converted_to_kcal(self, tmp_path, frequency_result):
        row = _load(tmp_path, frequency_result)[THERMO_TABLE].iloc[0]
        # 0.0243735 hartree * 627.509 = 15.294 kcal/mol
        assert float(row["Zero-point energy (kcal/mol)"]) == pytest.approx(15.294, abs=0.01)
        assert float(row["Entropy (cal/mol/K)"]) == pytest.approx(45.485, abs=0.001)

    def test_a_clean_minimum_carries_no_warning(self, tmp_path, frequency_result):
        assert _load(tmp_path, frequency_result)[FREQ_NOTE] == ""

    def test_a_saddle_point_is_called_out(self, tmp_path, saddle_point_result):
        # An imaginary frequency invalidates the thermochemistry underneath it, so the
        # user has to be told rather than left to spot a minus sign in the table.
        note = _load(tmp_path, saddle_point_result)[FREQ_NOTE]
        assert "imaginary" in note.lower()

    def test_imaginary_modes_are_excluded_from_the_spectrum(self, tmp_path, saddle_point_result):
        # An imaginary mode has no absorption band; broadening it would put a phantom
        # peak into the spectrum at a mirrored wavenumber.
        figure = _load(tmp_path, saddle_point_result)[IR_PLOT]
        assert figure is not None
        assert float(min(figure.data[0].x)) > 0


class TestExcitations:
    def test_reveals_the_excitation_accordion(self, tmp_path, tddft_result):
        outputs = _load(tmp_path, tddft_result)
        assert _visible(outputs[EXC_ACCORDION]) is True
        assert len(outputs[EXC_TABLE]) == 3
        assert outputs[ABS_PLOT] is not None

    def test_ecd_button_enabled_when_rotatory_strengths_are_non_zero(self, tmp_path, tddft_result):
        assert dict(_load(tmp_path, tddft_result)[ECD_BUTTON])["interactive"] is True

    def test_ecd_button_disabled_for_an_achiral_molecule(self, tmp_path, tddft_result):
        # Every rotatory strength of an achiral molecule is zero, and its "ECD spectrum"
        # is a flat line -- an enabled button there just invites confusion.
        for state in tddft_result["excitations"]:
            state["rotatory_strength"] = 0.0
        assert dict(_load(tmp_path, tddft_result)[ECD_BUTTON])["interactive"] is False

    def test_ecd_spectrum_preserves_the_sign_of_the_cotton_effects(self, tddft_result):
        # Two enantiomers differ only by the sign of every R, so their ECD spectra are
        # mirror images. Taking magnitudes would destroy the only information ECD carries.
        figure = on_show_ecd_spectrum(tddft_result)
        assert figure is not None
        y = figure.data[0].y
        assert min(y) < 0 < max(y)


class TestEmission:
    def test_reveals_the_emission_accordion(self, tmp_path, emission_result):
        outputs = _load(tmp_path, emission_result)
        assert _visible(outputs[EMISSION_ACCORDION]) is True
        assert outputs[EMISSION_PLOT] is not None

    def test_reports_the_emission_energy_and_both_state_energies(self, tmp_path, emission_result):
        row = _load(tmp_path, emission_result)[EMISSION_TABLE].iloc[0]
        assert row["Emitting state"] == "S1"
        assert float(row["Wavelength (nm)"]) == pytest.approx(567.8, abs=0.01)
        assert float(row["Emission energy (eV)"]) == pytest.approx(2.1837, abs=1e-4)

    def test_the_band_sits_at_the_emission_wavelength(self, tmp_path, emission_result):
        figure = _load(tmp_path, emission_result)[EMISSION_PLOT]
        peak_x = figure.data[0].x[list(figure.data[0].y).index(max(figure.data[0].y))]
        assert float(peak_x) == pytest.approx(567.8, abs=5.0)

    def test_other_calculation_types_do_not_show_it(self, tmp_path, single_point_result):
        assert _visible(_load(tmp_path, single_point_result)[EMISSION_ACCORDION]) is False


class TestFailedAndMalformed:
    def test_a_failed_result_reports_its_error_and_hides_everything(self, tmp_path, failed_result):
        outputs = _load(tmp_path, failed_result)
        assert "NOSUCHBASIS" in outputs[STATUS]
        assert "red" in outputs[STATUS]
        for index in (ENERGY_ACCORDION, OPT_ACCORDION, FREQ_ACCORDION, EXC_ACCORDION, ORBITAL_ACCORDION):
            assert _visible(outputs[index]) is False

    def test_a_missing_file_is_reported_not_raised(self, tmp_path):
        # Handlers in this app never propagate exceptions to Gradio; they return a red
        # status instead.
        outputs = on_load_result_file(str(tmp_path), "does_not_exist" + RESULT_SUFFIX)
        assert "red" in outputs[STATUS]
        assert len(outputs) == RESULT_OUTPUT_COUNT

    def test_unreadable_json_is_reported_not_raised(self, tmp_path):
        path = tmp_path / ("broken" + RESULT_SUFFIX)
        path.write_text("{not valid json", encoding="utf-8")
        outputs = on_load_result_file(str(tmp_path), "broken" + RESULT_SUFFIX)
        assert "red" in outputs[STATUS]
