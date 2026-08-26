"""Tests for the spectrum-generation helpers (Plotly figures + math kernels)."""
import numpy as np
import plotly.graph_objects as go

from psi4_webui import utils


# --- broadening kernels ------------------------------------------------------

def test_lorentzian_ir_peaks_at_position():
    # Value at the peak equals the intensity; falls off away from it.
    assert utils.lorentzian_ir(1000.0, 1000.0, 1.0, width=10) == 1.0
    assert utils.lorentzian_ir(1050.0, 1000.0, 1.0, width=10) < 0.1


def test_gaussian_normalized_at_center():
    assert utils.gaussian(5.0, 5.0, fwhm=2.0) == 1.0
    assert utils.gaussian(100.0, 5.0, fwhm=2.0) < 1e-6


# --- IR spectrum -------------------------------------------------------------

def test_ir_spectrum_returns_figure():
    fig = utils.generate_ir_spectrum_interactive([1000.0, 1600.0], [50.0, 80.0])
    assert isinstance(fig, go.Figure)
    y = np.asarray(fig.data[0].y)
    assert not np.isnan(y).any()


def test_ir_spectrum_none_when_empty():
    assert utils.generate_ir_spectrum_interactive([], []) is None


# --- absorption / emission spectrum -----------------------------------------

def test_absorption_spectrum_returns_figure():
    fig = utils.generate_absorption_emission_spectrum_interactive(
        wavelengths=[250.0, 400.0], oscs=[0.5, 0.2])
    assert isinstance(fig, go.Figure)
    y = np.asarray(fig.data[0].y)
    assert not np.isnan(y).any()


def test_absorption_spectrum_none_when_empty():
    assert utils.generate_absorption_emission_spectrum_interactive(
        wavelengths=[], oscs=[]) is None


def test_absorption_spectrum_zero_oscillators_no_nan():
    """Regression: all-zero oscillator strengths must not produce a NaN spectrum."""
    fig = utils.generate_absorption_emission_spectrum_interactive(
        wavelengths=[250.0, 400.0], oscs=[0.0, 0.0])
    assert isinstance(fig, go.Figure)
    y = np.asarray(fig.data[0].y)
    assert not np.isnan(y).any()
    assert np.allclose(y, 0.0)


# --- ECD spectrum ------------------------------------------------------------

def test_ecd_spectrum_returns_figure():
    fig = utils.generate_ecd_spectrum_interactive([300.0, 250.0], [12.0, -25.0],
                                                  plot_range=(200, 400))
    assert isinstance(fig, go.Figure)
    y = np.asarray(fig.data[0].y)
    assert not np.isnan(y).any()


def test_ecd_spectrum_none_without_transitions():
    assert utils.generate_ecd_spectrum_interactive([], []) is None
    assert utils.generate_ecd_spectrum_interactive(None, None) is None


def test_ecd_spectrum_keeps_the_sign_of_rotatory_strengths():
    """The whole point of ECD: a negative R must produce a negative band.

    Taking magnitudes (as the absorption spectrum effectively does with oscillator
    strengths, which are never negative) would erase the Cotton-effect signs.
    """
    positive = np.asarray(utils.generate_ecd_spectrum_interactive(
        [300.0], [10.0], plot_range=(250, 350)).data[0].y)
    negative = np.asarray(utils.generate_ecd_spectrum_interactive(
        [300.0], [-10.0], plot_range=(250, 350)).data[0].y)

    assert positive.max() > 0.9 and negative.min() < -0.9
    assert np.allclose(positive, -negative, atol=1e-12)


def test_ecd_spectrum_of_enantiomers_are_mirror_images():
    """Two enantiomers differ only by the sign of every R, so CD_S = -CD_R."""
    wavelengths = [320.0, 280.0, 240.0]
    r_strengths = np.array([12.0, -25.0, 8.0])

    y_r = np.asarray(utils.generate_ecd_spectrum_interactive(
        wavelengths, r_strengths, plot_range=(200, 400)).data[0].y)
    y_s = np.asarray(utils.generate_ecd_spectrum_interactive(
        wavelengths, -r_strengths, plot_range=(200, 400)).data[0].y)

    assert np.allclose(y_r, -y_s, atol=1e-12)


def test_ecd_spectrum_scaled_by_largest_absolute_value():
    """Scaling must use max(|y|), not max(y), or a mostly-negative spectrum blows up."""
    y = np.asarray(utils.generate_ecd_spectrum_interactive(
        [300.0, 250.0], [-30.0, 5.0], plot_range=(200, 400)).data[0].y)
    assert np.isclose(np.max(np.abs(y)), 1.0)
    assert y.min() < -0.99  # the dominant band is the negative one


def test_ecd_spectrum_opposite_bands_partially_cancel():
    """Signed summation means adjacent opposite bands cancel, unlike absorption.

    Two nearby bands of equal and opposite R nearly annihilate, leaving the
    characteristic ECD couplet (one positive lobe, one negative) instead of the single
    tall band an absorption spectrum would show.
    """
    separate = np.asarray(utils.generate_ecd_spectrum_interactive(
        [300.0], [10.0], plot_range=(200, 400)).data[0].y)
    couplet = np.asarray(utils.generate_ecd_spectrum_interactive(
        [300.0, 305.0], [10.0, -10.0], plot_range=(200, 400)).data[0].y)

    # Compare integrals rather than amplitudes, which normalisation hides. The
    # cancellation is strong but not exact: summing over an evenly spaced *nm* grid
    # weights the two wavenumber-space Gaussians unequally.
    assert abs(couplet.sum()) < abs(separate.sum()) / 5
    # The couplet keeps lobes on both sides of zero; the single band does not.
    assert couplet.max() > 0.5 and couplet.min() < -0.5
    assert separate.min() > -1e-6


def test_ecd_broadening_is_done_in_wavenumber_space():
    """A constant width in cm^-1 is narrower in nm at short wavelengths.

    Broadening directly in nm would give both bands the same nm width; doing it in
    wavenumber space (as the physics wants) makes the high-energy band narrower.
    """
    def nm_width_at_half_max(centre):
        fig = utils.generate_ecd_spectrum_interactive(
            [centre], [10.0], points=20000, plot_range=(100, 900))
        x = np.asarray(fig.data[0].x)
        y = np.asarray(fig.data[0].y)
        above = x[y > 0.5]
        return above.max() - above.min()

    assert nm_width_at_half_max(200.0) < nm_width_at_half_max(600.0) / 2
