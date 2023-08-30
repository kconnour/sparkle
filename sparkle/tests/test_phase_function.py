import numpy as np
import pytest

from sparkle.phase_function import henyey_greenstein


class TestHenyeyGreenstein:
    def test_asymmetry_parameter_of_0_gives_same_value_everywhere(self):
        asymmetry_parameter = 0
        scattering_angles = np.radians(np.linspace(0, 180, num=1801))

        phase_function = henyey_greenstein(asymmetry_parameter, scattering_angles)

        assert np.all(phase_function == 1 / (4 * np.pi))

    def test_function_is_normalized_to_1(self):
        # Check the integral equals 1 via Riemann summation
        asymmetry_parameter = 0
        scattering_angles = np.radians(np.linspace(0, 180, num=1801))
        step_size = np.pi / len(scattering_angles)

        phase_function = henyey_greenstein(asymmetry_parameter, scattering_angles)

        # This may be true, but why introduce the sine term?
        assert np.sum(phase_function * np.sin(scattering_angles)) * 2 * np.pi * step_size == pytest.approx(1, 0.001)

    def test_asymmetry_parameter_of_negative_1_gives_0(self):
        asymmetry_parameter = -1
        scattering_angles = np.radians(np.linspace(0, 180, num=1801))[:-1]  # Avoid a singularity at 180 degrees

        phase_function = henyey_greenstein(asymmetry_parameter, scattering_angles)

        assert np.all(phase_function == 0)

    def test_asymmetry_parameter_of_positive_1_gives_0_at_nonzero_scattering_angles(self):
        asymmetry_parameter = 1
        scattering_angles = np.radians(np.linspace(0, 180, num=1801))[1:]  # Avoid a singularity at 0 degrees

        phase_function = henyey_greenstein(asymmetry_parameter, scattering_angles)

        assert np.all(phase_function == 0)
