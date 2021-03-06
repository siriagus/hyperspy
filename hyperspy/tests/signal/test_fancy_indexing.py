# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
from nose.tools import (
    assert_true,
    assert_equal,
    raises)

from hyperspy.signal import Signal
from hyperspy import signals


class Test1D:

    def setUp(self):
        self.spectrum = Signal(np.arange(10))
        self.data = self.spectrum.data.copy()

    def test_slice_None(self):
        s = self.spectrum.isig[:]
        d = self.data
        np.testing.assert_array_equal(s.data, d)
        assert_equal(s.axes_manager._axes[0].offset,
                     self.spectrum.axes_manager._axes[0].offset)
        assert_equal(s.axes_manager._axes[0].scale,
                     self.spectrum.axes_manager._axes[0].scale)

    def test_reverse_slice(self):
        s = self.spectrum.isig[-1:1:-1]
        d = self.data[-1:1:-1]
        np.testing.assert_array_equal(s.data, d)
        assert_equal(s.axes_manager._axes[0].offset, 9)
        assert_equal(s.axes_manager._axes[0].scale,
                     self.spectrum.axes_manager._axes[0].scale * -1)

    def test_slice_out_of_axis(self):
        np.testing.assert_array_equal(
            self.spectrum.isig[-1.:].data, self.spectrum.data)
        np.testing.assert_array_equal(
            self.spectrum.isig[
                :11.].data, self.spectrum.data)

    @raises(ValueError)
    def test_step0_slice(self):
        self.spectrum.isig[::0]

    def test_index(self):
        s = self.spectrum.isig[3]
        assert_equal(s.data, 3)
        assert_equal(len(s.axes_manager._axes), 1)
        assert_equal(s.data.shape, (1,))

    def test_float_index(self):
        s = self.spectrum.isig[3.4]
        assert_equal(s.data, 3)
        assert_equal(len(s.axes_manager._axes), 1)
        assert_equal(s.data.shape, (1,))

    def test_signal_indexer_slice(self):
        s = self.spectrum.isig[1:-1]
        d = self.data[1:-1]
        np.testing.assert_array_equal(s.data, d)
        assert_equal(s.axes_manager._axes[0].offset, 1)
        assert_equal(s.axes_manager._axes[0].scale,
                     self.spectrum.axes_manager._axes[0].scale)

    def test_signal_indexer_reverse_slice(self):
        s = self.spectrum.isig[-1:1:-1]
        d = self.data[-1:1:-1]
        np.testing.assert_array_equal(s.data, d)
        assert_equal(s.axes_manager._axes[0].offset, 9)
        assert_equal(s.axes_manager._axes[0].scale,
                     self.spectrum.axes_manager._axes[0].scale * -1)

    def test_signal_indexer_step2_slice(self):
        s = self.spectrum.isig[1:-1:2]
        d = self.data[1:-1:2]
        np.testing.assert_array_equal(s.data, d)
        assert_equal(s.axes_manager._axes[0].offset, 1)
        assert_equal(np.sign(s.axes_manager._axes[0].scale),
                     np.sign(self.spectrum.axes_manager._axes[0].scale))
        assert_equal(s.axes_manager._axes[0].scale,
                     self.spectrum.axes_manager._axes[0].scale * 2.)

    def test_signal_indexer_index(self):
        s = self.spectrum.isig[3]
        assert_equal(s.data, 3)
        assert_equal(len(s.axes_manager._axes), 1)
        assert_equal(s.data.shape, (1,))

    @raises(IndexError)
    def test_navigation_indexer_navdim0(self):
        self.spectrum.inav[3]

    def test_minus_one_index(self):
        s = self.spectrum.isig[-1]
        assert_equal(s.data, self.data[-1])


class Test2D:

    def setUp(self):
        self.spectrum = Signal(np.arange(24).reshape(6, 4))
        self.spectrum.axes_manager.set_signal_dimension(2)
        self.data = self.spectrum.data.copy()

    def test_index(self):
        s = self.spectrum.isig[3, 2]
        assert_equal(s.data[0], 11)
        assert_equal(len(s.axes_manager._axes), 1)
        assert_equal(s.data.shape, (1,))

    def test_partial(self):
        s = self.spectrum.isig[3, 2:5]
        np.testing.assert_array_equal(s.data, [11, 15, 19])
        assert_equal(len(s.axes_manager._axes), 1)
        assert_equal(s.data.shape, (3,))


class Test3D_SignalDim0:

    def setUp(self):
        self.spectrum = Signal(np.arange(24).reshape((2, 3, 4)))
        self.data = self.spectrum.data.copy()
        self.spectrum.axes_manager._axes[2].navigate = True

    @raises(IndexError)
    def test_signal_indexer_signal_dim0_idx_error1(self):
        s = self.spectrum
        s.isig[:].data

    @raises(IndexError)
    def test_signal_indexer_signal_dim0_idx_error2(self):
        s = self.spectrum
        s.isig[:, :].data

    @raises(IndexError)
    def test_signal_indexer_signal_dim0_idx_error3(self):
        s = self.spectrum
        s.isig[0]

    def test_navigation_indexer_signal_dim0(self):
        s = self.spectrum
        np.testing.assert_array_equal(s.data, s.inav[:].data)


class Test3D_Navigate_0_and_1:

    def setUp(self):
        self.spectrum = Signal(np.arange(24).reshape((2, 3, 4)))
        self.data = self.spectrum.data.copy()
        self.spectrum.axes_manager._axes[0].navigate = True
        self.spectrum.axes_manager._axes[1].navigate = True
        self.spectrum.axes_manager._axes[2].navigate = False

    def test_1px_navigation_indexer_slice(self):
        s = self.spectrum.inav[1:2]
        d = self.data[:, 1:2]
        np.testing.assert_array_equal(s.data, d)
        assert_equal(s.axes_manager._axes[1].offset, 1)
        assert_equal(s.axes_manager._axes[1].size, 1)
        assert_equal(s.axes_manager._axes[1].scale,
                     self.spectrum.axes_manager._axes[1].scale)

    def test_1px_signal_indexer_slice(self):
        s = self.spectrum.isig[1:2]
        d = self.data[:, :, 1:2]
        np.testing.assert_array_equal(s.data, d)
        assert_equal(s.axes_manager.signal_axes[0].offset, 1)
        assert_equal(s.axes_manager.signal_axes[0].size, 1)
        assert_equal(s.axes_manager.signal_axes[0].scale,
                     self.spectrum.axes_manager.signal_axes[0].scale)

    def test_signal_indexer_slice_variance_signal(self):
        s1 = self.spectrum
        s1.estimate_poissonian_noise_variance()
        s1_1 = s1.isig[1:2]
        np.testing.assert_array_equal(
            s1.metadata.Signal.Noise_properties.variance.data[:, :, 1:2],
            s1_1.metadata.Signal.Noise_properties.variance.data)

    def test_navigation_indexer_slice_variance_signal(self):
        s1 = self.spectrum
        s1.estimate_poissonian_noise_variance()
        s1_1 = s1.inav[1:2]
        np.testing.assert_array_equal(
            s1.metadata.Signal.Noise_properties.variance.data[:, 1:2],
            s1_1.metadata.Signal.Noise_properties.variance.data)

    def test_signal_indexer_slice_variance_float(self):
        s1 = self.spectrum
        s1.metadata.set_item("Signal.Noise_properties.variance", 1.2)
        s1_1 = s1.isig[1:2]
        assert_equal(
            s1.metadata.Signal.Noise_properties.variance,
            s1_1.metadata.Signal.Noise_properties.variance)

    def test_navigation_indexer_slice_variance_float(self):
        s1 = self.spectrum
        s1.metadata.set_item("Signal.Noise_properties.variance", 1.2)
        s1_1 = s1.inav[1:2]
        assert_equal(
            s1.metadata.Signal.Noise_properties.variance,
            s1_1.metadata.Signal.Noise_properties.variance)

    def test_dimension_when_indexing(self):
        s = self.spectrum.inav[0]
        assert_equal(s.data.shape, self.data[:, 0, :].shape)

    def test_dimension_when_slicing(self):
        s = self.spectrum.inav[0:1]
        assert_equal(s.data.shape, self.data[:, 0:1, :].shape)


class Test3D_Navigate_1:

    def setUp(self):
        self.spectrum = Signal(np.arange(24).reshape((2, 3, 4)))
        self.data = self.spectrum.data.copy()
        self.spectrum.axes_manager._axes[0].navigate = False
        self.spectrum.axes_manager._axes[1].navigate = True
        self.spectrum.axes_manager._axes[2].navigate = False

    def test_1px_navigation_indexer_slice(self):
        s = self.spectrum.inav[1:2]
        d = self.data[:, 1:2]
        np.testing.assert_array_equal(s.data, d)
        assert_equal(s.axes_manager._axes[1].offset, 1)
        assert_equal(s.axes_manager._axes[1].size, 1)
        assert_equal(s.axes_manager._axes[1].scale,
                     self.spectrum.axes_manager._axes[1].scale)

    def test_1px_signal_indexer_slice(self):
        s = self.spectrum.isig[1:2]
        d = self.data[:, :, 1:2]
        np.testing.assert_array_equal(s.data, d)
        assert_equal(s.axes_manager.signal_axes[0].offset, 1)
        assert_equal(s.axes_manager.signal_axes[0].size, 1)
        assert_equal(s.axes_manager.signal_axes[0].scale,
                     self.spectrum.axes_manager.signal_axes[0].scale)

    def test_subclass_assignment(self):
        im = self.spectrum.as_image((-2, -1))
        assert_true(isinstance(im.isig[0], signals.Spectrum))


class TestFloatArguments:

    def setUp(self):
        self.spectrum = Signal(np.arange(10))
        self.spectrum.axes_manager[0].scale = 0.5
        self.spectrum.axes_manager[0].offset = 0.25
        self.data = self.spectrum.data.copy()

    def test_float_start(self):
        s = self.spectrum.isig[0.75:-1]
        d = self.data[1:-1]
        np.testing.assert_array_equal(s.data, d)
        assert_equal(s.axes_manager._axes[0].offset, 0.75)
        assert_equal(s.axes_manager._axes[0].scale,
                     self.spectrum.axes_manager._axes[0].scale)

    def test_float_end(self):
        s = self.spectrum.isig[1:4.75]
        d = self.data[1:-1]
        np.testing.assert_array_equal(s.data, d)
        assert_equal(s.axes_manager._axes[0].offset, 0.75)
        assert_equal(s.axes_manager._axes[0].scale,
                     self.spectrum.axes_manager._axes[0].scale)

    def test_float_both(self):
        s = self.spectrum.isig[0.75:4.75]
        d = self.data[1:-1]
        np.testing.assert_array_equal(s.data, d)
        assert_equal(s.axes_manager._axes[0].offset, 0.75)
        assert_equal(s.axes_manager._axes[0].scale,
                     self.spectrum.axes_manager._axes[0].scale)

    def test_float_step(self):
        s = self.spectrum.isig[::1.1]
        d = self.data[::2]
        np.testing.assert_array_equal(s.data, d)
        assert_equal(s.axes_manager._axes[0].offset, 0.25)
        assert_equal(s.axes_manager._axes[0].scale,
                     self.spectrum.axes_manager._axes[0].scale * 2)

    def test_negative_float_step(self):
        s = self.spectrum.isig[::-1.1]
        d = self.data[::-2]
        np.testing.assert_array_equal(s.data, d)
        assert_equal(s.axes_manager._axes[0].offset, 4.75)
        assert_equal(s.axes_manager._axes[0].scale,
                     self.spectrum.axes_manager._axes[0].scale * -2)


class TestEllipsis:

    def setUp(self):
        self.spectrum = Signal(np.arange(2 ** 5).reshape(
            (2, 2, 2, 2, 2)))
        self.data = self.spectrum.data.copy()

    def test_in_between(self):
        s = self.spectrum.inav[0, ..., 0]
        np.testing.assert_array_equal(s.data, self.data[0, ..., 0, :])

    def test_ellipsis_navigation(self):
        s = self.spectrum.inav[..., 0]
        np.testing.assert_array_equal(s.data, self.data[0, ...])

    def test_ellipsis_navigation2(self):
        self.spectrum.axes_manager._axes[-2].navigate = False
        self.spectrum.axes_manager._axes[-3].navigate = False
        s = self.spectrum.isig[..., 0]
        np.testing.assert_array_equal(s.data, self.data[:, :, 0, ...])
