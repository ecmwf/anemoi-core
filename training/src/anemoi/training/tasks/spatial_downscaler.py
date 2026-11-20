# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from itertools import pairwise

from anemoi.training.diagnostics.callbacks.plot_adapter import SpatialDownscalerPlotAdapter
from anemoi.training.tasks.base import BaseSingleStepTask
from anemoi.utils.dates import as_timedelta
from anemoi.utils.dates import frequency_to_string


class SpatialDownscaler(BaseSingleStepTask):
    """Spatial downscaling: predict the fields of one dataset at the times its inputs are valid.

    The input and output offsets are the same, for example ``["0H", "6H"]`` for two snapshots; output
    step i is valid at the time of input step i. Which datasets are inputs and which are predicted is
    set by the model's encoders and decoders, as for every task. A low-resolution dataset is
    typically an input only, projected onto the grid of the predicted high-resolution dataset by its
    spatial preprocessor, and the tendency model learns the high-resolution fields as tendencies
    against it (``model.tendency_reference``).

    Parameters
    ----------
    offsets : list[str]
        Times of the snapshots, relative to the sample's reference time, as duration strings. They
        must be evenly spaced.
    """

    name: str = "spatial_downscaler"

    def __init__(self, offsets: list[str] | None = None, **_kwargs) -> None:
        offsets = sorted(as_timedelta(offset) for offset in (offsets or ["0H"]))
        if len(set(offsets)) != len(offsets):
            msg = f"The offsets of the spatial downscaler must be distinct, got {offsets}."
            raise ValueError(msg)
        spacings = {later - earlier for earlier, later in pairwise(offsets)}
        if len(spacings) > 1:
            msg = f"The offsets of the spatial downscaler must be evenly spaced, got {offsets}."
            raise ValueError(msg)
        super().__init__(input_offsets=offsets, output_offsets=offsets)
        self._plot_adapter = SpatialDownscalerPlotAdapter(self)

    def _get_timestep_for_metadata(self) -> str:
        """Zero, as for any task whose outputs are valid at the times of its inputs; see :meth:`fill_metadata`."""
        return "0H"

    def fill_metadata(self, md_dict: dict) -> None:
        """Record the offsets explicitly, and that downscaling feeds no output back into the inputs.

        Inference derives the offsets and the advance map from ``timestep`` unless the metadata states
        them. With a ``timestep`` of ``0H`` every derived offset would be zero, so the inputs of a window
        of several snapshots could not be found.

        No ``rollout_shift`` is written: with no feedback it would only say how far to jump to the next
        independent window, which training does not determine, so it belongs in the inference
        configuration.
        """
        super().fill_metadata(md_dict)

        downscaling_timesteps = {
            "input_offsets": [frequency_to_string(offset) for offset in self._input_offsets],
            "output_offsets": [frequency_to_string(offset) for offset in self._output_offsets],
            "advance_map": {"inin": [], "outin": []},
        }
        for dataset_name in md_dict["metadata_inference"]["dataset_names"]:
            md_dict["metadata_inference"][dataset_name]["timesteps"].update(downscaling_timesteps)
