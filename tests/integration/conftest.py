"""Test configuration before everything runs."""


from __future__ import annotations

import os

import numpy as np
import pytest
import segyio

from mdio.segy.geometry import StreamerShotGeometryType


def create_segy_mock_6d(
    fake_segy_tmp: str,
    num_samples: int,
    shots: list,
    cables: list,
    receivers_per_cable: list,
    shot_lines: list = [
        "1",
    ],
    comp_types: list = [
        "1",
    ],
    chan_header_type: StreamerShotGeometryType = StreamerShotGeometryType.A,
    index_receivers: bool = True,
) -> str:
    """Dummy 6D SEG-Y file for use in tests.

    Data will be created with:

    offset is byte location 37 - offset 4 bytes
    fldr is byte location 9 - shot 4 byte
    ep is byte location 17 - shot 4 byte
    stae is byte location 137 - cable 2 byte
    tracf is byte location 13 - channel 4 byte
    styp is byte location 133 - shot_line 2 byte
    afilf is byte location 141 - comptype 2 byte

    """
    spec = segyio.spec()
    segy_file = os.path.join(fake_segy_tmp, f"4d_type_{chan_header_type}.sgy")

    shot_count = len(shots)
    total_chan = np.sum(receivers_per_cable)
    trace_count_per_line = shot_count * total_chan
    sline_count = len(shot_lines)
    comp_trace_count = trace_count_per_line * sline_count
    comp_count = len(comp_types)
    trace_count = comp_trace_count * comp_count

    spec.format = 1
    spec.samples = range(num_samples)
    spec.tracecount = trace_count
    spec.endian = "big"

    # Calculate shot, cable, channel/receiver numbers and header values
    cable_headers = []
    channel_headers = []

    # TODO: Add strict=True and remove noqa when minimum Python is 3.10
    for cable, num_rec in zip(cables, receivers_per_cable):  # noqa: B905
        cable_headers.append(np.repeat(cable, num_rec))

        channel_headers.append(np.arange(num_rec) + 1)

    cable_headers = np.hstack(cable_headers)
    channel_headers = np.hstack(channel_headers)

    if chan_header_type == StreamerShotGeometryType.B:
        channel_headers = np.arange(total_chan) + 1

    index_receivers = True
    if chan_header_type == StreamerShotGeometryType.C:
        index_receivers = False

    shot_headers = np.hstack([np.repeat(shot, total_chan) for shot in shots])
    cable_headers = np.tile(cable_headers, shot_count)
    channel_headers = np.tile(channel_headers, shot_count)

    # Add shot lines
    shot_line_headers = np.hstack(
        [np.repeat(shot_line, trace_count_per_line) for shot_line in shot_lines]
    )

    shot_headers = np.tile(shot_headers, sline_count)
    cable_headers = np.tile(cable_headers, sline_count)
    channel_headers = np.tile(channel_headers, sline_count)

    # Add multiple components
    comptype_headers = np.hstack(
        [np.repeat(comp, comp_trace_count) for comp in comp_types]
    )

    shot_line_headers = np.tile(shot_line_headers, comp_count)
    shot_headers = np.tile(shot_headers, comp_count)
    cable_headers = np.tile(cable_headers, comp_count)
    channel_headers = np.tile(channel_headers, comp_count)

    with segyio.create(segy_file, spec) as f:
        for trc_idx in range(trace_count):
            shot = shot_headers[trc_idx]
            cable = cable_headers[trc_idx]
            channel = channel_headers[trc_idx]
            shot_line = shot_line_headers[trc_idx]
            comptype = comptype_headers[trc_idx]

            # offset is byte location 37 - offset 4 bytes
            # fldr is byte location 9 - shot 4 byte
            # ep is byte location 17 - shot 4 byte
            # stae is byte location 137 - cable 2 byte
            # tracf is byte location 13 - channel 4 byte
            # styp is byte location 133 - shot_line 2 byte
            # afilf is byte location 141 - comptype 2 byte

            if index_receivers:
                f.header[trc_idx].update(
                    offset=0,
                    fldr=shot,
                    ep=shot,
                    stae=cable,
                    tracf=channel,
                    styp=shot_line,
                    afilf=comptype,
                )
            else:
                f.header[trc_idx].update(
                    offset=0,
                    fldr=shot,
                    ep=shot,
                    stae=cable,
                    styp=shot_line,
                    afilf=comptype,
                )

            samples = np.linspace(start=shot, stop=shot + 1, num=num_samples)
            f.trace[trc_idx] = samples.astype("float32")

        f.bin.update()

    return segy_file


@pytest.fixture(scope="module")
def segy_mock_6d_as_4d_shots(fake_segy_tmp: str) -> dict[str, str]:
    """Generate mock 4D shot SEG-Y files."""
    num_samples = 25
    shots = [2, 3, 5]
    cables = [0, 101, 201, 301]
    receivers_per_cable = [1, 5, 7, 5]

    segy_paths = {}

    for type_ in ["a", "b", "c"]:
        segy_paths[type_] = create_segy_mock_6d(
            fake_segy_tmp,
            num_samples=num_samples,
            shots=shots,
            cables=cables,
            receivers_per_cable=receivers_per_cable,
            chan_header_type=type_,
        )

    return segy_paths


@pytest.fixture(scope="module")
def segy_mock_6d_as_5d_shots(fake_segy_tmp: str) -> dict[str, str]:
    """Generate mock 5D shot SEG-Y files."""
    num_samples = 25
    shots = [2, 3, 5]
    cables = [0, 101, 201, 301]
    receivers_per_cable = [1, 5, 7, 5]
    shot_lines = [1, 2, 4, 5, 99]
    segy_paths = {}

    for type_ in ["a", "b", "c"]:
        segy_paths[type_] = create_segy_mock_6d(
            fake_segy_tmp,
            num_samples=num_samples,
            shots=shots,
            cables=cables,
            receivers_per_cable=receivers_per_cable,
            chan_header_type=type_,
            shot_lines=shot_lines,
        )

    return segy_paths


@pytest.fixture(scope="module")
def segy_mock_6d_shots(fake_segy_tmp: str) -> dict[str, str]:
    """Generate mock 6D shot SEG-Y files."""
    num_samples = 25
    shots = [2, 3, 5]
    cables = [0, 101, 201, 301]
    receivers_per_cable = [1, 5, 7, 5]
    shot_lines = [1, 2, 4, 5, 99]
    comp_types = [1, 2, 3, 4]
    segy_paths = {}

    for type_ in ["a", "b", "c"]:
        segy_paths[type_] = create_segy_mock_6d(
            fake_segy_tmp,
            num_samples=num_samples,
            shots=shots,
            cables=cables,
            receivers_per_cable=receivers_per_cable,
            chan_header_type=type_,
            shot_lines=shot_lines,
            comp_types=comp_types,
        )

    return segy_paths


def create_segy_mock_4d(
    fake_segy_tmp: str,
    num_samples: int,
    shots: list,
    cables: list,
    receivers_per_cable: list,
    chan_header_type: StreamerShotGeometryType = StreamerShotGeometryType.A,
    index_receivers: bool = True,
) -> str:
    """Dummy 4D SEG-Y file for use in tests."""
    spec = segyio.spec()
    segy_file = os.path.join(fake_segy_tmp, f"4d_type_{chan_header_type}.sgy")

    shot_count = len(shots)
    total_chan = np.sum(receivers_per_cable)
    trace_count = shot_count * total_chan

    spec.format = 1
    spec.samples = range(num_samples)
    spec.tracecount = trace_count
    spec.endian = "big"

    # Calculate shot, cable, channel/receiver numbers and header values
    cable_headers = []
    channel_headers = []

    # TODO: Add strict=True and remove noqa when minimum Python is 3.10
    for cable, num_rec in zip(cables, receivers_per_cable):  # noqa: B905
        cable_headers.append(np.repeat(cable, num_rec))

        channel_headers.append(np.arange(num_rec) + 1)

    cable_headers = np.hstack(cable_headers)
    channel_headers = np.hstack(channel_headers)

    if chan_header_type == StreamerShotGeometryType.B:
        channel_headers = np.arange(total_chan) + 1

    index_receivers = True
    if chan_header_type == StreamerShotGeometryType.C:
        index_receivers = False

    shot_headers = np.hstack([np.repeat(shot, total_chan) for shot in shots])
    cable_headers = np.tile(cable_headers, shot_count)
    channel_headers = np.tile(channel_headers, shot_count)

    with segyio.create(segy_file, spec) as f:
        for trc_idx in range(trace_count):
            shot = shot_headers[trc_idx]
            cable = cable_headers[trc_idx]
            channel = channel_headers[trc_idx]

            # offset is byte location 37 - offset 4 bytes
            # fldr is byte location 9 - shot 4 byte
            # ep is byte location 17 - shot 4 byte
            # stae is byte location 137 - cable 2 byte
            # tracf is byte location 13 - channel 4 byte

            if index_receivers:
                f.header[trc_idx].update(
                    offset=0,
                    fldr=shot,
                    ep=shot,
                    stae=cable,
                    tracf=channel,
                )
            else:
                f.header[trc_idx].update(
                    offset=0,
                    fldr=shot,
                    ep=shot,
                    stae=cable,
                )

            samples = np.linspace(start=shot, stop=shot + 1, num=num_samples)
            f.trace[trc_idx] = samples.astype("float32")

        f.bin.update()

    return segy_file


@pytest.fixture(scope="module")
def segy_mock_4d_shots(fake_segy_tmp: str) -> dict[StreamerShotGeometryType, str]:
    """Generate mock 4D shot SEG-Y files."""
    num_samples = 25
    shots = [2, 3, 5]
    cables = [0, 101, 201, 301]
    receivers_per_cable = [1, 5, 7, 5]

    segy_paths = {}

    for chan_header_type in StreamerShotGeometryType:
        segy_paths[chan_header_type] = create_segy_mock_4d(
            fake_segy_tmp,
            num_samples=num_samples,
            shots=shots,
            cables=cables,
            receivers_per_cable=receivers_per_cable,
            chan_header_type=chan_header_type,
        )

    return segy_paths
