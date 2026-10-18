# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026 SonicSampler Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
import triton                               # noqa.

from collections.abc import Iterator
from contextlib import contextmanager
from os import cpu_count, environ, sched_setaffinity
from typing import Final

from msgspec import Struct
from pynvml import (
    NVML_CLOCK_MEM,
    NVML_CLOCK_SM,
    nvmlDeviceGetClockInfo,
    nvmlDeviceGetCount,
    nvmlDeviceGetCpuAffinity,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMaxClockInfo,
    nvmlDeviceGetName,
    nvmlDeviceGetUUID,
    nvmlInit,
    nvmlShutdown,
    nvmlSystemGetDriverVersion,
)

from triton import cdiv                     # noqa.
from triton.runtime import driver           # noqa.

from sonic_sampler import __version__


# The CPUs per word of an NVML CPU affinity mask.
MASK_BITS: Final[int] = 64


@contextmanager
def nvml() -> Iterator[None]:

    nvmlInit()

    try:

        yield

    finally:

        nvmlShutdown()


def visible_devices() -> tuple[int, ...]:
    """
    The NVML index of every GPU on the node, as `nvidia-smi` numbers them.

    """
    with nvml():

        return tuple(range(nvmlDeviceGetCount()))


def versions() -> dict[str, str]:

    with nvml():

        driver_version = nvmlSystemGetDriverVersion()

    return (
        {
            "sonic-sampler": __version__,
            "torch": str(torch.__version__),
            "triton": triton.__version__,
            "cuda": torch.version.cuda,
            "driver": driver_version,
        }
    )


def local_cpus(mask: list[int]) -> set[int]:
    """
    The CPUs set in an NVML affinity mask: an array of 64-bit words, in which CPU `64·w + s` is
    local to the GPU when bit `s` of word `w` is set.

    Note(s):

        • Reversing `bin(value)` lists the bits least significant first, so each digit's position
          is its shift. The `0b` prefix falls past the highest set bit and never reads as "1".

    """
    return (
        {
            ((MASK_BITS * word) + shift)
            for word, value in enumerate(mask)
            for shift, bit in enumerate(reversed(bin(value))) if bit == "1"
        }
    )


def pin(index: int) -> None:
    """
    Binds this process to one GPU: CUDA sees only that GPU, as device 0, and the process runs on
    the CPUs local to it.

    Note(s):

        • CUDA reads `CUDA_VISIBLE_DEVICES` once, at its first call, so this runs before any.

        • The GPU is selected by UUID, so NVML and CUDA agree on it whatever the CUDA device
          order.

    """
    words = cdiv(cpu_count(), MASK_BITS)

    with nvml():

        handle = nvmlDeviceGetHandleByIndex(index)
        uuid, mask = nvmlDeviceGetUUID(handle), nvmlDeviceGetCpuAffinity(handle, words)

    environ["CUDA_VISIBLE_DEVICES"] = uuid

    # NVML reports an empty mask where it knows no CPU locality, which leaves affinity alone.

    if cpus := local_cpus(list(mask)):

        sched_setaffinity(0, cpus)


class Device(Struct):

    """
    The identity, limits, and clocks of one GPU, as probed from a process pinned to it.

    """

    # The NVML index of the GPU, as `nvidia-smi` numbers it.
    index: int

    # The product name.
    name: str

    # The UUID.
    uuid: str

    # The compute capability, e.g. 100 for SM100.
    arch: int

    # The streaming multiprocessor count.
    multiprocessors: int

    # The opt-in shared memory per block, in bytes.
    shared_memory: int

    # The SM clock when probed, in MHz.
    sm_clock: int

    # The memory clock when probed, in MHz.
    memory_clock: int

    # The maximum SM clock, in MHz.
    max_sm_clock: int

    @classmethod
    def probe(cls, index: int) -> Device:
        """
        Probes the GPU this process is pinned to, which CUDA therefore sees as device 0.

        """
        properties = driver.active.utils.get_device_properties(0)
        target = driver.active.get_current_target()

        with nvml():

            handle = nvmlDeviceGetHandleByIndex(index)
            name, uuid = nvmlDeviceGetName(handle), nvmlDeviceGetUUID(handle)

            sm_clock = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM)
            memory_clock = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_MEM)

            max_sm_clock = nvmlDeviceGetMaxClockInfo(handle, NVML_CLOCK_SM)

        return cls(
            index=index,
            name=name,
            uuid=uuid,
            arch=target.arch,
            multiprocessors=properties["multiprocessor_count"],
            shared_memory=properties["max_shared_mem"],
            sm_clock=sm_clock,
            memory_clock=memory_clock,
            max_sm_clock=max_sm_clock,
        )

    @property
    def model(self) -> tuple[int, int]:
        """
        The properties every GPU of one run must share: the architecture and the shared memory
        the limits are derived from.

        """
        return self.arch, self.shared_memory
