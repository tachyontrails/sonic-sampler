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
    NVML_FI_DEV_CLOCKS_EVENT_REASON_HW_POWER_BRAKE_SLOWDOWN,
    NVML_FI_DEV_CLOCKS_EVENT_REASON_HW_THERM_SLOWDOWN,
    NVML_FI_DEV_CLOCKS_EVENT_REASON_SW_POWER_CAP,
    NVML_FI_DEV_CLOCKS_EVENT_REASON_SW_THERM_SLOWDOWN,
    NVML_FI_DEV_MEMORY_TEMP,
    NVML_SUCCESS,
    NVML_TEMPERATURE_GPU,
    NVML_VALUE_TYPE_DOUBLE,
    NVML_VALUE_TYPE_SIGNED_INT,
    NVML_VALUE_TYPE_SIGNED_LONG_LONG,
    NVML_VALUE_TYPE_UNSIGNED_INT,
    NVML_VALUE_TYPE_UNSIGNED_LONG,
    NVML_VALUE_TYPE_UNSIGNED_LONG_LONG,
    NVML_VALUE_TYPE_UNSIGNED_SHORT,
    c_nvmlFieldValue_t,
    nvmlDeviceGetClockInfo,
    nvmlDeviceGetCount,
    nvmlDeviceGetCpuAffinity,
    nvmlDeviceGetEnforcedPowerLimit,
    nvmlDeviceGetFieldValues,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMaxClockInfo,
    nvmlDeviceGetName,
    nvmlDeviceGetPowerManagementDefaultLimit,
    nvmlDeviceGetPowerUsage,
    nvmlDeviceGetTemperature,
    nvmlDeviceGetTotalEnergyConsumption,
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

# The milliwatts per watt, the unit of NVML power readings.
MILLIWATTS: Final[int] = 1_000

# The member of an NVML field value that holds each value type.
MEMBERS: Final[dict[int, str]] = {
    NVML_VALUE_TYPE_DOUBLE: "dVal",
    NVML_VALUE_TYPE_UNSIGNED_INT: "uiVal",
    NVML_VALUE_TYPE_UNSIGNED_LONG: "ulVal",
    NVML_VALUE_TYPE_UNSIGNED_LONG_LONG: "ullVal",
    NVML_VALUE_TYPE_SIGNED_LONG_LONG: "sllVal",
    NVML_VALUE_TYPE_SIGNED_INT: "siVal",
    NVML_VALUE_TYPE_UNSIGNED_SHORT: "usVal",
}

# The NVML fields a telemetry reading takes, by the `Telemetry` field each one fills.
FIELDS: Final[dict[str, int]] = {
    "memory_temp": NVML_FI_DEV_MEMORY_TEMP,
    "power_capping_ns": NVML_FI_DEV_CLOCKS_EVENT_REASON_SW_POWER_CAP,
    "thermal_slowdown_ns": NVML_FI_DEV_CLOCKS_EVENT_REASON_SW_THERM_SLOWDOWN,
    "hw_thermal_slowdown_ns": NVML_FI_DEV_CLOCKS_EVENT_REASON_HW_THERM_SLOWDOWN,
    "power_braking_ns": NVML_FI_DEV_CLOCKS_EVENT_REASON_HW_POWER_BRAKE_SLOWDOWN,
}


@contextmanager
def nvml() -> Iterator[None]:

    nvmlInit()

    try:

        yield

    finally:

        nvmlShutdown()


def field_value(field: c_nvmlFieldValue_t) -> int | float | None:
    """
    The value of one NVML field, or none when the GPU does not report it.

    """
    if field.nvmlReturn != NVML_SUCCESS:

        return None

    else:

        return getattr(field.value, MEMBERS[field.valueType])


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

    # The enforced power limit, in W.
    power_limit: float

    # The default power limit, in W.
    default_power_limit: float

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

            power_limit = nvmlDeviceGetEnforcedPowerLimit(handle) / MILLIWATTS
            default_power_limit = nvmlDeviceGetPowerManagementDefaultLimit(handle) / MILLIWATTS

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
            power_limit=power_limit,
            default_power_limit=default_power_limit,
        )

    @property
    def model(self) -> tuple[int, int]:
        """
        The properties every GPU of one run must share: the architecture and the shared memory
        the limits are derived from.

        """
        return self.arch, self.shared_memory


class Telemetry(Struct, kw_only=True, omit_defaults=True):

    """
    The state of one GPU at one instant: its SM clock, temperatures, power, and the time its
    clocks have been held back, by reason.

    Note(s):

        • The energy and the held-back times accumulate from when the driver loaded, so only the
          difference between two readings is meaningful.

        • A field the GPU does not report is left out.

    """

    # The SM clock, in MHz.
    sm_clock: int

    # The GPU temperature, in °C.
    gpu_temp: int

    # The power draw averaged over the last second, in W.
    power: float

    # The energy consumed, in mJ.
    energy: int

    # The memory temperature, in °C.
    memory_temp: int | None = None

    # The time the clocks were held back to stay within the power limit, in ns.
    power_capping_ns: int | None = None

    # The time the clocks were held back to stay within the temperature limits, in ns.
    thermal_slowdown_ns: int | None = None

    # The time the clocks were at least halved by an over-temperature, in ns.
    hw_thermal_slowdown_ns: int | None = None

    # The time the clocks were at least halved by an external power brake, in ns.
    power_braking_ns: int | None = None

    @classmethod
    def probe(cls, index: int) -> Telemetry:

        with nvml():

            handle = nvmlDeviceGetHandleByIndex(index)
            values = nvmlDeviceGetFieldValues(handle, list(FIELDS.values()))

            return cls(
                sm_clock=nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM),
                gpu_temp=nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU),
                power=nvmlDeviceGetPowerUsage(handle) / MILLIWATTS,
                energy=nvmlDeviceGetTotalEnergyConsumption(handle),
                **dict(zip(FIELDS, map(field_value, values))),
            )
