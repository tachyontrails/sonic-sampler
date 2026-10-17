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

from functools import partial
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

from msgspec import Struct, toml

from torch import Tensor
from torch.distributions import StudentT


BF16: torch.dtype = torch.bfloat16
F32: torch.dtype = torch.float32


class LogitsConfig(Struct):

    """
    The fitted parameters of the synthetic logits distribution.

    """

    # The degrees of freedom of the Student-t distribution.
    dof: float

    # The location of the Student-t distribution.
    mean: float

    # The scale of the Student-t distribution.
    scale: float

    # The exponent of the Zipfian rank profile.
    alpha: float

    # The fitted standard deviation, recorded for reference.
    sigma: float

    # The fitted excess kurtosis, recorded for reference.
    excess: float

    # The lower clamp of the sampled logits.
    lower: float

    # The upper clamp of the sampled logits.
    upper: float

    @classmethod
    def load(cls, path: Path | Traversable) -> LogitsConfig:

        return toml.decode(path.read_bytes(), type=cls)

    def save(self, path: Path | str) -> None:

        (
            Path(path)
            .resolve()
            .write_bytes(toml.encode(self))
        )


class LogitsFactory(Struct):

    """
    Samples synthetic bf16 logits on one device.

    """

    # The device the logits are sampled on.
    device: torch.device

    # The Student-t distribution of the raw logits.
    distribution: StudentT

    # The location the Zipfian profile is re-centred on.
    mu: float

    # The exponent of the Zipfian rank profile.
    alpha: float

    # The default lower clamp of the sampled logits.
    lower: float

    # The default upper clamp of the sampled logits.
    upper: float

    @classmethod
    def from_config(cls, config: LogitsConfig, device: torch.device) -> LogitsFactory:

        tensor = partial(torch.tensor, dtype=F32, device=device)
        dof, mean, scale = map(tensor, (config.dof, config.mean, config.scale))

        return cls(
            device=device,
            distribution=StudentT(df=dof, loc=mean, scale=scale),
            mu=config.mean,
            alpha=config.alpha,
            lower=config.lower,
            upper=config.upper,
        )

    @classmethod
    def default(cls, device: torch.device) -> LogitsFactory:

        path = (
            resources.files("sonic_sampler")
            / "resources/benchmark"
            / "logits.toml"
        )

        return cls.from_config(LogitsConfig.load(path), device)

    def zipfy(self, logits: Tensor) -> Tensor:

        with torch.device(logits.device):

            weights = (
                torch.arange(1, logits.size(1) + 1, dtype=F32)
                .pow(-self.alpha)
            )

            targets = (
                weights.div_(weights.sum())
                .expand_as(logits)
            )

            indices = (
                torch.softmax(logits, dim=-1, dtype=F32)
                .argsort(dim=-1, descending=True)
            )

            zipfian = (
                torch.zeros_like(targets)
                .scatter_(-1, indices, targets)
                .add_(1e-20)
                .log_()
            )

            delta = self.mu - zipfian.mean()

            return zipfian.add_(delta)

    def sample(
        self,
        rows: int,
        cols: int,
        *,
        seed: int = 0,
        minima: float | None = None,
        maxima: float | None = None,
        zipfian: bool = True,
        noise: float = 1.2,
    ) -> Tensor:

        lower = minima if minima is not None else self.lower
        upper = maxima if maxima is not None else self.upper

        # The draw is seeded in a forked RNG state, leaving the caller's generator(s) untouched.

        with torch.random.fork_rng(devices=[self.device]):

            torch.manual_seed(seed)

            logits = (
                self.distribution.sample((rows, cols))
                .clamp_(min=lower, max=upper)
            )

            if zipfian:

                logits = self.zipfy(logits)

            if noise:

                logits += torch.randn_like(logits).mul_(noise)

        return logits.to(dtype=BF16)
