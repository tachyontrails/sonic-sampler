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

import tyro

from sonic_sampler.benchmark.base import require_extra


def main() -> int:
    """
    The `sonic-sampler` entry point, with one command group per tool.

    """
    require_extra()

    # The commands import the extra, so they are imported once it is known to be installed.

    from sonic_sampler.benchmark.cli import Bench, execute

    command = tyro.extras.subcommand_cli_from_dict(dict(bench=Bench), prog="sonic-sampler")

    return execute(command)
