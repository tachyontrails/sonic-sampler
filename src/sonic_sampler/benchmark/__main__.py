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

import sys

import tyro

from sonic_sampler.benchmark.base import require_extra


# Worker processes start by `spawn`, which imports this module again, so the command runs only
# in the parent. The commands import the extra, so they are imported once it is known to be
# installed.

if __name__ == "__main__":

    require_extra()

    from sonic_sampler.benchmark.cli import Bench, execute

    sys.exit(execute(tyro.cli(Bench, prog="python -m sonic_sampler.benchmark")))
