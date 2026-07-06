# SPDX-FileCopyrightText: 2026-present U.N. Owen <void@some.where>
#
# SPDX-License-Identifier: MIT

import logging

from . import config, dataset, sources


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)