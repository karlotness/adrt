# Copyright Karl Otness, Donsub Rim
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import argparse
import hashlib
import sys
import pathlib
import requests


CATCH2_URL = "https://github.com/catchorg/Catch2/releases/download/v2.13.10/catch.hpp"
CATCH2_SHA512 = "b7dd8acce2d32e86f5356c7b7e96a72c6aecb0529af29a7ab85b8dfb1649d510bcfe117f57691b75783ca90fd21c347f64c9cf6d4b996d686f82f081840e89cb"  # noqa: E501
CATCH2_DIGEST = bytes.fromhex(CATCH2_SHA512)


parser = argparse.ArgumentParser()
parser.add_argument("out_path", help="Path to write downloaded header")


def download_catch2():
    with requests.get(CATCH2_URL) as response:
        response.raise_for_status()
        body = response.content
    # Check the hash
    digest = hashlib.sha512(body)
    if digest.digest() != CATCH2_DIGEST:
        raise ValueError(f"Invalid hash for catch2 header. Got: {digest.hexdigest()}")
    return body


if __name__ == "__main__":
    args = parser.parse_args()
    out_path = pathlib.Path(args.out_path)
    if out_path.is_file():
        print("Catch2 already present", file=sys.stderr)
        sys.exit(0)

    # Download fresh copy
    body = download_catch2()
    if body:
        print("Downloaded Catch2", file=sys.stderr)
    else:
        # Failed to retrieve header
        print("Failed to retrieve headers", file=sys.stderr)
        sys.exit(1)

    # Write out the result
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as out_file:
        out_file.write(body)
    sys.exit(0)
