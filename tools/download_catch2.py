# Copyright 2023 Karl Otness, Donsub Rim
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
#
# /// script
# dependencies = [
#   "requests",
# ]
# ///


import argparse
import hashlib
import sys
import pathlib
import requests

CATCH2_VERSION = "3.15.0"
CATCH2_URLS = {
    "catch_amalgamated.cpp": (
        f"https://github.com/catchorg/Catch2/releases/download/v{CATCH2_VERSION}/catch_amalgamated.cpp",
        "f4b1fa292bc2db56b0e8e3377aa8dec77f0e69fac8c64ee01cf2fecf467685ba2b86fcc950821f11d4f753ea07efaf990c09f9026cdae9bf8b6c7c63b21fe076",
    ),
    "catch_amalgamated.hpp": (
        f"https://github.com/catchorg/Catch2/releases/download/v{CATCH2_VERSION}/catch_amalgamated.hpp",
        "8edaa6ec80a5227b68b32d0ba3935b62305b40764df0a55ea37ba06587e3fac3ef60c232f3e0210dfc622f9801ea687952fce06d0907b862c184107084568ed5",
    ),
}


parser = argparse.ArgumentParser()
parser.add_argument("out_dir", help="Directory for downloaded sources")


def download_file(name, url, sha512, session):
    with session.get(url) as response:
        response.raise_for_status()
        content = response.content
    digest = hashlib.sha512(content)
    # Check the hash
    if digest.digest() != bytes.fromhex(sha512):
        raise ValueError(f"Invalid hash for {name}. Got: {digest.hexdigest()}")
    return content


def main(args):
    out_dir = pathlib.Path(args.out_dir)
    with requests.Session() as session:
        sources = {
            name: download_file(name, url, sha512, session)
            for name, (url, sha512) in CATCH2_URLS.items()
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, result in sources.items():
        (out_dir / name).write_bytes(result)
    print("Downloaded Catch2", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))
