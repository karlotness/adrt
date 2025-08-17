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
#   "httpx[http2]",
# ]
# ///


import argparse
import hashlib
import sys
import pathlib
import asyncio
import io
import httpx


CATCH2_VERSION = "3.9.1"
CATCH2_URLS = {
    "catch_amalgamated.cpp": (
        f"https://github.com/catchorg/Catch2/releases/download/v{CATCH2_VERSION}/catch_amalgamated.cpp",
        "8343102239871ff697f02c5898238fa1d6e4476e03078cc07bcae57ad295a1e765f6a2c9ab5e9d794333550528de10df7c5d82f125e0e800efa36541c58c683f",
    ),
    "catch_amalgamated.hpp": (
        f"https://github.com/catchorg/Catch2/releases/download/v{CATCH2_VERSION}/catch_amalgamated.hpp",
        "ca02443d6134ab9bfbe107f8f0cba539f54a94f16b2157ee394fe3251950c3a5a8ef3d37e79f89f6e98ce8b9bc0781f5de5ad25cc6a1ee6bd82b4818c229391b",
    ),
}


parser = argparse.ArgumentParser()
parser.add_argument("out_dir", help="Directory for downloaded sources")


async def download_file(name, url, sha512, client):
    digest = hashlib.sha512()
    with io.BytesIO() as buffer:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                digest.update(chunk)
                buffer.write(chunk)
        # Check the hash
        if digest.digest() != bytes.fromhex(sha512):
            raise ValueError(f"Invalid hash for {name}. Got: {digest.hexdigest()}")
        return buffer.getvalue()


async def main(args):
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(
        http1=True, http2=True, follow_redirects=True
    ) as client:
        sources = {
            name: asyncio.create_task(download_file(name, url, sha512, client))
            for name, (url, sha512) in CATCH2_URLS.items()
        }
        for name, result in sources.items():
            (out_dir / name).write_bytes(await result)
    print("Downloaded Catch2", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parser.parse_args())))
