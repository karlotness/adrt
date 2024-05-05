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
import asyncio
import io
import httpx


CATCH2_VERSION = "3.6.0"
CATCH2_URLS = {
    "catch_amalgamated.cpp": (
        f"https://github.com/catchorg/Catch2/releases/download/v{CATCH2_VERSION}/catch_amalgamated.cpp",
        "ed2d74239f6d8c91facfd14aaffbd2a28318cfbdf64084a98d913c79bafc369fcc2591be19b52f2963b98f12ca63ab3ac27dd339b3e0231f1a0c732711b29467",
    ),
    "catch_amalgamated.hpp": (
        f"https://github.com/catchorg/Catch2/releases/download/v{CATCH2_VERSION}/catch_amalgamated.hpp",
        "efa9494b8ddd747205df02cb91bb885efcae60bcf523c51dbc4a700746cbfa999446130b34b5fb78493e7b0607088dbb43c5fd5e67e4053a12cb294bce6add54",
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
