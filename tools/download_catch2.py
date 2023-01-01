# Copyright (c) 2023 Karl Otness, Donsub Rim
# All rights reserved
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
CATCH2_SHA256 = "3725c0f0a75f376a5005dde31ead0feb8f7da7507644c201b814443de8355170"
CATCH2_DIGEST = bytes.fromhex(CATCH2_SHA256)


parser = argparse.ArgumentParser()
parser.add_argument("out_path", help="Path to write downloaded header")
parser.add_argument(
    "--cache_dir", default=None, help="Directory to check for cached files"
)


def download_catch2():
    with requests.get(CATCH2_URL) as response:
        response.raise_for_status()
        body = response.content
    # Check the hash
    digest = hashlib.sha256(body)
    if digest.digest() != CATCH2_DIGEST:
        raise ValueError(f"Invalid hash for catch2 header. Got: {digest.hexdigest()}")
    print("Downloaded Catch2", file=sys.stderr)
    return body


def check_cache_dir(cache_dir):
    if not cache_dir:
        print("No cache specified", file=sys.stderr)
        return None
    cache_path = pathlib.Path(cache_dir)
    if not cache_path.is_dir():
        print(f"Cache directory does not exist: {cache_dir}", file=sys.stderr)
        return None
    digest_prefix = CATCH2_SHA256[:15].lower()
    candidate_path = cache_path / f"catch-{digest_prefix}.hpp"
    if candidate_path.is_file():
        with open(candidate_path, "rb") as candidate_file:
            body = candidate_file.read()
            digest = hashlib.sha256(body)
            if digest.digest() == CATCH2_DIGEST:
                print(f"Found cached headers: {candidate_path}", file=sys.stderr)
                return body
    print("Header not found in cache", file=sys.stderr)
    return None


def store_cache(cache_dir, body):
    if not cache_dir:
        return None
    digest_prefix = hashlib.sha256(body).hexdigest()[:15].lower()
    file_name = f"catch-{digest_prefix}.hpp"
    cache_path = pathlib.Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    with open(cache_path / file_name, "wb") as out_file:
        out_file.write(body)
        print(f"Cached headers as {file_name}", file=sys.stderr)


if __name__ == "__main__":
    args = parser.parse_args()
    out_path = pathlib.Path(args.out_path)
    if out_path.is_file():
        print("Catch2 already present", file=sys.stderr)
        sys.exit(0)

    # Find the contents
    body = None
    from_cache = False
    if args.cache_dir:
        # Check in cache directory
        body = check_cache_dir(args.cache_dir)
        from_cache = bool(body)
    if not body:
        # Fall back to a fresh download
        body = download_catch2()
        from_cache = False
    if not body:
        # Failed to retrieve header
        print("Failed to retrieve headers", file=sys.stderr)
        sys.exit(1)
    if args.cache_dir and not from_cache:
        # Retrieved header but not from cache, store it
        store_cache(args.cache_dir, body)

    # Write out the result
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as out_file:
        out_file.write(body)
    sys.exit(0)
