# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""URLab client namespaces (`client.scene`, `client.sim`, `client.runtime`,
`client.outliner`, `client.recording`, `client.replay`).

Each namespace concentrates the hand-written marshalling for a related
group of server ops; the common `_RpcNamespace` base provides a generic
RPC-synthesis fallback for ops the server advertises via `meta` that
have no hand-written wrapper yet.
"""
