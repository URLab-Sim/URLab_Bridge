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

"""ROS 2 rebroadcaster for URLab's ZMQ streams.

Run via:

    python -m urlab_tools.ros2_broadcaster

The implementation lives in :mod:`.broadcaster`.
"""

from .broadcaster import URLabBridge, main  # noqa: F401

__all__ = ["URLabBridge", "main"]
