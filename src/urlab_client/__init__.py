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

"""URLab remote-stepping client.

Public surface:

    URLabClient        -- session handshake + step / reset / set_mode transport
    URLabArticulation  -- per-articulation accessor (actuators, joints, sensors,
                          bodies, cameras, controller, xfrc, art.mj(kind, name))
    URLabController    -- kind-tagged controller config surface
    URLabPDController  -- PD-specific typed setters
    URLabEntity        -- non-articulation dynamic body accessor (and base
                          class of URLabArticulation)
    URLabCameraView    -- shared camera accessor type

Recording and replay:

    URLabRecordingAPI  -- session record / save / clear (see client.recording)
    URLabReplayAPI     -- load / set_active / play (see client.replay)

Render-farm pool:

    URLabPool          -- discover / lease editor instances (same-host registry
                          or a static multi-machine endpoint list)
    InstanceInfo       -- one discoverable instance (host / ports / capabilities)

Scene authoring:

    URLabAsset, URLabBlueprint, URLabSpawnHandle, URLabLightHandle

Errors:

    URLabRPCError, URLabPIEError, URLabVersionMismatch, URLabTimeoutError,
    URLabPuppetDriftError

Enums:

    StepMode, ControlMode, ActuatorType, ControllerKind, CameraMode,
    CameraTiming, ObservationLevel, SpaceMode

Typed result objects:

    PIEState, PIEStartResult, PIEStatus, SimOptions, ActorInfo,
    BlueprintInfo, RecordingHandle, RecordingSummary, ReplaySession,
    ReplayStatus, StepResult, CameraStreamInfo, Readiness
"""

from .articulation import (
    Actuator,
    Body,
    Joint,
    Sensor,
    URLabArticulation,
    URLabCameraView,
    URLabController,
    URLabEntity,
    URLabPDController,
)
from .client import Readiness, URLabClient
from .render_client import (
    USER_CAMERA,
    CameraFrame,
    RenderClient,
    RenderError,
    UserPose,
    poses_from_mjdata,
)
from . import viewer_sync
from .render_pool import (
    InstanceSpec,
    RenderPool,
    RenderPoolError,
    parse_endpoints,
)
from .peek import PeekViewer
from .session import OwnerInfo, discover_owners
from .enums import (
    ActuatorType,
    CameraMode,
    CameraTiming,
    ControlMode,
    ControlSource,
    ControllerKind,
    LightKind,
    ObservationLevel,
    SpaceMode,
    StepMode,
)
from .errors import (
    URLabPIEError,
    URLabPuppetDriftError,
    URLabRPCError,
    URLabTimeoutError,
    URLabVersionMismatch,
)
from .namespaces.recording import URLabRecordingAPI
from .namespaces.replay import URLabReplayAPI
from .results import (
    ActorBounds,
    ActorHierarchyNode,
    ActorInfo,
    BlueprintInfo,
    CameraPose,
    CameraStreamInfo,
    Contact,
    ContactsResult,
    StepResult,
    KeyframeInfo,
    MocapPose,
    PIEStartResult,
    PIEState,
    PIEStatus,
    RecordingHandle,
    RecordingSummary,
    ReplaySession,
    ReplayStatus,
    SceneSnapshot,
    SceneSnapshotActor,
    SimOptions,
)
from .pool import (
    InstanceInfo,
    URLabPool,
    default_registry_dir,
    pid_alive,
    read_registry,
)
from .scene_authoring import (
    URLabAsset,
    URLabBlueprint,
    URLabLightHandle,
    URLabSpawnHandle,
)
from .transports import resolve_endpoint

__all__ = [
    # client + articulation
    "URLabClient",
    "Readiness",
    # fast-path render client
    "RenderClient",
    "CameraFrame",
    "RenderError",
    "poses_from_mjdata",
    "USER_CAMERA",
    "UserPose",
    "viewer_sync",
    # multi-instance render pool
    "RenderPool",
    "InstanceSpec",
    "RenderPoolError",
    "parse_endpoints",
    # live peek viewer (attach + push)
    "PeekViewer",
    # session discovery / join
    "OwnerInfo",
    "discover_owners",
    "URLabArticulation",
    "URLabCameraView",
    "URLabController",
    "URLabEntity",
    "URLabPDController",
    "URLabRecordingAPI",
    "URLabReplayAPI",
    # per-kind handles
    "Actuator",
    "Body",
    "Joint",
    "Sensor",
    # scene authoring
    "URLabAsset",
    "URLabBlueprint",
    "URLabLightHandle",
    "URLabSpawnHandle",
    # errors
    "URLabPIEError",
    "URLabPuppetDriftError",
    "URLabRPCError",
    "URLabTimeoutError",
    "URLabVersionMismatch",
    # enums
    "ActuatorType",
    "CameraMode",
    "CameraTiming",
    "ControlMode",
    "ControlSource",
    "ControllerKind",
    "LightKind",
    "ObservationLevel",
    "SpaceMode",
    "StepMode",
    # typed results
    "ActorBounds",
    "ActorHierarchyNode",
    "ActorInfo",
    "BlueprintInfo",
    "CameraPose",
    "CameraStreamInfo",
    "Contact",
    "ContactsResult",
    "StepResult",
    "KeyframeInfo",
    "MocapPose",
    "PIEStartResult",
    "PIEState",
    "PIEStatus",
    "RecordingHandle",
    "RecordingSummary",
    "ReplaySession",
    "ReplayStatus",
    "SceneSnapshot",
    "SceneSnapshotActor",
    "SimOptions",
    # transport helpers
    "resolve_endpoint",
    # render-farm pool discovery + leasing
    "URLabPool",
    "InstanceInfo",
    "default_registry_dir",
    "pid_alive",
    "read_registry",
]
