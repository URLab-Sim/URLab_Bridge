# urlab_dashboard -- unified bridge UI

One dearpygui window with a connection bar, a status pill, a tab bar
(`Debug | Cameras | Recording | Policy`), and a log panel. Replaces the
two parallel UIs that used to live alongside each other (`debug_ui.py`
for the new step-server, `dashboard.py` for the legacy ZMQ PUB/SUB
visualiser + policy runner). Both old files stay around for reference
until the rewrite is verified.

## Run it

```bash
uv run python -m urlab_dashboard
uv run python -m urlab_dashboard --host tcp://192.168.1.50 --port 5559
```

## Layout

```
ui/
├── app.py            # shell: dpg context, top connection bar, tab container, main loop
├── state.py          # AppState singleton (URLabClient, selections, render flags, policy run)
├── log.py            # log() helper writing to the "log_panel" widget
├── renderer.py       # EmbeddedRenderer (mujoco offscreen → dpg dynamic_texture)
└── tabs/
    ├── debug.py      # step mode + control surface + step/reset + articulation inspector + render
    ├── cameras.py    # streamed UE camera textures
    ├── recording.py  # recording / replay
    └── policy.py     # policy launcher (registry dropdown + step-mode constraint + launch/stop)
```

## Tab contract

Each tab module exposes:

```python
def build(parent_tab_tag: str) -> None:
    """Add the tab body widgets under the given dpg tab tag."""

def tick() -> None:
    """Per-frame work (texture writes, status refresh). Called from the
    shell's render loop on the main thread; safe to do GL work."""
```

Callbacks are module-level `on_*` functions; they read inputs from `dpg`
by tag and write back into `STATE` and the log via `log()`.

## Adding a tab

1. Create `tabs/foo.py` with `build(parent)` + `tick()` + `on_*` handlers.
2. In `app.py`, import it and mount it in the `dpg.tab_bar`:

```python
with dpg.tab(label="Foo", tag="tab_foo"):
    tab_foo.build("tab_foo")
```

3. Call `tab_foo.tick()` from the main render loop.

## Adding a policy launcher

The Policy tab reads `urlab_policy.registry.POLICIES` for the
dropdown and metadata. Each registered launcher lives in
`urlab_dashboard.launchers.<name>` and registers itself into
`tabs/policy.LAUNCHERS[<policy_key>]` at import time. The Policy tab
imports the launchers package on tab build, which triggers registration.

Drop a new file alongside `launchers/wtw.py`:

```python
# urlab_dashboard/launchers/my_policy.py
import threading, time
from typing import Optional

from ..log import log
from ..state import STATE
from ..tabs.policy import LAUNCHERS


def _my_launcher(entry: dict, step_mode_str: str, art_prefix: Optional[str]) -> threading.Thread:
    # Defer heavy imports (RoboJuDo, torch, mlc) so importing this
    # file doesn't break the UI when those packages are missing.
    from urlab_policy.policies.my_policy import MyPolicy
    from urlab_policy.native_runner import NativePolicyRunner

    client = STATE.client
    if client is None:
        raise RuntimeError("not connected")

    # Resolve articulation, build cfg + policy + runner here.
    art = client.articulations[art_prefix or next(iter(client.articulations))]
    runner = NativePolicyRunner(client=client, art=art, policy=MyPolicy(...))

    pr = STATE.policy_run
    pr.stop_flag.clear(); pr.step_count = 0; pr.last_error = ""

    def _loop():
        try:
            while not pr.stop_flag.is_set():
                runner.step()
                pr.step_count += 1
        except Exception as exc:
            pr.last_error = f"{type(exc).__name__}: {exc}"
            log(f"my_policy crashed: {pr.last_error}", error=True)

    t = threading.Thread(target=_loop, daemon=True); t.start()
    return t


LAUNCHERS["my_policy_key"] = _my_launcher
```

Then add it to the import list in `launchers/__init__.py`:

```python
from . import my_policy  # noqa: F401
```

Until a launcher is registered for a policy key, the UI shows the
metadata and points the user at the matching `scripts/run_*.py` script.

## Known follow-ups

- **Renderer setup ordering** — the mujoco GL warmup vs dearpygui's GLFW
  init still has Windows edge cases. `EmbeddedRenderer` is the place to fix it.
- **Inspector not refreshing in lock-step with the renderer** — currently
  `tab_debug.tick()` fires both per render-fps interval; decouple if needed.
- **Wire LAUNCHERS for built-in policies** — start with `go2_wtw` and
  `beyondmimic_dance` since those have working scripts to model from.
