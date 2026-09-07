"""Role-direction rules for the P/D switch.

Self-contained on purpose: it loads `pd_role.py` by path and hands the function
plain objects, so it runs anywhere without a vLLM runtime or a GPU.

    python3 tools/pd_role_switch/test_kv_role.py

The case that matters is kv_both. llm-d deploys BOTH its prefill and its decode
replicas with `kv_role: kv_both`, and the switch must leave it alone. Moving it
would be one-way: the budget mapping only ever answers kv_producer or
kv_consumer, so a round trip could not put kv_both back and the engine would
drift permanently away from its deployed configuration.
"""

import importlib.util
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PD_PATH = os.environ.get(
    "PD_ROLE_PATH",
    os.path.join(HERE, "..", "..", "vllm", "v1", "engine", "pd_role.py"),
)


def load_module(path):
    """Import pd_role.py without importing vLLM."""
    for name in ("vllm", "vllm.logger", "vllm.v1", "vllm.v1.engine"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["vllm.logger"].init_logger = lambda *_a, **_k: types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    spec = importlib.util.spec_from_file_location("pd_role_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Cfg:
    def __init__(self, role):
        self.kv_connector = "NixlConnector"
        self.kv_role = role


class _Core:
    def __init__(self, role):
        self.vllm_config = types.SimpleNamespace(kv_transfer_config=_Cfg(role))


def main():
    mod = load_module(PD_PATH)

    def switch(core, previous, current):
        return mod._switch_kv_role(
            core, "deepep_v2", previous_budget=previous, current_budget=current
        )

    def role_of(core):
        return core.vllm_config.kv_transfer_config.kv_role

    failures = []

    # kv_both means "both directions" already, so a switch must not touch it.
    core = _Core("kv_both")
    switch(core, 2048, 128)
    switch(core, 128, 2048)
    if role_of(core) != "kv_both":
        failures.append("kv_both was overwritten -> %r" % role_of(core))

    # ...and the ordinary direction flip must still work.
    core = _Core("kv_producer")
    switch(core, 2048, 128)
    if role_of(core) != "kv_consumer":
        failures.append("shrinking budget did not select kv_consumer -> %r" % role_of(core))
    switch(core, 128, 2048)
    if role_of(core) != "kv_producer":
        failures.append("growing budget did not restore kv_producer -> %r" % role_of(core))

    # An unchanged budget is not a role change.
    core = _Core("kv_producer")
    switch(core, 2048, 2048)
    if role_of(core) != "kv_producer":
        failures.append("equal budgets moved the role -> %r" % role_of(core))

    # No connector configured: nothing to point.
    core = _Core("kv_producer")
    core.vllm_config.kv_transfer_config = None
    if switch(core, 2048, 128) is not None:
        failures.append("a missing connector should report no role")

    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("PASS: kv_both preserved; producer/consumer flips both ways; "
          "equal budgets and a missing connector are no-ops")
    return 0


if __name__ == "__main__":
    sys.exit(main())
