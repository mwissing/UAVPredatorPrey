from __future__ import annotations

from isaac_startup import exclude_isaac_telemetry


def test_excludes_telemetry_from_empty_kit_args() -> None:
    assert exclude_isaac_telemetry("") == (
        "--/app/extensions/excluded/0=omni.kit.telemetry"
    )


def test_preserves_other_kit_args_and_uses_free_exclusion_index() -> None:
    kit_args = (
        "--/app/extensions/excluded/0=omni.kit.splash "
        "--/renderer/multiGpu/enabled=false"
    )

    assert exclude_isaac_telemetry(kit_args) == (
        f"{kit_args} --/app/extensions/excluded/1=omni.kit.telemetry"
    )


def test_existing_telemetry_exclusion_is_idempotent() -> None:
    kit_args = (
        "--/app/extensions/excluded/0=omni.kit.telemetry "
        "--/app/extensions/excluded/1=omni.kit.splash"
    )

    assert exclude_isaac_telemetry(kit_args) == kit_args
