from __future__ import annotations

import torch

from compose_checkpoint import compose_checkpoint


def _agent_block(value: float) -> dict:
    return {
        "policy": {"weight": torch.full((2, 2), value)},
        "value": {"weight": torch.full((1, 2), value)},
        "optimizer": {"state": value},
        "state_preprocessor": {"running_mean": torch.tensor([value])},
    }


def _write_checkpoint(path, predator_value: float, prey_value: float, marker: str) -> None:
    torch.save(
        {
            "predator": _agent_block(predator_value),
            "prey": _agent_block(prey_value),
            "metadata": {"marker": marker},
        },
        path,
    )


def test_compose_checkpoint_selects_each_role_and_resets_optimizers(tmp_path) -> None:
    base = tmp_path / "base.pt"
    predator = tmp_path / "predator.pt"
    prey = tmp_path / "prey.pt"
    output = tmp_path / "composed.pt"
    _write_checkpoint(base, 1.0, 1.0, "base")
    _write_checkpoint(predator, 2.0, 20.0, "predator-source")
    _write_checkpoint(prey, 3.0, 30.0, "prey-source")

    compose_checkpoint(
        output,
        base=base,
        predator=predator,
        prey=prey,
        keep_optimizers=False,
        reset_preprocessors=False,
    )
    composed = torch.load(output, map_location="cpu", weights_only=False)

    assert torch.equal(composed["predator"]["policy"]["weight"], torch.full((2, 2), 2.0))
    assert torch.equal(composed["prey"]["policy"]["weight"], torch.full((2, 2), 30.0))
    assert "optimizer" not in composed["predator"]
    assert "optimizer" not in composed["prey"]
    assert "state_preprocessor" in composed["predator"]
    assert composed["metadata"] == {"marker": "base"}


def test_compose_checkpoint_rejects_incompatible_role_shapes(tmp_path) -> None:
    base = tmp_path / "base.pt"
    source = tmp_path / "source.pt"
    output = tmp_path / "composed.pt"
    _write_checkpoint(base, 1.0, 1.0, "base")
    _write_checkpoint(source, 2.0, 2.0, "source")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    checkpoint["predator"]["policy"]["weight"] = torch.ones(3, 3)
    torch.save(checkpoint, source)

    try:
        compose_checkpoint(
            output,
            base=base,
            predator=source,
            prey=None,
            keep_optimizers=False,
            reset_preprocessors=False,
        )
    except RuntimeError as exc:
        assert "Shape mismatch" in str(exc)
    else:
        raise AssertionError("Expected incompatible checkpoint composition to fail")
