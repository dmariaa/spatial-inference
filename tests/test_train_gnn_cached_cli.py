import importlib.util
from pathlib import Path

from click.testing import CliRunner


_SCRIPT = Path(__file__).parents[1] / "scripts" / "train_gnn_cached.py"
_SPEC = importlib.util.spec_from_file_location("train_gnn_cached", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
main = _MODULE.main


def test_help_exposes_click_options() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "--train-start" in result.output
    assert "--traffic-cache" in result.output
    assert "--pin-memory / --no-pin-memory" in result.output
    assert "--progress / --no-progress" in result.output


def test_test_sensor_ids_option_is_repeatable() -> None:
    context = main.make_context(
        "train-gnn-cached",
        ["--test-sensor-ids", "10", "--test-sensor-ids", "20"],
    )

    assert context.params["test_sensor_ids"] == (10, 20)
