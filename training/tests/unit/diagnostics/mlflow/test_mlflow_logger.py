import pytest

from anemoi.training.diagnostics.mlflow.logger import AnemoiMLflowLogger


@pytest.fixture
def default_logger() -> AnemoiMLflowLogger:
    return AnemoiMLflowLogger(
        experiment_name="test_experiment",
        run_name="test_run",
        offline=True,
        tracking_uri=None,
        authentication=False,
    )


def test_mlflowlogger_metric_deduplication(default_logger: AnemoiMLflowLogger) -> None:

    default_logger.log_metrics({"foo": 1.0}, step=5)
    default_logger.log_metrics({"foo": 1.0}, step=5)  # duplicate

    # Only the first metric should be logged
    assert len(default_logger._logged_metrics) == 1
    assert next(iter(default_logger._logged_metrics))[0] == "foo"  # key
    assert next(iter(default_logger._logged_metrics))[1] == 5  # step
