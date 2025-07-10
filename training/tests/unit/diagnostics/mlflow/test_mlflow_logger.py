from anemoi.training.diagnostics.mlflow.logger import AnemoiMLflowLogger


def test_mlflowlogger_metric_deduplication() -> None:

    logger = AnemoiMLflowLogger(experiment_name="test_exp", tracking_uri=None, offline=True)
    logger.log_metrics({"foo": 1.0}, step=5)
    logger.log_metrics({"foo": 1.0}, step=5)  # duplicate

    # Only the first metric should be logged
    assert len(logger._logged_metrics) == 1
    assert next(iter(logger._logged_metrics))[0] == "foo"  # key
    assert next(iter(logger._logged_metrics))[1] == 5  # step
