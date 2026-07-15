
def test_fixed_offsets_expose_local_reference_positions() -> None:
    task = FixedOffsetsTask(input_offsets=["6h", "-6h", "0h"], output_offsets=["6h", "0h"])

    assert task.get_batch_reference_input_indices(strict=True) == [2, 1]
    assert task.get_input_reference_positions(strict=True) == [0, 2]
