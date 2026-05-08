# This schema defines the expected structure of the metadata dictionary produced by the trainer.
# We only check a partial schema here, since we are in the process of consolidating the metadata structure and content.
# The goal is to prevent regressions in the metadata structure we are establishing for anemoi-inference
# while allowing flexibility in the rest of the metadata content and structure as we iterate towards a complete schema.
# After the consolidation is complete, we can migrate to a complete schema and potentially use pydantic for validation.
# Before making changes to the partial schema below, check whether the change is compatible with anemoi-inference.

PARTIAL_METADATA_SCHEMA = {
    "version": None,
    "config": None,
    "seed": None,
    "run_id": None,
    "dataset": None,
    "data_indices": None,
    "provenance_training": None,
    "timestamp": None,
    "metadata_inference": {
        "seed": None,
        "run_id": None,
        "dataset_names": list,  # list of datasets
        "task": None,
        "__datasets__": {  # schema applied to each dataset entry
            "timesteps": {
                "relative_date_indices_training": None,
                "input_relative_date_indices": None,
                "output_relative_date_indices": None,
                "timestep": None,
            },
            "data_indices": {
                "input": None,
                "output": None,
            },
            "variable_types": {
                "forcing": None,
                "target": None,
                "prognostic": None,
                "diagnostic": None,
            },
            "shapes": {
                "variables": None,
                "input_timesteps": None,
                "ensemble": None,
                "grid": None,
            },
        },
    },
}
