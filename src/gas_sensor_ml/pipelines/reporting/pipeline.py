"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import split_data, study_model


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance = pipeline(
        [
            node(
                func=split_data,
                inputs=["model_input_table", "params:model_options"],
                outputs=["X_train_tensor", "X_val_tensor", "X_test_tensor",
                          "y_train_tensor", "y_val_tensor", "y_test_tensor"],
                name="split_data_node",
            ),
            node(
                func=study_model,
                inputs=["X_train_tensor", "y_train_tensor",
                        "X_val_tensor", "y_val_tensor",
                        "params:model_options","params:study_options"],
                outputs="trial",
                name="study_model_node",
            ),
        ]
    )
    ds_pipeline_1 = pipeline(
        pipe=pipeline_instance,
        inputs="model_input_table",
        namespace="complete_lstm_model",
    )
    ds_pipeline_2 = pipeline(
        pipe=pipeline_instance,
        inputs="model_input_table",
        namespace="trimmed_lstm_model",
    )

    return ds_pipeline_1 + ds_pipeline_2

