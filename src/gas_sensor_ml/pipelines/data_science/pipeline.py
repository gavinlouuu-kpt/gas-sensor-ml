"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import split_data, train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["model_input_table", "params:model_options"],
                outputs=["X_train_tensor", "X_val_tensor", "X_test_tensor",
                          "y_train_tensor", "y_val_tensor", "y_test_tensor"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train_tensor", "y_train_tensor",
                        "X_val_tensor", "y_val_tensor",
                        "params:model_options"],
                outputs="lstm_model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["lstm_model", "X_test_tensor", 
                        "y_test_tensor", "params:model_options"],
                outputs=None,
                name="evaluate_model_node",
            ),
        ]
    )
