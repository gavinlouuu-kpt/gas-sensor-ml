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
                outputs=["X_train_tensor", "X_test_tensor", "y_train_tensor", "y_test_tensor"],
                name="split_data_node",
            ),
            # node(
            #     func=train_model,
            #     inputs=["X_train_tensor", "y_train_tensor"],
            #     outputs="regressor",
            #     name="train_model_node",
            # ),
            # node(
            #     func=evaluate_model,
            #     inputs=["regressor", "X_test", "y_test"],
            #     outputs=None,
            #     name="evaluate_model_node",
            # ),
        ]
    )
