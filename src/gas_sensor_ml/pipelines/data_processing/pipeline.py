"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import preprocess_data_bin, create_model_input_table


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_data_bin,
            inputs=[
                'mox',
                'params:process_options'],
            outputs='mox_bin',
            name='data_processing_node'
        ),
        node(
            func=create_model_input_table,
            inputs=['mox_bin',
                'params:process_options'],
            outputs="model_input_table",
            name="create_model_input_table_node",
            ),
    ])
