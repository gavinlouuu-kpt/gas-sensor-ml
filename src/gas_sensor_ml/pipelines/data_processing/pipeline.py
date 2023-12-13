"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import preprocess_data_bin

# def create_pipeline(**kwargs) -> Pipeline:
#     return pipeline([
#         node(
#             func=preprocess_data_bucket,
#             inputs='mox',
#             outputs='mox_bucket',
#             name='data_processing'
#         )
#     ])

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_data_bin,
            inputs=[
                'mox',
                'params:num_bins'],
            outputs='mox_bin',
            name='data_processing_node'
        )
    ])
