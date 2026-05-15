from pathlib import Path
from jinja2 import Template, Environment, FileSystemLoader
import math

def generate_dnni_config(layer_data_list, input_data_list, length_timeseries, input_feature_size, q_feature_size, o_feature_size, output_feature_size, residual_feature_size, path_output=None):

    config = {
        "input_feature_size": input_feature_size,
        "q_feature_size": q_feature_size,
        "o_feature_size": o_feature_size,
        "output_feature_size": output_feature_size,
        "residual_feature_size": residual_feature_size,
        "layer_data_list": layer_data_list,
        "input_data_list": input_data_list,
        "length_timeseries": length_timeseries
    }

    if path_output is None:
        path_output = f"{Path(__file__).parent.absolute()}/../firmware/distributed_inference"

    jinja_environment = Environment(loader=FileSystemLoader(f'{Path(__file__).parent.absolute()}/templates'))
    mixer_config_h = jinja_environment.get_template('dnni_config.h.jinja')
    output = mixer_config_h.render(config)
    with open(f"{path_output}/dnni_config.h", 'w') as f:
        f.write(output)
