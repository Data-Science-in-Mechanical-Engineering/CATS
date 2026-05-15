from pathlib import Path
from jinja2 import Template, Environment, FileSystemLoader

def generate_model_c(num_layers, message_sizes,  attention_only: bool = False, path_output=None):
    """
    Generates the model.c file for the DNNI firmware based on the provided model and message sizes.
    The model.c file contains the initialization and execution code for the model.
    Weights are in separate files.

    Args:
        model (TimeseriesPatchedDecoder): The model to be exported.
        message_sizes (dict): A dictionary containing the message sizes for each layer. (if attention_only is True, the message sizes for only the attention block should be )
        attention_only (bool): If True, only export the attention block for benchmarking.
    """
    init_model = ""
    for k in message_sizes:
        init_model += f"init_message_assignment(&message_assignment_{k});\n"

    if not attention_only:
        run_model = "residual_block(&residual_block_config_input_residual, input, output1, 1);\n"
        # run_model += "print_matrix(output1, &output_dims);\n"
        # run_model += f'printf("in\\n\\n");\n'

        adding_pos_enc = """
            for (uint32_t i = 0; i < residual_block_config_input_residual.layers[residual_block_config_input_residual.num_layers-1]->input_dims->n * residual_block_config_input_residual.layers[residual_block_config_input_residual.num_layers-1]->weight_dims->c; i++) {
            int32_t temp = (int32_t) (roundf((output1[i] * input_scaling_pos_enc + pos_enc_pos_enc[i] * pos_enc_scaling_pos_enc) * sum_scaling_pos_enc));
            output1[i] = (int8_t)(temp > INT8_MAX ? INT8_MAX : (temp < INT8_MIN ? INT8_MIN : temp));
        }"""

        run_model += adding_pos_enc + "\n"

        for i in range(num_layers):
            run_model += f"multi_head_attention_block(&mha_config_attention_block_{i}, output1, output2);\n"
            # run_model += "print_matrix(output2, &output_dims);\n"
            # run_model += f'printf("att {i}\\n\\n");\n'
            run_model += f"residual_block(&residual_block_config_residual_block_{i}, output2, output1, 0);\n"
            # run_model += "print_matrix(output1, &output_dims);\n"
            # run_model += f'printf("res {i}\\n\\n");\n'

        run_model += f"residual_block(&residual_block_config_output_residual, output1, output2, 0);\n"

        run_model += "cmsis_nn_dims output_dims2 = {.n=residual_block_config_output_residual.layers[residual_block_config_output_residual.num_layers-1]->input_dims->n, .h=1, .w=1, .c=residual_block_config_output_residual.layers[residual_block_config_output_residual.num_layers-1]->weight_dims->c};\n"
        run_model += 'printf("Result:\\n");\n'
        run_model += 'for (uint32_t i = 0; i < 10; i++) { printf("%d, ", output1 [i]);\n printf("\\n");}\n'
        run_model += "print_matrix(output2, &output_dims2);\n"
        # run_model += "print_matrix(output2, &output_dims2);\n"
        run_model += 'printf("--------------------------\\n");\n'
    else:
        # only export the first attention block (we do this for the benchmarking)
        run_model = ""
        run_model += f"residual_block(&residual_block_config_residual_block_{0}, output2, output1, 1);\n"
        run_model += f"multi_head_attention_block(&mha_config_attention_block_{0}, output1, output2);\n"

    # write to file
    config = {
        "init_model": init_model,
        "run_model": run_model,
    }

    if path_output is None:
        path_output = f"{Path(__file__).parent.absolute()}/../firmware/distributed_inference"

    jinja_environment = Environment(loader=FileSystemLoader(f'{Path(__file__).parent.absolute()}/templates'))
    mixer_config_h = jinja_environment.get_template('model.c.jinja')
    output = mixer_config_h.render(config)
    with open(f"{path_output}/model.c", 'w') as f:
        f.write(output)