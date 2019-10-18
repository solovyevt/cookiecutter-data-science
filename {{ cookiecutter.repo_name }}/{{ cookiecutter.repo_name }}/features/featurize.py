from {{cookiecutter.repo_name}}.utils import check_args_num, \
    read_config, set_random_seed
from {{cookiecutter.repo_name}}.inference import InferenceStage


def read_inp_file(filepath):
    raise NotImplementedError


def write_output(out, filepath):
    raise NotImplementedError


def apply_tfms(dataset, tfm_list):
    raise NotImplementedError


if __name__ == "__main__":
    _, config_file, *input_files, output_file, inference_file = \
        check_args_num(5, strict=False)
    set_random_seed()

    inp = [read_inp_file(inp_file) for inp_file in input_files]
    config = read_config(config_file)
    tfm_list = config.get('tfm_list')
    out = apply_tfms(inp, tfm_list)

    write_output(out, output_file)

    inference_stage = InferenceStage(input_list=input_files,
                                     params=config, transformers=tfm_list)
    inference_stage.save(inference_file)
