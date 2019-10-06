import optuna
from {{cookiecutter.repo_name}}.utils import check_args_num \
    read_config, set_random_seed
from {{cookiecutter.repo_name}}.settings import cwd, optuna_db_path


def read_inp_file(filepath):
    raise NotImplementedError


def write_output(out, filepath):
    raise NotImplementedError


def get_objective(config):
    """
    more on optuna objectives:
    https://optuna.readthedocs.io/en/stable/faq.html
    """
    raise NotImplementedError


def check_descr_unique(data_descr, data_hash):
    """
    raises if database contains a row with the same data description
    but different data hash
    """
    raise NotImplementedError


def create_predictor():
    """
    Creates a predictor object using inference stages and model object
    """
    raise NotImplementedError


def measure_inference_time(predictor):
    """
    Creates a predictor object using inference stages and model object
    """
    raise NotImplementedError


if __name__ == "__main__":
    _, config_file, X_file, y_file, best_model_path, predictor_file, \
        metrics_file, study_name_file = check_args_num(8)
    set_random_seed()

    data_hash = str_hash(file_hash(X_file) + file_hash(y_file))

    config = read_config(config_file)
    study_name = str_hash(data_hash + config.get('objective_name'))

    X = read_inp_file(X_file)
    y = read_inp_file(y_file)

    objective = get_objective(config)

    sampler = optuna.samplers.TPESampler(seed=None)
    study = optuna.create_study(optuna_db_path, study_name=study_name,
                                sampler=sampler, load_if_exists=True)

    data_descr = config.get('data_descr')
    check_descr_unique(data_descr, data_hash)

    study.set_user_attr("data_description", data_descr)
    study.set_user_attr("data_hash", data_hash)

    try:
        study.optimize(objective, n_trials=config.get('n_trials'))
    except KeyboardInterrupt:
        pass

    write_output('{:6f}\n'.format(study.best_value), metrics_file)
    write_output('{}\n'.format(study_name), study_name_file)

    if (study.best_value is not None) and (objective.best_result is not None) \
            and ((objective.best_result - study.best_value)
                 < params['metric_precision']):
        write_output(objective.best_model, best_model_path)

    predictor = create_predictor()
    write_output(predictor, predictor_file)
    measure_inference_time(predictor)
