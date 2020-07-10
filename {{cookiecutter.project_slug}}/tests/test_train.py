from {{cookiecutter.project_slug}}.train import write_stats, load_stats


def test_write_and_load_stats__simple(tmp_path):
    best_eval_score = 1.0
    epoch = 1
    remaining_patience = 2
    mlflow_run_id = 'NO_MLFLOW'
    write_stats(tmp_path, best_eval_score, epoch, remaining_patience)
    result = load_stats(tmp_path)
    assert (best_eval_score, epoch, remaining_patience, mlflow_run_id) == result
