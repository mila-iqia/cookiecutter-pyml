import argparse
from typing import List

from omegaconf import OmegaConf


def load_configs(configs: List, cli_config_params: List):
    """Load multiple config files together with the CLI parameters.

    :param configs: list of config file in reversed order of priority.
                    E.g.,[A, B, C] C takes precedence over A, B, and B takes precedence over A.
    :param cli_config_params:
    :return:
    """
    parsed_configs = [OmegaConf.load(config) for config in configs]
    merge = OmegaConf.merge(*parsed_configs, OmegaConf.from_dotlist(cli_config_params))
    return merge


def save_hparams(hparams: dict, output_file: str):
    """Saves the hyper-parameters to a file.

    :param hparams: dict containing the hyper-prameters to save
    :param output_file: output file path
    :return:
    """
    OmegaConf.save(config=hparams, f=output_file)


def add_config_file_params_to_argparser(parser):
    """Add the parser options to deal with multiple config files and CLI config."""
    parser.add_argument('--config', nargs='*', default=[],
                        help='config files with generic hyper-parameters,  such as optimizer, '
                             'batch_size, ... -  in yaml format. Can be zero, one or more than '
                             'one file. If multiple configs are passed, the latter files will '
                             'take precedence.')
    parser.add_argument('--cli-config-params', nargs='*', default=[], type=str,
                        help='additional parameters for the config. The format of a parameter is '
                             '"architecture.hidden_size=512, which would nest the "hidden_size=512"'
                             ' under the "architecture" key. A full examples of usage is '
                             '"--cli-config-params architecture..hidden_size=512 log.file=log.txt".'
                             'These params take precedence over the ones in the config files.')


def main():
    """Main that will load various config file plus CLI parameters, and save the merged result."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--merged-config-file', help='will write the merged '
                                                     'config file here', required=True)
    add_config_file_params_to_argparser(parser)
    args = parser.parse_args()
    hyper_params = load_configs(args.config, args.cli_config_params)
    save_hparams(hyper_params, args.merged_config_file)


if __name__ == '__main__':
    main()
