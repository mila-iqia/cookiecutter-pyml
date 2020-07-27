import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def rsync_folder(source, target):

    if not os.path.exists(target):
        os.makedirs(target)

    logger.info('rsyincing {} to {}'.format(source, target))
    subprocess.check_call(["rsync", "-avzq", source, target])
