import logging
import os
import sys

import yaml
import pickle

logger = logging.getLogger('messenger')


class OptunaMessenger:
    """
    This class is used to read and write information between different components of the system.
    It is used to store and retrieve monitor data from training and use in HPO framework like Optuna or NNI.

    File Storage:
    - location: a prefix path root, followed by the trial id.
    """
    def __init__(self):
        self.platform_trials_dir = './recording'
        if not os.path.exists(self.platform_trials_dir):
            os.makedirs(self.platform_trials_dir)
        self.enable_dict = None
        # self.step_counter = 0

    def get_file_path(self, trial_id, key):
        parent_file_path = os.path.join(self.platform_trials_dir, str(trial_id))
        if not os.path.exists(parent_file_path):
            os.makedirs(parent_file_path)
        file_path = os.path.join(parent_file_path, key)
        return file_path

    def write_json_info(self, info_dict, trial_id, key):
        dumped_info = pickle.dumps(info_dict)
        file_path = self.get_file_path(trial_id, key)
        logger.info('write info into %s for trial %s by key %s' % (file_path, trial_id, key))
        storage_file = open(file_path, 'wb')
        pickle.dump(dumped_info, storage_file)
        storage_file.close()

    def read_json_info(self, trial_id, key):
        file_path = self.get_file_path(trial_id, key)
        logger.info("read info from %s for key %s" % (file_path, key))
        if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
            return None
        storage_file = open(file_path, 'rb')
        info_dict = pickle.load(storage_file)
        info_dict = pickle.loads(info_dict)
        storage_file.close()
        return info_dict

    def read_yaml_info(self, trial_id, key):
        file_path = self.get_file_path(trial_id, key)
        if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
            file_obj = open(file_path, 'r')
            info_dict = yaml.load(file_obj, Loader=yaml.FullLoader)
            file_obj.close()
            return info_dict
        return None

    def write_monitor_info(self, trial_id, d):
        self.write_json_info(d, trial_id, key='monitor')

    def read_monitor_info(self, trial_id):
        return self.read_json_info(trial_id, key='monitor')

    def write_assessor_info(self, trial_id, d):
        self.write_json_info(d, trial_id, key='assessor')

    def read_assessor_info(self, trial_id):
        return self.read_json_info(trial_id, key='assessor')

    def read_tuner_info(self, trial_id):  # reproducer
        return self.read_json_info(trial_id, key='tuner')

    def write_tuner_info(self, trial_id, d):
        self.write_json_info(d, trial_id, key='tuner')

    def read_default_config_info(self):
        return self.read_yaml_info(key='default_config')

