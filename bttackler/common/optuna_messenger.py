import logging
import os
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

    def get_file_path(self, study_name, trial_id, key):
        parent_file_path = os.path.join(self.platform_trials_dir, study_name, str(trial_id))
        if not os.path.exists(parent_file_path):
            os.makedirs(parent_file_path)
        file_path = os.path.join(parent_file_path, key)
        return file_path

    def write_json_info(self, info_dict, study_name, trial_id, key):
        dumped_info = pickle.dumps(info_dict)
        file_path = self.get_file_path(study_name, trial_id, key)
        logger.info('write info into %s for trial %s by key %s' % (file_path, trial_id, key))
        storage_file = open(file_path, 'wb')
        pickle.dump(dumped_info, storage_file)
        storage_file.close()

    def read_json_info(self, study_name, trial_id, key):
        file_path = self.get_file_path(study_name, trial_id, key)
        logger.info("read info from %s for key %s" % (file_path, key))
        if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
            return None
        storage_file = open(file_path, 'rb')
        info_dict = pickle.load(storage_file)
        info_dict = pickle.loads(info_dict)
        storage_file.close()
        return info_dict

    def write_monitor_info(self, study_name, trial_id, d):
        self.write_json_info(d, study_name, trial_id, key='monitor')

    def read_monitor_info(self, study_name, trial_id):
        return self.read_json_info(study_name, trial_id, key='monitor')




