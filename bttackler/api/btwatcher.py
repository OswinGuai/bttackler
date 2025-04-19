import logging
import os

from bttackler.common.optuna_messenger import OptunaMessenger as BTMessenger
from bttackler.bridger.btmonitor import BTMonitor
from bttackler.common.utils import set_seed

logger = logging.getLogger(__name__)


class BTWatcher:
    def __init__(self, max_epoch, quick_calc, intermediate_default, final_default, seed=None):
        set_seed(seed, "manager", logger)
        self.seed = seed
        self.stop_trigger = 0
        self.monitor = BTMonitor(max_epoch, quick_calc, intermediate_default, final_default) 

    def collect_per_batch(self, *args):
        self.monitor.collect_in_training(*args)

    def collect_after_training(self, *args):
        self.monitor.collect_after_training(*args)
        self.monitor.calculate_after_training()

    def collect_after_validating(self, *args):
        self.monitor.collect_after_validating(*args)

    def collect_after_testing(self, *args):
        self.monitor.collect_after_testing(*args)

    def init_basic(self, *args):
        self.monitor.init_basic(*args)

    def refresh_before_epoch_start(self):
        self.monitor.refresh_before_epoch_start()

    def report_intermediate_result(self, study_name, trial_id):
        d = {}
        d1 = self.monitor.get_intermediate_dict()
        BTMessenger().write_monitor_info(study_name, trial_id, d1)
        d.update(d1)
        logger.info(" ".join(["intermediate_result_dict:", str(d)]))
        return d

    def report_final_result(self, study_name, trial_id):
        d = {}
        d1 = self.monitor.get_final_dict()
        BTMessenger().write_monitor_info(study_name, trial_id, d1)
        d.update(d1)
        logger.info(" ".join(["final_result_dict:", str(d)]))
        return d
