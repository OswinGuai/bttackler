from __future__ import annotations

import numpy as np

import optuna
from optuna._experimental import experimental_class
from optuna.pruners import BasePruner
from optuna.study._study_direction import StudyDirection

from bttackler.bridger.btassessor import MyAssessor

@experimental_class("2.8.0")
class BTTPruner(BasePruner):

    def __init__(
        self, max_epoch, quick_calc, symptom_name_list, cmp_percent, min_cmp_num, diagnose, seed
    ) -> None:

        self.etr = MyAssessor(max_epoch, quick_calc, symptom_name_list, cmp_percent, min_cmp_num, diagnose, seed)

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        early_stop = self.etr.assess_trial(trial.number, study.study_name)
        return early_stop

