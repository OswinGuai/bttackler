import logging

import numpy as np
from bttackler.common.utils import set_seed, diagnose_params
from bttackler.common.optuna_messenger import OptunaMessenger as BTMessenger

logger = logging.getLogger(__name__)


class MyAssessor:
    def __init__(self, max_epoch, quick_calc, symptom_name_list, cmp_percent, min_cmp_num, diagnose, seed=None):
        set_seed(seed, "assessor", logger)
        self.max_epoch = max_epoch
        self.quick_calc = quick_calc
        self.symptom_name_list = symptom_name_list
        logger.info("bttackler assessor using %s" % ','.join(self.symptom_name_list))
        self.cmp_percent = cmp_percent
        self.min_cmp_num = min_cmp_num
        self.dp = diagnose_params(**diagnose)

        self.epoch_train_loss_list_dict = {idx: [] for idx in range(self.max_epoch)}
        self.epoch_vg_metric_list_dict = {idx: [] for idx in range(self.max_epoch)}
        self.epoch_dr_metric_list_dict = {idx: [] for idx in range(self.max_epoch)}
        self.protect_id_set = set()
        self.half1_cmp_step_list = np.arange(self.max_epoch // 4, self.max_epoch // 2)  # del 0 ???????????self.max_epoch // 4

        self.trial_id = None
        self.result_dict_list = None
        self.result_dict = None
        self.epoch_idx = None

        self.info_dict = None

        self.module_metric_2da = None
        self.metric_prefix_list = None
        self.metric_suffix_list = None
        self.weight_grad_rate0_1da = None
        self.weight_grad_abs_avg_1da = None
        self.param_name_list = None
        self.has_nan_inf_list = None

        self.has_nan_inf = None

    def assess_trial(self, study_name, trial_id):
        self.trial_id = trial_id
        self.has_nan_inf = False

        self.info_dict = self.get_default_info_dict()
        self.receive_monitor_result(study_name, trial_id)

        self.diagnose_symptom()
        
        return self.assess_trial_end()

    def get_default_info_dict(self):
        d = {}
        for symptom_name in self.symptom_name_list:
            d[symptom_name] = None
        d["early_stop"] = False
        return d

    def get_metric(self, metric_name):
        if metric_name == "train_loss":
            return self.result_dict["train_loss"] if type(self.result_dict["train_loss"]) is float else None
        if metric_name == "median_grad_abs_avg":
            lst = self.weight_grad_abs_avg_1da
            return np.median(lst) if type(lst[0]) is np.float64 else None
        if metric_name == "median_grad_rate0":
            lst = self.weight_grad_rate0_1da
            return np.median(lst) if type(lst[0]) is np.float64 else None
        return 0

    def record_global_metric(self):
        # train_loss
        train_loss = self.get_metric("train_loss")
        if train_loss is not None:
            self.epoch_train_loss_list_dict[self.epoch_idx].append(train_loss)
        # median_grad_abs_avg
        median_grad_abs_avg = self.get_metric("median_grad_abs_avg")
        if median_grad_abs_avg is not None:
            self.epoch_vg_metric_list_dict[self.epoch_idx].append(median_grad_abs_avg)
        # median_grad_rate0
        median_grad_rate0 = self.get_metric("median_grad_rate0")
        if median_grad_rate0 is not None:
            self.epoch_dr_metric_list_dict[self.epoch_idx].append(median_grad_rate0)

    def assess_trial_end(self):
        early_stop = False
        for symptom_name in self.symptom_name_list:
            if self.info_dict[symptom_name] is not None:  ###
                early_stop = True
                self.info_dict["early_stop"] = True
        ###########
        logger.info(early_stop)
        self.record_global_metric()
        return early_stop

    def receive_monitor_result(self, study_name,  trial_id):
        # import pdb
        # pdb.set_trace()
        monitor_info = BTMessenger().read_monitor_info(study_name, trial_id)
        def get_metric_array(p, s):
            idx = self.metric_prefix_list.index(p) * len(self.metric_suffix_list) + self.metric_suffix_list.index(s)
            return self.module_metric_2da[:, idx].flatten()

        d = monitor_info
        self.result_dict = monitor_info
        self.epoch_idx = self.result_dict["epoch_idx"]
        self.param_name_list = d["param_name_list"]
        self.has_nan_inf_list = d["has_nan_inf_list"]
        if self.quick_calc:
            self.weight_grad_abs_avg_1da = d["weight_grad_abs_avg_1da"]
            self.weight_grad_rate0_1da = d["weight_grad_rate0_1da"]
        else:
            self.module_metric_2da = d["module_metric_2da"]
            self.metric_prefix_list = d["metric_prefix_list"]
            self.metric_suffix_list = d["metric_suffix_list"]
            self.weight_grad_rate0_1da = get_metric_array("weight_grad_rate", "rate0")
            self.weight_grad_abs_avg_1da = get_metric_array("weight_grad_abs", "avg")

        if type(self.weight_grad_abs_avg_1da) is dict:
            self.weight_grad_abs_avg_1da = np.array(self.weight_grad_abs_avg_1da["__ndarray__"])
            self.weight_grad_rate0_1da = np.array(self.weight_grad_rate0_1da["__ndarray__"])

    def if_top_train_loss(self, train_loss):
        if self.trial_id in self.protect_id_set:
            return True
        _train_loss_list = self.epoch_train_loss_list_dict[self.epoch_idx]  ####
        if len(_train_loss_list) < self.min_cmp_num:
            return False
        train_loss_t = np.percentile(_train_loss_list, self.cmp_percent)  # idx越大val越大
        if train_loss <= train_loss_t:
            self.protect_id_set.add(self.trial_id)
            return True
        return False

    def if_in_stage(self, stage_name):
        if stage_name == "half1":
            return self.epoch_idx < self.max_epoch // 2
        elif stage_name == "half2":
            return self.epoch_idx >= self.max_epoch // 2
        else:
            return True

    def diagnose_symptom(self):

        logger.info(self.symptom_name_list)
        self.diagnose_eg() if "EG" in self.symptom_name_list else None
        if self.has_nan_inf:
            return
        self.diagnose_vg() if "VG" in self.symptom_name_list else None
        self.diagnose_dr() if "DR" in self.symptom_name_list else None
        self.diagnose_sc() if "SC" in self.symptom_name_list else None
        self.diagnose_ho() if "HO" in self.symptom_name_list else None
        self.diagnose_nmg() if "NMG" in self.symptom_name_list else None
        # self.diagnose_of() if "OF" in self.symptom_name_list else None

    def diagnose_of(self):
        self.info_dict["OF"] = []
        val_loss_list = np.array(self.result_dict["val_loss_list"])
        latest_val_loss = val_loss_list[-self.dp.wd_nmg:]
        slope, _ = np.polyfit(np.arange(len(latest_val_loss)), latest_val_loss, 1)
        if slope < self.dp.p_of:
            self.info_dict["HO"].append("ho_rule_train")
        self.info_dict["OF"] = self.info_dict["OF"] if len(self.info_dict["OF"]) != 0 else None

    def diagnose_eg(self):
        # EG: (step:half1)
        # eg_rule0: any(has_nan or has_inf)
        # eg_rule1: max(grad_abs_ave) > p_eg1 ||| (p_eg1:10)
        # eg_rule2: max(adjacent_quotient) > p_eg2 ||| (p_eg2:1000)
        # eg_rule3: train_loss >= cmp_median_train_loss * p_eg3 ||| (p_eg3:10)
        self.info_dict["EG"] = []
        self.eg_rule0()
        if self.has_nan_inf:
            pass
        else:
            self.eg_rule1()
            self.eg_rule2()
        self.info_dict["EG"] = self.info_dict["EG"] if len(self.info_dict["EG"]) != 0 else None

    def eg_rule0(self):
        symptom_flag = False
        if True in self.has_nan_inf_list:
            symptom_flag = True
        if type(self.get_metric("train_loss")) is None:
            symptom_flag = True
        if type(self.weight_grad_abs_avg_1da[0]) is not np.float64:
            symptom_flag = True
        self.has_nan_inf = symptom_flag
        return symptom_flag

    def eg_rule1(self):
        symptom_flag = False
        if np.median(self.weight_grad_abs_avg_1da) > self.dp.p_eg1:
            symptom_flag = True
        if symptom_flag:
            self.info_dict["EG"].append("eg_rule1")

    def eg_rule2(self):
        symptom_flag = False
        if 0 in self.weight_grad_abs_avg_1da:
            return  # vg
        adjacent_quotient_list = self.weight_grad_abs_avg_1da[:-1] / self.weight_grad_abs_avg_1da[1:]  ####
        if True in np.isnan(adjacent_quotient_list) or True in np.isinf(adjacent_quotient_list):
            symptom_flag = True
        elif np.median(adjacent_quotient_list) > self.dp.p_eg2:
            symptom_flag = True
        if symptom_flag:
            self.info_dict["EG"].append("eg_rule2")

    def diagnose_vg(self):
        # VG: (step:half1)
        # protect_top_loss: True
        # vg_rule1: median(grad_abs_ave) < p_vg1 ||| (p_vg1:1.e-7) 。。。有点魔法数,而且作用很小。。。
        # vg_rule2: median(adjacent_quotient) < p_vg2 ||| (p_vg3:0.01) 逻辑修复+已经改大p_vg3+又改小了些
        # vg_rule3: mean(abs_delta_train_loss) < train_loss[0] * p_vg3 ||| (p_vg3:0.001) 逻辑已修改！！！！！！!!!!!!!!!!!
        # vg_rule4: enough(cmp_num) && percent(global_vg_metric_list) < p_vg4 ||| (p_vg4:0.1) 新加入！！！！！
        self.info_dict["VG"] = []
        if not self.if_top_train_loss(self.get_metric("train_loss")):
            self.vg_rule1()
            self.vg_rule2()
        self.info_dict["VG"] = self.info_dict["VG"] if len(self.info_dict["VG"]) != 0 else None

    def vg_rule1(self):
        if not self.if_in_stage("half1"):
            return
        symptom_flag = False
        if np.median(self.weight_grad_abs_avg_1da) < self.dp.p_vg1:
            symptom_flag = True
        if symptom_flag:
            self.info_dict["VG"].append("vg_rule1")

    def vg_rule2(self):
        if not self.if_in_stage("half1"):
            return
        symptom_flag = False
        if 0 in self.weight_grad_abs_avg_1da:
            symptom_flag = True
        else:
            adjacent_quotient_list = np.abs(self.weight_grad_abs_avg_1da[:-1] / self.weight_grad_abs_avg_1da[1:])  ####
            if True in np.isnan(adjacent_quotient_list) or True in np.isinf(adjacent_quotient_list):
                symptom_flag = True
            elif np.median(adjacent_quotient_list) < self.dp.p_vg2:
                symptom_flag = True
        if symptom_flag:
            self.info_dict["VG"].append("vg_rule2")

    def diagnose_dr(self):
        # DR: (step:all)
        # protect_top_loss: True
        # dr_rule1: median(rate0) < p_dr1 ||| (p_dr1:0.1) 已经改any为median且调小了p_dr1
        # dr_rule2: weighted_mean(rate0) > p_dr2 ||| (p_dr2:0.5) 。。。同质，考虑取消
        # dr_rule3: enough(cmp_num) && percent(global_median_rate0) > p_dr3 ||| (p_dr3:0.9) 新加入！！！！！
        self.info_dict["DR"] = []
        if not self.if_top_train_loss(self.get_metric("train_loss")):
            self.dr_rule()
        self.info_dict["DR"] = self.info_dict["DR"] if len(self.info_dict["DR"]) != 0 else None

    def dr_rule(self):
        train_loss_list = np.array(self.result_dict["train_loss_list"])
        if len(train_loss_list) < 5:
            return
        if np.median(self.weight_grad_rate0_1da) > self.dp.p_dr1:
            self.info_dict["DR"].append("dr_rule1")

    def diagnose_sc(self):
        # SC: (step:half1)
        # protect_top_loss: True
        # sc_rule1: (acc[0]-acc[-1]) / acc[0] > p_sc1 ||| (p_sc1:0) 。。。同质acc混，考虑取消
        # sc_rule2: (loss[-1]-loss[0]) / loss[0] > p_sc2 ||| (p_sc2:0) 已经改逻辑，只检测比初始还差的
        # sc_rule3: percentile(loss) < p_sc3 * 100 ||| (p_sc3:0.5) ...新增，为了三角
        self.info_dict["SC"] = []
        if not self.if_top_train_loss(self.get_metric("train_loss")):  #######
            self.sc_rule_early()
            self.sc_rule_late_train()
            self.sc_rule_late_val()
        self.info_dict["SC"] = self.info_dict["SC"] if len(self.info_dict["SC"]) != 0 else None

    def sc_rule_early(self):
        if not self.if_in_stage("half1"):
            return
        train_loss_list = np.array(self.result_dict["train_loss_list"])
        if len(train_loss_list) < 3:
            return
        sub_list = train_loss_list
        cova = np.std(sub_list) / np.mean(sub_list)
        if cova < self.dp.p_sc1:
            self.info_dict["SC"].append("sc_rule_early")

    def sc_rule_late_train(self):
        if not self.if_in_stage("half2"):
            return
        train_loss_list = np.array(self.result_dict["train_loss_list"])
        if len(train_loss_list) < self.dp.wd_nmg:
            return
        sub_list_ahead = train_loss_list[:int(len(train_loss_list)/2)]
        sub_list_curr = train_loss_list[int(len(train_loss_list)/2):]
        ahead_cova = np.std(sub_list_ahead) / np.mean(sub_list_ahead)
        curr_cova = np.std(sub_list_curr) / np.mean(sub_list_curr)
        if ahead_cova < curr_cova:
            self.info_dict["SC"].append("sc_rule_late_train")

    def sc_rule_late_val(self):
        if not self.if_in_stage("half2"):
            return
        val_loss_list = np.array(self.result_dict["val_loss_list"])
        if len(val_loss_list) < self.dp.wd_nmg:
            return
        sub_list_ahead = val_loss_list[:int(len(val_loss_list)/2)]
        sub_list_curr = val_loss_list[int(len(val_loss_list)/2):]
        ahead_cova = np.std(sub_list_ahead) / np.mean(sub_list_ahead)
        curr_cova = np.std(sub_list_curr) / np.mean(sub_list_curr)
        if ahead_cova < curr_cova:
            self.info_dict["SC"].append("sc_rule_late_val")

    def diagnose_ho(self):
        # HO(heavy oscillation): (step:half2) (wd:0.25) 。。。。斜率配合MAE
        # ho_rule1: std(acc[-wd:]) / mean(acc[-wd:]) > p_ho1 ||| (p_ho1:0) 。。。同质acc混，考虑取消
        # ho_rule2: MAE(loss[-wd:] - line(loss[-wd:])) > mean(loss[-wd:]) * p_ho2  ||| (p_ho2:0.1) 已经改逻辑，有待验证
        self.info_dict["HO"] = []
        self.ho_rule_train()
        self.ho_rule_val()
        self.info_dict["HO"] = self.info_dict["HO"] if len(self.info_dict["HO"]) != 0 else None

    def ho_rule_train(self):
        if not self.if_in_stage("half2"):
            return
        train_loss_list = np.array(self.result_dict["train_loss_list"])
        window_size = int(round(self.dp.wd_ho))
        sub_list = train_loss_list[-window_size:]
        cova = np.std(sub_list) / np.mean(sub_list)
        slope, _ = np.polyfit(np.arange(len(sub_list)), sub_list, 1)
        if slope >= self.dp.p_ho1 and cova > self.dp.p_ho2:
            self.info_dict["HO"].append("ho_rule_train")

    def ho_rule_val(self):
        if not self.if_in_stage("half2"):
            return
        val_loss_list = np.array(self.result_dict["val_loss_list"])
        window_size = int(round(self.dp.wd_ho))
        sub_list = val_loss_list[-window_size:]
        cova = np.std(sub_list) / np.mean(sub_list)
        slope, _ = np.polyfit(np.arange(len(sub_list)), sub_list, 1)
        if slope >= self.dp.p_ho1 and cova > self.dp.p_ho2:
            self.info_dict["HO"].append("ho_rule_val")


    # NMG is useless. It should be replaced by OF. TODO Zhongyi Pei
    def diagnose_nmg(self):
        self.info_dict["NMG"] = []
        if not self.if_top_train_loss(self.get_metric("train_loss")):  ######
            self.nmg_rule_train()
            self.nmg_rule_val()
        self.info_dict["NMG"] = self.info_dict["NMG"] if len(self.info_dict["NMG"]) != 0 else None

    def nmg_rule_val(self):
        val_loss_list = np.array(self.result_dict["val_loss_list"])
        window_size = int(round(self.dp.wd_nmg/2))
        head_avg = np.mean(val_loss_list[-self.dp.wd_nmg:-window_size])
        tail_avg = np.mean(val_loss_list[-window_size:])
        if head_avg < tail_avg:
            self.info_dict["NMG"].append("nmg_rule_val")

    def nmg_rule_train(self):
        train_loss_list = np.array(self.result_dict["train_loss_list"])
        window_size = int(round(self.dp.wd_nmg/2))
        head_avg = np.mean(train_loss_list[-self.dp.wd_nmg:-window_size])
        tail_avg = np.mean(train_loss_list[-window_size:])
        if head_avg < tail_avg:
            self.info_dict["NMG"].append("nmg_rule_train")
