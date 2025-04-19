# BTTackler
A Diagnosis-based Framework for Efficient Deep Learning Hyperparameter Optimization.

## Install
*Based on Python 3.10*
```
git clone git@github.com:OswinGuai/bttackler.git
cd bttackler
export PYTHONPATH=.:$PYTHONPATH
pip install torch scipy torchvision optuna
```

## Run Cases
```
python test.py
```
or for multiprocessing,
```
bash cifar10cnn_parallel.sh
```

## TODO
1. Higher code readability and configurable quality indicators.
2. Code formatting and complete experiment cases.
3. Cases for calibrating quality indicators.

## Citation
If you find BTTackler useful, please cite our paper.
```
@inproceedings{pei2024bttackler,
  title={BTTackler: A Diagnosis-based Framework for Efficient Deep Learning Hyperparameter Optimization},
  author={Zhongyi Pei and Zhiyao Cen and Yipeng Huang and Chen Wang and Lin Liu and Philip Yu and Mingsheng Long and Jianmin Wang},
  booktitle={In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2024},
}
```
## Contact
If you have any questions or suggestions, feel free to contact:

- Zhongyi Pei (peizhyi@tsinghua.edu.cn)
- Zhiyao Cen (cenzy23@mails.tsinghua.edu.cn)

Or describe it in Issues.
