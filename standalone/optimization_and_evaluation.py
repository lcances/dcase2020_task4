import os
import torch.utils.data

import sys
sys.path.append("..")

import numpy
from tqdm import tqdm
import argparse

# dataset manager
from dcase2020.datasetManager import DESEDManager
from dcase2020.datasets import DESEDDataset

# utility function & metrics & augmentation
from metric_utils.metrics import FScore
from dcase2020_task4.util.utils import reset_seed, get_model_from_name

# All models available
from dcase2020_task4.baseline.models import *
from dcase2020_task4.dcase2019.models import *

# for strong output optimization
from aeseg.Encoder import Encoder
from aeseg.optimizers import GenOptimizer, DichotomicOptimizer
from aeseg.aeseg import eb_evaluator, sb_evaluator


parser = argparse.ArgumentParser()
parser.add_argument("--model_save", default="../models/best_dcase2019.torch", help="Path to model save using checkpoint")
parser.add_argument("--model_name", default="dcase2019_model", help="Name of the Class/function to use to construct the model")
parser.add_argument("-a", "--audio_root", default="../dataset/DESED/dataset/audio", type=str)
parser.add_argument("-m", "--metadata_root", default="../dataset/DESED/dataset/metadata", type=str)
parser.add_argument("-w", "--num_workers", default=1, type=int, help="Choose number of worker to train the model")
parser.add_argument("-o", "--output", default="submission.csv", type=str, help="submission file name")
args = parser.parse_args()


# ==== set the log ====
import logging
import logging.config
from dcase2020.util.log import DEFAULT_LOGGING
logging.config.dictConfig(DEFAULT_LOGGING)
log = logging.getLogger(__name__)


# ==== reset the seed for reproductability ====
reset_seed(1234)


# ==== load the dataset ====
desed_metadata_root = args.metadata_root
desed_audio_root = args.audio_root



# ======================================================================================================================
# PREPARE DATASET AND MODEL
# ======================================================================================================================
manager = DESEDManager(
    desed_metadata_root, desed_audio_root,
    sampling_rate = 22050,
    from_disk=False,
    nb_vector_bin=431, # there is no temporal reduction in this model
    verbose = 1
)
manager.add_subset("eval")
manager.add_subset("validation")

eval_dataset = DESEDDataset(manager, train=True, val=False, weak=False, strong=False, augments=[], cached=False)
val_dataset = DESEDDataset(manager, train=False, val=True, weak=True, strong=True, augments=[], cached=True)

eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=64, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)


model_func = get_model_from_name(args.model_name)
best_model = model_func()
best_model.cuda()
best_model.eval() # <-- for consistency in scoring (deactivate dropout and batchNorm)
log.info("Model %s loaded" % args.model_name)

checkpoint = torch.load(args.model_save)
best_model.load_state_dict(checkpoint["state_dict"])
log.info("Best state reach at epoch %d with score %.2f" % (checkpoint["epoch"], checkpoint["best_metric"]))


# ======================================================================================================================
# PREDICTION ON VALIDATION DATASET
# ======================================================================================================================
weak_y_true, strong_y_true = None, None
weak_y_pred, strong_y_pred = None, None
y_filenames = list(val_dataset.X.keys())

with torch.set_grad_enabled(False):
    for i, (X, y) in tqdm(enumerate(val_loader)):
        weak_y, strong_y = y
        weak_y, strong_y = weak_y.cuda(), strong_y.cuda()
        X = X.cuda()
        
        weak_logits, strong_logits = best_model(X)
        
        weak_pred = torch.sigmoid(weak_logits)
        strong_pred = torch.sigmoid(strong_logits)
        
        # accumulate prediction and ground truth
        if i == 0:
            weak_y_true = weak_y.cpu()
            strong_y_true = strong_y.cpu()

            weak_y_pred =  weak_pred.cpu()
            strong_y_pred =  strong_pred.cpu()
        else:
            weak_y_true = torch.cat((weak_y_true, weak_y.cpu()), dim=0)
            strong_y_true = torch.cat((strong_y_true, strong_y.cpu()), dim=0)

            weak_y_pred =  torch.cat((weak_y_pred, weak_pred.cpu()), dim=0)
            strong_y_pred =  torch.cat((strong_y_pred, strong_pred.cpu()), dim=0)


# ======================================================================================================================
# audio tagging threshold optimization
# ======================================================================================================================
classwise_f1 = FScore(dim=0)


def simulated_anealing(y_pred, y_true, macro_iteration = 50, micro_iteration = 400):
    weak_y_pred_ = y_pred.clone().detach()

    classwise_f1(weak_y_pred_, y_true)
    start_f1 = classwise_f1.value

    min_delta = 10e-7
    delta_ratio = 0.2
#     macro_iteration = 30
#     micro_iteration = 400

    history = {
        "best_f1": [[] for _ in range(10)],
        "f1": [],
        "delta_ratio": []
    }

    best_thresholds = torch.ones(10) * 0.5
    best_f1 = start_f1.clone().detach()

    total_iteration = macro_iteration * micro_iteration

    for M in range(macro_iteration):
        thresholds = torch.ones(10) * 0.5
        delta_ratio = 0.3
        delta_decay = ( (min_delta + delta_ratio) / micro_iteration )

        for m in range(micro_iteration):
            bin_y_pred = y_pred.clone().detach()

            # calc new threhsold
            r = torch.normal(mean=thresholds, std=0.4)
            delta = r * delta_ratio
            new_thresholds = thresholds + delta
            delta_ratio -= delta_decay
            history["delta_ratio"].append(delta_ratio)

            # apply threshold
            weak_y_pred_[bin_y_pred > new_thresholds] = 1
            weak_y_pred_[bin_y_pred <= new_thresholds] = 0

            # calc new f1
            classwise_f1(weak_y_pred_, y_true)
            new_f1 = classwise_f1.value
            history["f1"].append(new_f1)

            # check
            for i in range(10):
                if new_f1[i] > best_f1[i]:
                    best_f1[i] = new_f1[i]
                    best_thresholds[i] = new_thresholds[i]
                    thresholds[i] = best_thresholds[i]
                    history["best_f1"][i].append(best_f1[i])

            step = M * micro_iteration + m
            print("%2.2f / 100.0 (%%)" % ((step / total_iteration) * 100), end="\r")
            
    return start_f1, best_f1, best_thresholds, history

log.info("Start tagging optimization")
initial_f1, best_f1, best_at_thresholds, history = simulated_anealing(weak_y_pred, weak_y_true, 20, 400)

log.info("Previous macro F1 score: %.2f - New macro F1 score: %.2f" % (torch.mean(initial_f1), torch.mean(best_f1)))

# ==== pruning of strong prediction ====
log.info("Pruning strong prediction ...")
best_weak_y_pred = weak_y_pred.clone().detach()
best_weak_y_pred[best_weak_y_pred > best_at_thresholds] = 1
best_weak_y_pred[best_weak_y_pred <= best_at_thresholds] = 0


# Prune the result of fill the missing curve with 0 segments
def prune_prediction(strong_y_pred, weak_y_pred):
    """ Prune the strong prediciton by zeroing all classes that are not predicted. """
    pruned_strong_y_pred = strong_y_pred.clone().detach()
    
    for index in range(len(pruned_strong_y_pred)):
        cls_result = weak_y_pred[index]
        
        # Create a full mask using repeat
        length = strong_y_pred.size()[-1]
        cls_mask = cls_result.unsqueeze(1).repeat(1, length)
        pruned_strong_y_pred[index] *= cls_mask
                
    return pruned_strong_y_pred


pruned_strong_y_pred = prune_prediction(strong_y_pred, best_weak_y_pred)


# ======================================================================================================================
# AUDIO LOC OPTIMIZATION
# ======================================================================================================================


def load_csv(path):
    with open(path, "r") as f:
        data = f.read().splitlines()[1:]
    return data

strong_y_true = strong_y_true.permute(0, 2, 1)
pruned_strong_y_pred = pruned_strong_y_pred.permute(0, 2, 1)


def class_wise_optimization(optimizer, class_to_monitor, strong_y_true, strong_pred, filenames, method="threshold"):
    class_wise_results = {}
    
    for cl in class_to_monitor:
        optimizer.fit(
            strong_y_true, strong_pred, filenames,
            monitor=["class_wise", cl, "f_measure", "f_measure"],
            method=method,
            verbose=1)
        
        parameters, score = optimizer.best
        print(cl, parameters, score)
        class_wise_results[cl] = (parameters, score)
        
    return class_wise_results


def get_class_dependant_parameters(class_wise_results, class_list):
    class_dependant_parameters = {}
    for cl in class_list:
        for param in class_wise_results[cl][0]:
            if param not in class_dependant_parameters.keys():
                class_dependant_parameters[param] = []
            class_dependant_parameters[param].append(class_wise_results[cl][0][param])
            
    # TODO find better way
    # for "smooth" common parameters, remove list
    class_dependant_parameters["smooth"] = class_dependant_parameters["smooth"][0]
    return class_dependant_parameters


class_list = list(DESEDManager.cls_dict.keys())

# ==== Create the encoder with challenge parameters ====
encoder = Encoder(
    classes=class_list,
    temporal_precision = 50,  # ms
    clip_length = 10,          # s
    minimal_segment_step = 150 # ms
)

val_csv_y_true= load_csv(os.path.join(desed_metadata_root, "validation", "validation.tsv"))

# ## Hysteresis thresholding
optimizer = DichotomicOptimizer(
    {
        "high": (0.01, 0.99),
        "low": (0.01, 0.99),
        "smooth": "smoothMovingAvg",
        "window_len": (3, 27)
    },
    
    encoder = encoder,
    step = 3,
    nb_recurse = 1,
    nb_process = args.num_workers
)

class_wise_results = class_wise_optimization(
    optimizer,
    class_list,
    method="hysteresis",
    strong_y_true=val_csv_y_true,
    strong_pred=pruned_strong_y_pred.numpy(),
    filenames=y_filenames
)


best_parameters = get_class_dependant_parameters(class_wise_results, class_list)

segments = encoder.encode(
    pruned_strong_y_pred.numpy(),
    method="hysteresis",
    **best_parameters
)
to_evaluate = encoder.parse(segments, y_filenames)
evaluator = eb_evaluator(val_csv_y_true, to_evaluate)
print(evaluator)


# # ♫♪.ılılıll|̲̅̅●̲̅̅|̲̅̅=̲̅̅|̲̅̅●̲̅̅|llılılı.♫♪


# ======================================================================================================================
# EVALUATION DATASET
# ======================================================================================================================
log.info("Prediction of the evaluation dataset ...")
weak_y_pred, strong_y_pred = None, None
y_filenames = list(eval_dataset.X.keys())

with torch.set_grad_enabled(False):
    for i, (X, y) in tqdm(enumerate(eval_loader)):
        X = X.cuda()

        weak_logits, strong_logits = best_model(X)

        weak_pred = torch.sigmoid(weak_logits)
        strong_pred = torch.sigmoid(strong_logits)

        # accumulate prediction and ground truth
        if i == 0:
            weak_y_pred = weak_pred.cpu()
            strong_y_pred = strong_pred.cpu()

        else:
            weak_y_pred = torch.cat((weak_y_pred, weak_pred.cpu()), dim=0)
            strong_y_pred = torch.cat((strong_y_pred, strong_pred.cpu()), dim=0)
            
pruned_strong_y_pred = pruned_strong_y_pred.permute(0, 2, 1)

log.info("Pruning strong prediction using best audio tagging thresholds ...")
best_weak_y_pred = weak_y_pred.clone().detach()
best_weak_y_pred[best_weak_y_pred > best_at_thresholds] = 1
best_weak_y_pred[best_weak_y_pred <= best_at_thresholds] = 0

pruned_strong_y_pred = prune_prediction(strong_y_pred, best_weak_y_pred)

log.info("Apply best segmentation algorithm")
segments = encoder.encode(
    pruned_strong_y_pred.numpy(),
    method="hysteresis",
    **best_parameters
)
to_evaluate = encoder.parse(segments, y_filenames)

log.info("Create submission.csv file")
with open("submission.csv", "w") as f:
    f.write(to_evaluate)
