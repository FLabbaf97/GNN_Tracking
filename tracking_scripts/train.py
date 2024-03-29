from typing import Optional
from bread.data import *
from bread.algo.tracking import AssignmentDataset, GNNTracker, AssignmentClassifier, GraphLoader, accuracy_assignment, f1_assignment

from glob import glob
from pathlib import Path
import datetime, json, argparse
import torch, os
from pprint import pprint
from sklearn.metrics import confusion_matrix


from skorch.dataset import ValidSplit
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler, Checkpoint, ProgressBar, EpochScoring, WandbLogger, EarlyStopping
from skorch.callbacks import Callback


from trainutils import SaveHyperParams, seed

from importlib import import_module
import numpy as np
import pandas as pd




import wandb
WANDB_API_KEY="e0f887ce4be7bebfe48930ffcff4027f49b02425"
os.environ['WANDB_API_KEY'] = WANDB_API_KEY
os.environ['WANDB_CONSOLE'] = "off"
os.environ['WANDB_JOB_TYPE'] = 'features_test'

def seed_torch(seed=42):
    # After all of this, still the seed seems not to be fixed!!
    import random
    import os
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    
def test_pipeline(net, test_array, result_dir, data_name, assigment_method = None):
    results = {}
    results['gcn'] = {
        't1': [],
        't2': [],
        'colony':[],
        'confusion': [],
        'num_cells1': [],
        'num_cells2': [],
    }

    for file in test_array:
        # read graph from .pt file
        graph = torch.load(file)
        yhat = net.predict_assignment(graph, assignment_method=assigment_method).flatten()
        y = graph.y.squeeze().cpu().numpy()

        results['gcn']['confusion'].append(confusion_matrix(y, yhat))
        idt1_pattern = r'__(\d{3})_to'
        idt2_pattern = r'_to_(\d{3})'
        dt_pattern = r'_dt_(\d{3})'
        #  colony is anything before _segmentation 
        c_pattern = r'(\w+)_segmentation'
        idt1 = int(re.findall(idt1_pattern, file)[0])
        idt2 = int(re.findall(idt2_pattern, file)[0])
        dt = int(re.findall(dt_pattern, file)[0])
        colony = re.findall(c_pattern, file)[0]

        results['gcn']['t1'].append(idt1)
        results['gcn']['t2'].append(idt2)
        results['gcn']['colony'].append(colony)
        results['gcn']['num_cells1'].append(len(set([list(graph.cell_ids)[i][0] for i in range(len(graph.cell_ids))])))
        results['gcn']['num_cells2'].append(len(set([list(graph.cell_ids)[i][1] for i in range(len(graph.cell_ids))])))
    
    res = pd.DataFrame(results['gcn'])
    res['method'] = 'gcn'
    
    # num_cells = [len(seg.cell_ids(idt)) for idt in range(len(seg))]

    res['tp'] = res['confusion'].map(lambda c: c[1, 1])
    res['fp'] = res['confusion'].map(lambda c: c[0, 1])
    res['tn'] = res['confusion'].map(lambda c: c[0, 0])
    res['fn'] = res['confusion'].map(lambda c: c[1, 0])
    res['acc'] = (res['tp'] + res['tn']) / \
        (res['tp'] + res['fp'] + res['tn'] + res['fn'])
    res['f1'] = 2*res['tp'] / (2*res['tp'] + res['fp'] + res['fn'])
    # res['num_cells1'] = res['t1'].map(lambda t: num_cells[int(t)])
    # res['num_cells2'] = res['t2'].map(lambda t: num_cells[int(t)])
    res['timediff'] = 5 * (res['t2'] - res['t1'])
    res.drop(columns='confusion', inplace=True)

    # save results
    res.to_csv(result_dir/f'result_{data_name}_{assigment_method}.csv', index=False)
    res_sum = res.groupby(['timediff', 'colony'])[
    ['f1', 'tp', 'fp', 'fn', 'tn', 'num_cells1', 'num_cells2']].sum()
    res_sum['precision'] = res_sum['tp'] / (res_sum['tp'] + res_sum['fp'])
    res_sum['recall'] = res_sum['tp'] / (res_sum['tp'] + res_sum['fn'])
    res_sum['f1'] = 2*res_sum['tp'] / \
        (2*res_sum['tp'] + res_sum['fp'] + res_sum['fn'])
    res_sum[['f1', 'precision', 'recall']]
    print('result on test data: ', data_name)
    res_sum.to_csv(result_dir/f'result_sum_{data_name}_{assigment_method}.csv')
    print(res_sum)


class AssignmentCallback(Callback):
    def __init__(self):
        # self.classifier = classifier
        self.weight_sum = 0

    def on_epoch_end(self, net, **kwargs):
        for param in net.module_.parameters():
            self.weight_sum += param.sum().item()
        print(f'Sum of all weights: {self.weight_sum}')
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', dest='config', required=True, type=str, help='config file')
    parser.add_argument('--device', dest='device', default='cuda', type=str, help='device')

    args = parser.parse_args()
    config = import_module("bread.config." + args.config).configuration
    
    pretty_config = json.dumps(config, indent=4)
    print(pretty_config)

    train_config = config.get('train_config')
    print("train with config: ", config)

    dataset_regex = {
        'colony_5__dt_1234__t_all': "^colony005_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
        'colony_012345__dt_1234__t_all': "^colony00(0|1|2|3|4|5)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
        'colony_01234__dt_1234__t_all': "^colony00(0|1|2|3|4)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
        'colony_0123__dt_1234__t_all': "^colony00(0|1|2|3)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
        'colony_6__dt_1234__t_all': "^colony006_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
        'colony_56__dt_1234__t_all': "^colony00(5|6)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
        'colony_0123478_test_set_1234567_dt_1234_t_all': "^(colony00(0|1|2|3|4|7|8)|test_set_(1|2|3|4|5|6|7))_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
        'colony_01234678_test_set_1234567_dt_1234_t_all': "^(colony00(0|1|2|3|4|6|7|8)|test_set_(1|2|3|4|5|6|7))_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$", 
        'colony_012345678_test_set_1234567_dt_1234_t_all': "^(colony00(0|1|2|3|4|5|6|7|8)|test_set_(1|2|3|4|5|6|7))_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$", # all and each colony we have

        'test_set_1234567_dt_1234_t_all': "^test_set_(1|2|3|4|5|6|7)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
        'colony_012345678__dt_1234__t_all': "^colony00(0|1|2|3|4|5|6|7|8)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
        'colony_0123478_test_set_1234567_dt_12_t_all': "^(colony00(0|1|2|3|4|7|8)|test_set_(1|2|3|4|5|6|7))_segmentation__assgraph__dt_00(1|2)__.*pt$",
        # 'fission_train': "^240102_30C_fig_(SW182|SX387)_(01|03|05)_segmentation__assgraph__dt_00(1|2|3|4|5|6)__.*pt$", # all fission files first batch except for the sw182_02 file which is for test
        'fission_all':  "^^(wt_pom1D_01_(07|15|20|30)_R3D_REF_dv_trk_segmentation|240102_30C_fig_(SW182|SX387)_(01|02|03|05)_segmentation)__assgraph__dt_00(1|2|3|4)__.*pt$", # all fission files (02 and 30 is new)
        'fission_train': "^(wt_pom1D_01_(07|30|20)_R3D_REF_dv_trk_segmentation|240102_30C_fig_(SW182|SX387)_(01|03|05)_segmentation)__assgraph__dt_00(1|2|3|4)__.*pt$", # all fission files excepr for sw182_02 and wt_15
        'fission_test': "^(wt_pom1D_01_(15)_R3D_REF_dv_trk_segmentation|240102_30C_fig_(SW182|SX387)_(02)_segmentation)__assgraph__dt_00(1|2|3|4)__.*pt$", # only sw182_02 and wt_15
    }

    device = torch.device('cuda' if (torch.cuda.is_available() and args.device=='cuda') else 'cpu')
    # TODO: remove this
    # device = 'cpu'
    
    if train_config['min_file_kb'] is not None:
        min_file_kb = train_config['min_file_kb']
        resultdir = Path(f'{train_config["result_dir"]}/_{str(min_file_kb)}KB_/{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}')
    else:
        resultdir = Path(f'{train_config["result_dir"]}/{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}')
    print("-- result directory --")
    print(resultdir)
    os.makedirs(resultdir, exist_ok=True)
    with open(resultdir / 'metadata.json', 'w') as file:
        json.dump(config, file)

    print('-- train arguments --')
    print(json.dumps(train_config, indent=4))

    # filter out files that are too large
    file_array = []
    import re
    regex = dataset_regex[train_config["dataset"]]
    for filename in os.listdir(train_config["ass_graphs"]):
        if re.search(regex, filename):
            file_array.append(os.path.join(train_config["ass_graphs"], filename))
    file_array = [ filepath for filepath in file_array if os.stat(filepath).st_size/2**20 < train_config["filter_file_mb"] ]
    # filter file size smaller than config['min_file_kb'] for sake of training
    if train_config['min_file_kb'] is not None:
        file_array = [ filepath for filepath in file_array if os.stat(filepath).st_size/2**10 > train_config["min_file_kb"] ]


    if train_config['valid_dataset'] != None:
        valid_array = [filepath for filepath in file_array if train_config['valid_dataset'] in str(filepath)]
        train_array = [filepath for filepath in file_array if train_config['valid_dataset'] not in str(filepath)]
        train_array = file_array
        train_dataset = AssignmentDataset(train_array)
        valid_dataset = AssignmentDataset(valid_array)
    else:
        train_array = file_array
        train_dataset = AssignmentDataset(train_array)
        valid_dataset = None

    print('-- training dataset --')
    print(train_dataset)

    seed_value = 42
    seed_torch(seed=seed_value)
 
    train_dataset = train_dataset.shuffle()

    cv = None  
    if train_config["cv"] is not None:
        cv = ValidSplit(
            # This generates one train and validation dataset for the entire fit
            cv=train_config["cv"],
            stratified=False,  # Since we are using y=None
            # train_split__random_state=seed_value
        )
    
    # Create a wandb Run
    wandb_run = wandb.init(config=config, group=args.config, project="GCN_tracker", reinit=True)

    # scoring_system = f1_assignment if train_config['scoring'] == 'valid_f1_ass' else accuracy_assignment
    # scoring_system = f1_assignment

    net = AssignmentClassifier(
        GNNTracker,
        module__num_node_attr=len(train_dataset.node_attr),
        module__num_edge_attr=len(train_dataset.edge_attr),
        module__dropout_rate=train_config["dropout_rate"],
        module__encoder_hidden_channels=train_config["encoder_hidden_channels"],
        module__encoder_num_layers=train_config["encoder_num_layers"],
        module__conv_hidden_channels=train_config["conv_hidden_channels"],
        module__conv_num_layers=train_config["conv_num_layers"],
        module__num_classes=1,  # fixed, we do binary classification
        max_epochs=train_config["max_epochs"],
        device=device,
        
        criterion=torch.nn.BCEWithLogitsLoss(
            # attribute more weight to the y == 1 samples, because they are more rare
            pos_weight=torch.tensor(100)
        ),

        optimizer=torch.optim.Adam,
        optimizer__lr=train_config["lr"],
        optimizer__weight_decay=train_config["weight_decay"],  # L2 regularization

        iterator_train=GraphLoader,
        iterator_valid=GraphLoader,

        iterator_train__shuffle=True,
        iterator_valid__shuffle=False,
        batch_size=1,

        # define valid dataset
        train_split=predefined_split(valid_dataset) if train_config['valid_dataset'] else None,

        # train_split=cv,

        callbacks=[
            LRScheduler(policy='StepLR', step_every='batch', step_size=train_config["step_size"], gamma=train_config["gamma"]),
            Checkpoint(monitor='valid_loss_best', dirname=resultdir, f_pickle='pickle.pkl'),
            SaveHyperParams(dirname=resultdir),
            EarlyStopping(patience=train_config["patience"]),
            ProgressBar(detect_notebook=False),
            WandbLogger(wandb_run, save_model=True),
            EpochScoring(scoring=f1_assignment, lower_is_better=False, name='valid_f1_ass'),
            # AssignmentCallback(),
            # Evaluator(GraphLoader(test_dataset), "test" , lower_is_better=False)
        ],
    )

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    torch.cuda.empty_cache()
    
    print('-- starting training --')
    net.fit(train_dataset, y=None)
    print('-- starting testing --')
    test_array = []
    test_config = config.get('test_config')
    regex = dataset_regex[test_config["dataset"]]
    for filename in os.listdir(test_config["ass_graphs"]):
        if re.search(regex, filename):
            test_array.append(os.path.join(test_config["ass_graphs"], filename))
    original_test_array = test_array
    test_array = [ filepath for filepath in test_array if os.stat(filepath).st_size/2**20 < test_config["filter_file_mb"] ]
    rest = result_list = [item for item in original_test_array if item not in test_array]
    test_dataset = AssignmentDataset(test_array)
    print(f'ignore last {len(rest)} file because they were too big.')
    
    filtered_result = test_pipeline(net, test_array, resultdir, f'{test_config["dataset"]}_filtered' , assigment_method = test_config['assignment_method'])
    filtered_result = test_pipeline(net, test_array, resultdir, f'{test_config["dataset"]}_filtered' , assigment_method = 'hungarian')
    
    torch.cuda.empty_cache()

    # original_result = test_pipeline(net, rest, resultdir, f'{test_config["dataset"]}_bigs')
    # torch.cuda.empty_cache()

    
