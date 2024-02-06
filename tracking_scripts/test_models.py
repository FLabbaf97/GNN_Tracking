from typing import Optional
from bread.data import *
from bread.algo.tracking import AssignmentDataset, GNNTracker, AssignmentClassifier, GraphLoader, accuracy_assignment, f1_assignment

from glob import glob
from pathlib import Path
import datetime, json, argparse
import torch, os
from pprint import pprint
from sklearn.metrics import confusion_matrix
import pandas as pd

from skorch.dataset import ValidSplit
from skorch.callbacks import LRScheduler, Checkpoint, ProgressBar, EpochScoring, WandbLogger, EarlyStopping

from trainutils import SaveHyperParams, seed

from importlib import import_module

import tqdm
import re


def sum_results(results):
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
    return res

def save_resutls(res_details_dataframe, result_dir, name):
    results_dataframe = res_details_dataframe.groupby(['timediff', 'colony', 'seed', 'assgraphs'])[
    ['f1', 'tp', 'fp', 'fn', 'tn', 'num_cells1', 'num_cells2']].sum()
    results_dataframe['precision'] = results_dataframe['tp'] / (results_dataframe['tp'] + results_dataframe['fp'])
    results_dataframe['recall'] = results_dataframe['tp'] / (results_dataframe['tp'] + results_dataframe['fn'])
    results_dataframe['f1'] = 2*results_dataframe['tp'] / (2*results_dataframe['tp'] + results_dataframe['fp'] + results_dataframe['fn'])
    results_dataframe[['f1', 'precision', 'recall']]
    results_dataframe.to_csv(os.path.join(result_dir, f'results_{name}.csv'))
    res_details_dataframe.to_csv(os.path.join(result_dir, f'results_details_{name}.csv'))    

def test_pipeline(net, test_array):
    results = {}
    results['gcn'] = {
        't1': [],
        't2': [],
        'colony':[],
        'confusion': [],
        'num_cells1': [],
        'num_cells2': [],
    }

    for file in tqdm.tqdm(test_array, desc='testing', position=0, leave=True):
        torch.cuda.empty_cache()

        # read graph from .pt file
        graph = torch.load(file)
        try:
            yhat = net.predict_assignment(graph).flatten()
            y = graph.y.squeeze().cpu().numpy()

            results['gcn']['confusion'].append(confusion_matrix(y, yhat))
            idt1_pattern = r'__(\d{3})_to'
            idt2_pattern = r'_to_(\d{3})'
            dt_pattern = r'_dt_(\d{3})'
            # data_name is from first underline to where we have "_segmentation"
            data_name_pattern = r'(\w+)_segmentation'
            
            idt1 = int(re.findall(idt1_pattern, file)[0])
            idt2 = int(re.findall(idt2_pattern, file)[0])
            dt = int(re.findall(dt_pattern, file)[0])
            data_name = re.findall(data_name_pattern, file)[0]
            
            results['gcn']['t1'].append(idt1)
            results['gcn']['t2'].append(idt2)
            results['gcn']['colony'].append(data_name)
            results['gcn']['num_cells1'].append(len(set([list(graph.cell_ids)[i][0] for i in range(len(graph.cell_ids))])))
            results['gcn']['num_cells2'].append(len(set([list(graph.cell_ids)[i][1] for i in range(len(graph.cell_ids))])))
        except Exception as e:
            print(f'error in {file}')
            print(e)
            continue
    res = sum_results(results)
    return res
                    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', dest='config', required=False, type=str, help='config file')
    parser.add_argument('--model', dest='model', required=False, type=str, help='model directory')
    parser.add_argument('--algo', dest='algo', default='hungarian', type=str, help='postprocessing algorithm')
    parser.add_argument('--resultdir', dest='resultdir', default='results', type=str, help='result directory')
    parser.add_argument('--data', dest='data', default='all', type=str, help='data to test on')
    parser.add_argument('--assgraphs', dest='assgraphs', default='all', type=str, help='assgraphs to test on')

    dataset_regex={
        'colony_5__dt_1_t_all': "^colony00(5)_segmentation__assgraph__dt_001__.*pt$",
        'test_set_1234567_dt_1234_t_all': "^test_set_(1|2|3|4|5|6|7)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",	
        'colony_056_dt_1234_t_all': "^colony00(0|5|6)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
        'colony_0567_test_set_1234567_dt_1234_t_all': "^(colony00(0|5|6|7)|test_set_(1|2|3|4|5|6|7))_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
        'colony_7_test_set_1234567_dt_1_t_all': "^(colony00(7)|test_set_(1|2|3|4|5|6|7))_segmentation__assgraph__dt_00(1)__.*pt$",
        'fission_all': "^240102_30C_fig_(SW182|SX387)_(01|02|03|05)_segmentation__assgraph__dt_00(1)__.*pt$",
    }
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device : {device}')

    result_dir = args.resultdir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    if args.model is None:
        model_dirs = {'/mlodata1/hokarami/fari/tracking/results/scaled_images/results_features/_100KB_': '/mlodata1/hokarami/fari/tracking/generated_data/scaled_images/ass_graphs_features',
                      '/mlodata1/hokarami/fari/tracking/results/scaled_images/results_fourier_10_and_f10_locality_False/_100KB_': '/mlodata1/hokarami/fari/tracking/generated_data/scaled_images/ass_graphs_fourier_10_f10_locality_False',
                      '/mlodata1/hokarami/fari/tracking/results/scaled_images/results_fourier_10_and_locality_False/_100KB_': '/mlodata1/hokarami/fari/tracking/generated_data/scaled_images/ass_graphs_fourier_10_locality_False',
                      }
    else:
        model_dirs = {args.model: args.assgraphs}
    
    results_dataframe = pd.DataFrame(columns = ['model', 'assgraphs','colony', 'seed', 'timediff', 'f1', 'precision', 'recall', 'acc', 'tp', 'fp', 'fn', 'tn', 'num_cells1', 'num_cells2'])
    res_details_dataframe = pd.DataFrame()
    # enumurate over dictionary
    
    for folder, assgraphs in model_dirs.items():
        file_array = []
    
        regex = dataset_regex[args.data]
        for filename in os.listdir(assgraphs):
            if re.search(regex, filename):
                file_array.append(os.path.join(assgraphs, filename))
        seed = 0
        for model in os.listdir(folder):
            model_dir = Path(folder)/ model
            with open(model_dir / 'hyperparams.json') as file:
                hparams = json.load(file)

            new_net = AssignmentClassifier(
                GNNTracker,
                module__num_node_attr=hparams['num_node_attr'],
                module__num_edge_attr=hparams['num_edge_attr'],
                module__dropout_rate=hparams['dropout_rate'],
                module__encoder_hidden_channels=hparams['encoder_hidden_channels'],
                module__encoder_num_layers=hparams['encoder_num_layers'],
                module__conv_hidden_channels=hparams['conv_hidden_channels'],
                module__conv_num_layers=hparams['conv_num_layers'],
                module__num_classes=1,
                iterator_train=GraphLoader,
                iterator_valid=GraphLoader,
                criterion=torch.nn.BCEWithLogitsLoss,
            ).initialize()
            new_net.load_params(model_dir / 'params.pt')
            print('model loaded')
            
            # Put the network in test mode
            new_net.module_.train(False)  # Set the network to evaluation mode
            res_details = test_pipeline(new_net, file_array)
            # results['assgraphs'] = assgraphs
            # results['seed'] = seed
            res_details['assgraphs'] = assgraphs
            res_details['seed'] = seed
            # results_dataframe = pd.concat([results_dataframe,results])
            res_details_dataframe = pd.concat([res_details_dataframe,res_details])
            name = f'{args.data}_{model}_seed_{seed}'
            save_resutls(res_details_dataframe, result_dir, name)
            seed += 1
    save_resutls(res_details_dataframe, result_dir, args.data)

    

