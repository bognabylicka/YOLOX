import argparse
import os
import os.path as osp
from datetime import datetime
from subprocess import run

EXPERIMENT_DIR = '/media/cluster_fs/user/bbylicka/experiments/object_detection'
SNAPSHOTS_DIR = '/media/cluster_fs/user/bbylicka/experiments/snapshots'
DATASETS_DIR = '/media/cluster_fs/datasets/object_detection/'

# all datasets: 'aerial_tiled' 'bbcd' 'pothole' 'wildfire' 'vitens-tiled'
#               'diopsis-tiled' 'weed-tiled' 'kbts_fish' 'pcd' 'weed' 'diopsis'
#               'aerial_large' 'PCB_original' 'minneapple' 'wgisd1' 'wgisd5' 'dice'
SETS_WITHOUT_TEST = ['kbts_fish', 'pcd', 'vitens-tiled', 'wgisd1', 'wgisd5', 'weed-tiled', 'diopsis-tiled']
DATASETS = [
    {
        'name': 'bbcd',
        'classes': ["cells", "Platelets", "RBC", "WBC"],
        'train-ann-file': 'BBCD/train/_annotations.coco.json',
        'train-data-root': 'BBCD/train/',
        'val-ann-file': 'BBCD/valid/_annotations.coco.json',
        'val-data-root': 'BBCD/valid/',
        'test-ann-file': 'BBCD/test/_annotations.coco.json',
        'test-data-root': 'BBCD/test/'
    },
    {
        'name': 'pothole',
        'classes': ["pothole"],
        'train-ann-file': 'Pothole/train/_annotations.coco.json',
        'train-data-root': 'Pothole/train/',
        'val-ann-file': 'Pothole/valid/_annotations.coco.json',
        'val-data-root': 'Pothole/valid/',
        'test-ann-file': 'Pothole/test/_annotations.coco.json',
        'test-data-root': 'Pothole/test'
    },
    {
        'name': 'wildfire',
        'classes': ["smoke"],
        'train-ann-file': 'Wildfire_smoke/train/_annotations.coco.json',
        'train-data-root': 'Wildfire_smoke/train/',
        'val-ann-file': 'Wildfire_smoke/valid/_annotations.coco.json',
        'val-data-root': 'Wildfire_smoke/valid/',
        'test-ann-file': 'Wildfire_smoke/test/_annotations.coco.json',
        'test-data-root': 'Wildfire_smoke/test'
    },
    {
        'name': 'vitens-tiled',
        'classes': ["object"],
        'train-ann-file': 'vitens-tiled-coco/annotations/instances_train.json',
        'train-data-root': 'vitens-tiled-coco/images/train',
        'val-ann-file': 'vitens-tiled-coco/annotations/instances_val.json',
        'val-data-root': 'vitens-tiled-coco/images/val',
        'test-ann-file': 'vitens-tiled-coco/annotations/instances_val.json',
        'test-data-root': 'vitens-tiled-coco/images/val'
    },
    {
        'name': 'diopsis-tiled',
        'classes': ["object"],
        'train-ann-file': 'diopsis-tiled-coco/annotations/instances_train.json',
        'train-data-root': 'diopsis-tiled-coco/images/train',
        'val-ann-file': 'diopsis-tiled-coco/annotations/instances_val.json',
        'val-data-root': 'diopsis-tiled-coco/images/val',
        'test-ann-file': 'diopsis-tiled-coco/annotations/instances_val.json',
        'test-data-root': 'diopsis-tiled-coco/images/val'
    },
    {
        'name': 'weed-tiled',
        'classes': ["object"],
        'train-ann-file': 'weed-tiled-coco/annotations/instances_train.json',
        'train-data-root': 'weed-tiled-coco/images/train',
        'val-ann-file': 'weed-tiled-coco/annotations/instances_val.json',
        'val-data-root': 'weed-tiled-coco/images/val',
        'test-ann-file': 'weed-tiled-coco/annotations/instances_val.json',
        'test-data-root': 'weed-tiled-coco/images/val'
    },
    {
        'name': 'kbts_fish',
        'classes': ["fish"],
        'train-ann-file': 'kbts-fish-coco/annotations/instances_train.json',
        'train-data-root': 'kbts-fish-coco/images/train/',
        'val-ann-file': 'kbts-fish-coco/annotations/instances_val.json',
        'val-data-root': 'kbts-fish-coco/images/val/',
        'test-ann-file': 'kbts-fish-coco/annotations/instances_test.json',
        'test-data-root': 'kbts-fish-coco/images/test/'
    },
    {
        'name': 'pcd',
        'classes': ["normal", "abnormal"],
        'train-ann-file': 'pcd-coco/annotations/instances_train.json',
        'train-data-root': 'pcd-coco/images/train/',
        'val-ann-file': 'pcd-coco/annotations/instances_val.json',
        'val-data-root': 'pcd-coco/images/val/',
        'test-ann-file': 'pcd-coco/annotations/instances_val.json',
        'test-data-root': 'pcd-coco/images/val/',
    },
    {
        'name': 'aerial_large',
        'classes': ["boat", "car", "dock", "jetski", "lift"],
        'train-ann-file': 'Aerial_Maritime/large/train/_annotations.coco.json',
        'train-data-root': 'Aerial_Maritime/large/train/',
        'val-ann-file': 'Aerial_Maritime/large/valid/_annotations.coco.json',
        'val-data-root': 'Aerial_Maritime/large/valid/',
        'test-ann-file': 'Aerial_Maritime/large/test/_annotations.coco.json',
        'test-data-root': 'Aerial_Maritime/large/test/'
    },
    {
        'name': 'dice',
        'classes': ["1", "2", "3", "4", "5", "6"],
        'train-ann-file': 'Dice/train.json',
        'train-data-root': 'Dice/export/',
        'val-ann-file': 'Dice/valid.json',
        'val-data-root': 'Dice/export/',
        'test-ann-file': 'Dice/test.json',
        'test-data-root': 'Dice/export/',
    },
    {
        'name': 'minneapple',
        'classes': ["apple"],
        'train-ann-file': 'MinneApple/detection/train/train_coco.json',
        'train-data-root': 'MinneApple/detection/train/images/',
        'val-ann-file': 'MinneApple/detection/train/val_coco.json',
        'val-data-root': 'MinneApple/detection/train/images/',
        'test-ann-file': 'MinneApple/detection/train/test_coco.json',
        'test-data-root': 'MinneApple/detection/train/images/',
    },
    {
        'name': 'wgisd1',
        'classes': ["grape"],
        'train-ann-file': 'wgisd/train_1_class.json',
        'train-data-root': 'wgisd/data/',
        'val-ann-file': 'wgisd/test_1_class.json',
        'val-data-root': 'wgisd/data/',
        'test-ann-file': 'wgisd/test_1_class.json',
        'test-data-root': 'wgisd/data/',
    },
    {
        'name': 'wgisd5',
        'classes': ["CDY", "CFR", "CSV", "SVB", "SYH"],
        'train-ann-file': 'wgisd/train_5_classes.json',
        'train-data-root': 'wgisd/data/',
        'val-ann-file': 'wgisd/test_5_classes.json',
        'val-data-root': 'wgisd/data/',
        'test-ann-file': 'wgisd/test_5_classes.json',
        'test-data-root': 'wgisd/data/',
    },
    {
        'name': 'PCB_original',
        'classes': ["short", "spur", "spurious_copper", "missing_hole", "mouse_bite", "open_circuit"],
        'train-ann-file': 'PCB_ORiGINAL-coco/annotations/instances_train.json',
        'train-data-root': 'PCB_ORiGINAL-coco/images/',
        'val-ann-file': 'PCB_ORiGINAL-coco/annotations/instances_val.json',
        'val-data-root': 'PCB_ORiGINAL-coco/images/',
        'test-ann-file': 'PCB_ORiGINAL-coco/annotations/instances_test.json',
        'test-data-root': 'PCB_ORiGINAL-coco/images/',
    }
]

def collect_ap(path):
    """ Collects average precision values in log file. """

    average_precisions = []
    beginning = 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = '
    with open(path) as read_file:
        content = [line.strip() for line in read_file]
        for line in content:
            if line.startswith(beginning):
                average_precisions.append(float(line.replace(beginning, '')))
    return average_precisions


def calculate_train_time(work_dir):
    if not osp.exists(work_dir):
        return None

    log = [file for file in os.listdir(work_dir) if file.endswith('.log')]
    if not log:
        raise KeyError(f'{work_dir} has not log file')
    log_path = osp.join(work_dir, sorted(log)[-1])
    first_line, last_line = '', ''
    with open(log_path, 'r') as log_file:
        for line in log_file:
            if line.startswith('2021-'):
                line = line[:19]
                if first_line == '':
                    first_line = line
                else:
                    last_line = line

    FMT = '%Y-%m-%d %H:%M:%S'
    tdelta = (datetime.strptime(last_line, FMT) - datetime.strptime(first_line,
                                                                    FMT)).total_seconds() / 60
    return tdelta


def get_eval_command_line(subset, dataset, update_config, config_path, model_path):
    subset_path = "train.dataset" if subset == "train" else subset
    if subset in ['train', 'val']:
        split_update_config = update_config.replace("data.test.ann_file", "ann_file") \
            .replace("data.test.img_prefix", "img_prefix") \
            .replace(f"data.{subset_path}.ann_file", "data.test.ann_file") \
            .replace(f"data.{subset_path}.img_prefix", "data.test.img_prefix")
    else:
        split_update_config = update_config
    # avoid time-concuming validation on test part which is equal to val part
    if subset == 'test' and dataset['name'] in SETS_WITHOUT_TEST:
        return f'cp {model_path}/val {model_path}/test'
    if os.path.exists(model_path):
        return f'python /mmdetection/tools/test.py {config_path} {model_path}/latest.pth ' \
               f'--eval bbox {split_update_config} > {model_path}/{subset}'
    else:
        print(f'get_command_eval_line: {model_path} does not exist')
        return ''


def print_summarized_statistics(DATASETS, model_name, work_dir):
    '''Prints line that could be directly inserted in excel spreadsheet with this experiment
       (map and time statistics)'''
    names = []
    metrics = []
    for dataset in DATASETS:
        model_dir = f'{work_dir}/{model_name}_{dataset["name"]}'
        names.append(dataset['name'])
        for subset in ('train', 'val', 'test'):
            try:
                [map] = collect_ap(f'{model_dir}/{subset}')
                metrics.append(str(map))
                if map is None:
                    metrics.append('')
            except Exception as e:
                print(dataset['name'], subset, str(e))
                metrics.append('')
        try:
            # append empty time, since is not currently estimated
            training_time = calculate_train_time(model_dir)
            metrics.append(f'{training_time:.0f}')
        except Exception as e:
            metrics.append('')

    print(work_dir)
    print(','.join(names))
    print(','.join(metrics))


def parse_args():
    parser = argparse.ArgumentParser(description='Train model on different datasets')
    parser.add_argument('--experiment-nr', help='Number of experiment')
    parser.add_argument('--model', help='Model name')
    # parser.add_argument('--config', help='Model config file')
    # parser.add_argument('--data-root', type=str)
    parser.add_argument('--snapshot')
    parser.add_argument('--datasets', nargs='+', type=str, help='List of datasets to run experiments on')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--val', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    exp_dir = EXPERIMENT_DIR + f'/{args.experiment_nr}'
    # config_path = CONFIGS_DIR + f'/{args.config}'

    for dataset in DATASETS:
        if dataset['name'] in args.datasets:
            output_dir =  exp_dir + '/' + dataset['name']
            update_opts = f'opts ' \
                            f'output_dir {output_dir} ' \
                            f'num_classes {len(dataset["classes"])} ' \
                            f'data_dir {DATASETS_DIR} ' \
                            f'train_ann {dataset["train-ann-file"]} ' \
                            f'name_train {dataset["train-data-root"]} ' \
                            f'val_ann {dataset["val-ann-file"]} ' \
                            f'name_val {dataset["val-data-root"]} ' \
                            f'data_num_workers {args.gpus}'

            if args.train:
                # setting env variable as workaround to make dist training work
                #os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
                train_command_line = f'python tools/train.py -n {args.model} -d {args.gpus} -b {args.batch} -c {SNAPSHOTS_DIR}/{args.snapshot} -o {update_opts}'
                run(train_command_line, shell=True, check=True)
            if args.val:
                for subset in ['train', 'val', 'test']:
                    eval_command_line = get_eval_command_line(subset, dataset, update_opts,
                                                              args.model, output_dir)
                    run(eval_command_line, shell=True, check=True)

    # print_summarized_statistics(DATASETS, args.model, exp_dir)


if __name__ == '__main__':
    main()