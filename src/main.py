"""
AICrowd FOOD Segmentation

An instance segmenation task runner for the AICrowd food dataset.
"""

import argparse

from loader.download_dataset import download_dataset, fix_errors, print_data_description
from loader.limit_classes import limit_files
from engine.detectron import setup_config, train


if __name__ == '__main__':
    # Get the arguments from the command
    parser = argparse.ArgumentParser(conflict_handler='resolve',
                                     description='Setup Config for this run of Object Detection.')
    parser.add_argument('-i', '--iterations', type=int, default=5000,
                        help='number of iterations to run the model for')
    parser.add_argument('-c', '--class-count', type=int, default=300,
                        help='number of classes to limit the training dataset to (top frequency)')
    parser.add_argument('-w', '--work-dir', type=str, default='scratch',
                        help='root working directory to store data and models')
    parser.add_argument('-b', '--batch-size', type=int, default=128,
                        help='batch size for the model (memory-speed tradeoff)')
    parser.add_argument('-r', '--resume', action='store_const', const=True, default=False,
                        help='resume from current model stored in output directory?')
    args = parser.parse_args()

    # Download the dataset
    download_dataset(args.work_dir)
    print_data_description(args.work_dir)
    fix_errors(args.work_dir, 'train')
    fix_errors(args.work_dir, 'val')

    # Limit the number of classes we are training the models on
    limit_files(args.work_dir, 'train', ['train', 'val'], args.class_count)

    #The model gets saved as model_final.pth inside cfg.OUTPUT_directory path
    setup_config(args.iterations, args.work_dir, args.batch_size, args.class_count)
    if args.resume:
        print('Attempting to resume training the model from the current state.')
    train(resume=args.resume)
