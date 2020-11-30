"""
AICrowd FOOD Segmentation

An instance segmenation task runner for the AICrowd food dataset.
"""

import argparse

from foogseg.loader.download_dataset import download_dataset, fix_errors, print_data_description
from foogseg.loader.limit_classes import limit_files
from foogseg.engine.detectron import DetectronEngine
from foogseg.engine.backbone import Backbone
from foogseg.loader.cutout_images import cutout_images


if __name__ == '__main__':
    # Get the arguments from the command
    PARSER = argparse.ArgumentParser(conflict_handler='resolve',
                                     description='Setup Config for this run of Object Detection.')
    PARSER.add_argument('-i', '--iterations', type=int, default=5000,
                        help='number of iterations to run the model for')
    PARSER.add_argument('-c', '--class-count', type=int, default=300,
                        help='number of classes to limit the training dataset to (top frequency)')
    PARSER.add_argument('-w', '--work-dir', type=str, default='scratch',
                        help='root working directory to store data and models')
    PARSER.add_argument('-b', '--batch-size', type=int, default=128,
                        help='batch size for the model (memory-speed tradeoff)')
    PARSER.add_argument('-r', '--resume', action='store_const', const=True, default=False,
                        help='resume from current model stored in output directory?')
    ARGS = PARSER.parse_args()

    # Download the dataset
    download_dataset(ARGS.work_dir)
    print_data_description(ARGS.work_dir)
    fix_errors(ARGS.work_dir, 'train')
    fix_errors(ARGS.work_dir, 'val')

    # Limit the number of classes we are training the models on
    limit_files(ARGS.work_dir, 'train', ['train', 'val'], ARGS.class_count)

    #The model gets saved as model_final.pth inside cfg.OUTPUT_directory path
    ENGINE = DetectronEngine(ARGS.work_dir, ARGS.batch_size, ARGS.class_count)
    if ARGS.resume:
        print('Attempting to resume training the model from the current state.')

    # Pretrain the backbone
    cutout_images(ARGS.work_dir, 'train')
    cutout_images(ARGS.work_dir, 'val')
    cutout_images(ARGS.work_dir, 'test')
    BACKBONE = Backbone(ENGINE.trainer)
    BACKBONE.fit(20, ARGS.work_dir)

    # Train the thing
    ENGINE.train(iterations=ARGS.iterations, resume=ARGS.resume)
