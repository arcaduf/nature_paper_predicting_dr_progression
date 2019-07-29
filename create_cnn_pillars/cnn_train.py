'''
Train a convolutional neural network.
This script is entirely based on the function contained in < cnn_class.py >.
'''


# Author: Filippo Arcadu 
#         28/01/2019




from __future__ import division , print_function
import argparse
import sys , os , glob
import time
import numpy as np


from cnn_class import CNN




# =============================================================================
# Parsing input arguments
# =============================================================================

def _examples():
    print( '\n\nEXAMPLES:\n\nQuick test for binary classification:\n"""\n' \
           'python cnn_train.py -i1 /pstore/data/pio/Tasks/PIO-40/tests/cnn/binary_classification/data/kaggledr_2classes_dataset_small_constr-id_train.csv -i2 /pstore/data/pio/Tasks/PIO-40/tests/cnn/binary_classification/data/kaggledr_2classes_dataset_small_constr-id_valid.csv -p /pstore/data/pio/Tasks/PIO-40/tests/cnn/binary_classification/configs/cnn_config.yml -o /pstore/data/pio/Tasks/PIO-40/tests/cnn/binary_classification/outputs/ -col-imgs filepath -col-label label\n"""'
            '\n\nQuick test on for multi-label classification:\n"""\n'
            'python cnn_train.py -i1 /pstore/data/pio/Tasks/PIO-40/tests/cnn/multilabel_classification/data/kaggledr_6classes_dataset_small_constr-id_train.csv -i2 /pstore/data/pio/Tasks/PIO-40/tests/cnn/multilabel_classification/data/kaggledr_6classes_dataset_small_constr-id_valid.csv -p /pstore/data/pio/Tasks/PIO-40/tests/cnn/multilabel_classification/configs/cnn_config.yml -o /pstore/data/pio/Tasks/PIO-40/tests/cnn/multilabel_classification/outputs/ -col-imgs filepath -col-label label\\n"""'
           '\n\nQuick test on for regression:\n"""\n'
            'python cnn_train.py -i1 /pstore/data/pio/Tasks/PIO-40/tests/cnn/regression/data/ride_rise_letters_small_constr-PATNUM_train.csv -i2 /pstore/data/pio/Tasks/PIO-40/tests/cnn/regression/data/ride_rise_letters_small_constr-PATNUM_valid.csv -p /pstore/data/pio/Tasks/PIO-40/tests/cnn/regression/configs/cnn_config.yml -o /pstore/data/pio/Tasks/PIO-40/tests/cnn/regression/outputs/ -col-imgs full_fname_png -col-label LETTERS\\n"""\n\n'
            )    

          
          
def _get_args():
    parser = argparse.ArgumentParser(    
                                        prog='cnn_train',
                                        description='Train Convolutional Neural Network',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                        ,add_help=False
                                    )
    
    parser.add_argument('-i1', '--csv_train', dest='csv_train',
                        help='Specify training CSV')
                        
    parser.add_argument('-i2', '--csv_valid', dest='csv_valid',
                        help='Specify validation CSV')                        
    
    parser.add_argument('-col-label', dest='label_id_col', nargs='+' ,
                        help='Select name of column containing labels' )

    parser.add_argument('-col-imgs', dest='file_id_col', nargs='+' ,
                        help='Select name of the columns containing the image filenames' )

    parser.add_argument('-o', '--path_out', dest='path_out', default='./',
                        help='Save model')  
    
    parser.add_argument('-p', '--config_file', dest='config_file',
                        help='Select parameter file .yml' ) 

    parser.add_argument('-l', '--add_label', dest='add_label',
                        help='Select additional label to output files to facilitate their identification' )                 

    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Debugging mode: set number of epochs and steps per epoch to 1' )    
                        
    parser.add_argument('-h', '--help', dest='help', action='store_true',
                        help='Print help and examples')                            

    args = parser.parse_args()
    
    if args.help is True:
        parser.print_help()
        _examples()
        sys.exit()

    if args.csv_train is None:
        parser.print_help()
        sys.exit('\nERROR: Input training CSV not specified!\n')
        
    if os.path.isfile( args.csv_train ) is False:
        parser.print_help()
        sys.exit('\nERROR: Input training CSV ' + args.csv_train + ' does not exist!\n')        
    if args.csv_valid is None:
        parser.print_help()
        sys.exit('\nERROR: Input validation CSV not specified!\n')    
        
    if os.path.isfile( args.csv_valid ) is False:
        parser.print_help()
        sys.exit('\nERROR: Input validation CSV ' + args.csv_valid + ' does not exist!\n')    
        
    if args.config_file is None:
        parser.print_help()
        sys.exit('\nERROR: Config file .yml not specified!\n')

    if os.path.isfile( args.config_file ) is False:
        parser.print_help()
        sys.exit('\nERROR: Config file .yml does not exist!\n')        
    
    if args.label_id_col is None:
        parser.print_help()
        sys.exit('\nERROR: Name of column containing the labels has not been specified!\n')
    else:
        if len( args.label_id_col ) > 1:
            args.label_id_col = ' '.join( args.label_id_col )
        else:
            args.label_id_col = args.label_id_col[0]

    if args.file_id_col is None:
        parser.print_help()
        sys.exit('\nERROR: Name of column containing the image filenames has not been specified!\n')
    else:
        if len( args.file_id_col ) > 1:
            args.file_id_col = ' '.join( args.file_id_col )
        else:
            args.file_id_col = args.file_id_col[0]

    return args




# =============================================================================
# Main
# =============================================================================

def main():
    # Get input arguments
    time1 = time.time()
    args = _get_args()
    

    # Debug mode if enabled
    if args.debug:
        print( '\n' )
        print( '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' )
        print( '                D E B U G  M O D E              ' )
        print( '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' )
        print( '\n' )

        
    # Initial prints
    print( '\n#######################################' )
    print(   '#######################################' )
    print(   '####                               ####' )
    print(   '####         TRAIN CNN MODEL       ####' )
    print(   '####                               ####' )    
    print(   '#######################################' )
    print(   '#######################################' )

    print( '\nTraining CSV: ', args.csv_train )
    print( 'Validation CSV: ', args.csv_valid )
    print( 'Config file: ', args.config_file )
    print( 'Data column: ', args.file_id_col )
    print( 'Label column: ', args.label_id_col )
    print( 'Output path: ', args.path_out )
    print( 'Additional output label: ', args.add_label )


    # Load model
    print( '\nLoading CNN model ....' )
    cnn = CNN( args.csv_train ,
               args.csv_valid ,
               args.config_file ,
               args.file_id_col   ,
               args.label_id_col  ,
               pathout   = args.path_out ,
               add_label = args.add_label,
               debug     = args.debug    )
               
               
    # Some prints
    print( '\nDATASET:' )
    print( 'Classes: ', cnn._class_names )
    print( 'Number of training samples: ', cnn._n_imgs_train )
    print( 'Number of validation samples: ', cnn._n_imgs_valid )
    print( 'Images have ', cnn._n_channels,' channels and extension ', cnn._file_ext )
    
    for i in range( cnn._n_classes ):
        print( 'Number of samples in class ', cnn._class_names[i],
               ' is ', cnn._n_class_elements[i] )

    print( '\nSETTING:' )
    print( 'Task: ', cnn._task )
    print( 'Mode: ', cnn._mode )
    print( 'Model: ', cnn._model_id )
    print( 'Model input shape: ', cnn._model_input_shape )
    print( 'Starting weights from: ', cnn._weights_start )
    print( 'Monitoring metric: ', cnn._metric )
    print( 'Number of epochs: ', cnn._n_epochs )
    print( 'Optimizer: ', cnn._optimizer_type )
    print( 'Loss function: ', cnn._loss )
    print( 'Batch size: ', cnn._batch_size )
    print( 'Training steps per epoch: ', cnn._steps_per_epoch_train )
    print( 'Validation steps per epoch: ', cnn._steps_per_epoch_valid )    
    print( 'Class weights: ', cnn._class_weight )
    
    if cnn._loss == 'wcce':
        print( 'WCCE weights:\n', cnn._wcce_weight )
    
    
    # Option 1: learning model from scratch
    if cnn._mode == 'learning':
        print( '\nLEARNING FROM SCRATCH (LS):' )
        print( 'Number of epochs: ', cnn._num_epochs )
        print( 'Optimizer type: ', cnn._optimizer_type )
        print( 'Optimizer parameters: ', cnn._optimizer_param_print )
        
        print( '\nLearning CNN model ....' )
        cnn._learning()
        print( '\n.... learning completed!' )

    
    # Option 2: only transfer-learning    
    if cnn._mode == 'transfer-learning' or cnn._mode == 'fine-tuning':
        print( '\nTRANSFER LEARNING (TL):' )
        print( 'Number of epochs: ', cnn._n_epochs )
        print( 'Optimizer type: ', cnn._optimizer_type )
        print( 'Optimizer parameters: ', cnn._optimizer_param_print )
        
        print( '\nTransfer learning CNN model ....' )
        cnn._transfer_learning()
        print( '\n.... transfer learning completed!' )        
    
    
    # Option 3: fine-tuning following transfer-learning
    if cnn._mode == 'fine-tuning':
        print( '\nFINE TUNING (FT):' )
        print( 'Number of epochs: ', cnn._n_epochs_ft )
        print( 'Optimizer type: ', cnn._optimizer_type_ft )
        print( 'Optimizer parameters: ', cnn._optimizer_param_print_ft )
        
        print( '\nFine tuning CNN model ....' )
        cnn._fine_tuning()
        print( '\n.... fine tuning completed!' )

    
    # Option 4: stepwise re-training
    if cnn._mode == 're-training':
        print( '\nRE-TRAINING (RT):' )
        print( 'Number of re-training steps: ', cnn._num_steps_rt )
        print( 'Number of epochs: ', cnn._n_epochs_rt )
        print( 'Optimizer type: ', cnn._optimizer_type )
        print( 'Optimizer parameters: ', cnn._optimizer_param_print )
        
        print( '\nRe-training CNN model ....' )
        cnn._re_training()
        print( '\n.... re-training completed!' )
 
    
    # Get elapsed time
    diff = time.time() - time1    
    print( '\nTime elapsed: ', diff )
    
    
    # Write output files
    print( '\nWriting output files ...' )
    cnn._write_outputs( diff )
               
    print( '\nCNN model training ended successfully!\n\n' )
    
    
    # Debug mode if enabled
    if args.debug:
        print( '\n' )
        print( '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' )
        print( '                D E B U G  M O D E              ' )
        print( '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' )
        print( '\n' )    
    
    
    
# =============================================================================
# Call to main
# =============================================================================

if __name__ == '__main__':
    main()
