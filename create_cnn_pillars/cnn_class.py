'''
CNN Training Class
'''


# Author: Filippo Arcadu 
#         14/01/2018




from __future__ import print_function , division
import os , sys , glob
import yaml , json , collections
import pandas as pd
import time
import numpy as np
from skimage import io , transform
import random
import numbers

from keras.models import load_model
from keras import backend as K
from keras import utils as kutils
from keras.preprocessing import image as kimage
from keras.callbacks import Callback
from keras.utils import to_categorical

import tensorflow as tf

from progressbar import ProgressBar




# =============================================================================
# Set tensorflow tensor ordering for Keras 
# =============================================================================

K.set_image_dim_ordering( 'tf' )




# =============================================================================
# Modules needed to compute various metrics 
# =============================================================================

import wcce
import metrics as me




# =============================================================================
# List of possible tasks
# =============================================================================

TASKS = [ 'classification' , 'regression' ]




# =============================================================================
# List of possible modes of use 
# =============================================================================

MODES = [ 'learning' , 'transfer-learning' , 'fine-tuning'  , 're-training']




# =============================================================================
# Available architectures 
# =============================================================================

KERAS_MODELS = [ 'vgg16' , 'vgg19' , 'inception-v3' , 'xception' , 
                 'inception-resnet-v2' , 'resnet50' , 'mobilenet' ,
                 'mobilenet-v2' , 'densenet-121' , 'densenet-169' ,
                 'densenet-201', 'nasnet-mobile' ]
                 
                 
                 

# =============================================================================
# List of optimizers available in Keras 
# =============================================================================

OPTIMIZERS = [ 'sgd' , 'rmsprop' , 'adagrad' , 'adadelta' , 'adam' ,
               'adamax' , 'nadam' ]




# =============================================================================
# List of image formats that can be used input 
# =============================================================================

IMG_FORMATS = ( 'png' , 'jpg' , 'jpeg' )




# =============================================================================
# Metrics that can be selected to dave the best model during training 
# =============================================================================

METRICS = [ 'accuracy' , 'precision' , 'recall' , 'f1score' , 'auc' , 
            'cohen_kappa' , 'r2' , 'mse' ]




# =============================================================================
# Numbers for debugging 
# =============================================================================

NUM_DEBUG_0 = 1
NUM_DEBUG_1 = 2
NUM_DEBUG_2 = 5




# =============================================================================
# Types of variables 
# =============================================================================

myfloat  = np.float32
myfloat2 = np.float64
myint    = np.int




# =============================================================================
# Select separator to write CSV 
# =============================================================================

SEP = ','



        
# =============================================================================
# Class CNN
# It contains all the functions needed to train a 
# convolutional neural network
# =============================================================================

class CNN:
    
    # ===================================
    # Init 
    # ===================================
    
    def __init__( self           , 
                  csv_train      , 
                  csv_valid      , 
                  config_file    ,
                  col_imgs       ,
                  col_label      ,
                  pathout='./'   , 
                  add_label=None ,
                  debug=False    ):

        # Assign entries to class fields
        self._csv_train   = csv_train
        self._csv_valid   = csv_valid
        self._pathout     = self._create_path( pathout )
        self._config_file = config_file
        self._col_imgs    = col_imgs
        self._col_label   = col_label
        self._add_label   = add_label
        self._debug       = debug
 
       
        # Create hashed config file
        self._hash_config_file()

        
        # Get dataset info
        self._get_dataset()
        
        
        # Load config file
        self._load_config()

        
        # Get dataset info
        self._compute_class_weights()
 
        
        # Get steps per epoch
        self._get_steps_per_epoch()
        
        
        # Load model
        self._load_model()


        # Init training monitoring
        self._init_monitoring()


    
    # ===================================
    # Create path 
    # ===================================
    
    def _create_path( self , path ):
        if os.path.isdir( path ) is False:
            os.mkdir( path )

        path = os.path.abspath( path )
                
        return path
 
    
    
    # ===================================
    # Hashing config file
    # ===================================
    
    def _hash_config_file( self ):
        self._output_hash        = str( random.getrandbits( 128 ) )
        filename                 = self._config_file
        filename                 = os.path.basename( filename )    
        self._config_file_hashed = os.path.join( self._pathout , filename[:len(filename)-4] ) + '_' + self._output_hash + '.yml'
        command                  = 'cp ' + self._config_file + ' ' + self._config_file_hashed
        os.system( command )
        print( 'Hashed config file:\n' , command )
        
        
        
    # ===================================
    # Get input dataset
    # ===================================
    
    def _get_dataset( self ):
        # Get number of images for training and validation
        self._df_train     = self._read_table( self._csv_train )
        self._imgs_train   = self._df_train[ self._col_imgs ].values
        self._y_train      = self._df_train[ self._col_label ].values
        self._n_imgs_train = len( self._imgs_train )
        self._ind_train    = np.arange( self._n_imgs_train )
        
        self._df_valid     = self._read_table( self._csv_valid )
        self._imgs_valid   = self._df_valid[ self._col_imgs ].values
        self._y_valid      = self._df_valid[ self._col_label ].values
        self._n_imgs_valid = len( self._imgs_valid )
    
    
        # Get file extension
        ind            = random.choice( self._ind_train )
        file_sel       = self._imgs_train[ind]
        self._file_ext = os.path.basename( file_sel ).split( '.' )[1]
        
        
        # Get number of channels
        img              = io.imread( file_sel )
        self._n_channels = img.ndim

        
    
    # ===================================
    # Read CSV table
    # ===================================
 
    def _read_table( self , file_csv ):
        df = pd.read_csv( file_csv , sep=SEP )

        if df.shape[1] == 1:
            df = pd.read_csv( file_csv , sep=';' )

        if df.shape[1] == 1:
            raise Warning( '\nData frame shape of ', file_csv,' is: ', df.shape )

        return df
            
   

    # ===================================
    # Load YAML config file
    # ===================================

    def _load_config( self ):

        # Import parameter file as yaml
        with open( self._config_file_hashed , 'r' ) as ymlfile:
            cfg = yaml.load( ymlfile )
        
        
        # Assign "MODEL" attributes to class fields
        self._task            = self._cast_parameter( cfg['model']['task'] , format='string' )
        self._mode            = self._cast_parameter( cfg['model']['mode'] , format='string' )
        self._model_id        = self._cast_parameter( cfg['model']['model'] , format='string' )
        self._metric          = self._cast_parameter( cfg['model']['metric'] , format='string' )
        self._loss            = self._cast_parameter( cfg['model']['loss'] , format='string' )
        self._n_epochs        = self._cast_parameter( cfg['model']['num_epochs'] , format='int' )
        self._optimizer_type  = self._cast_parameter( cfg['model']['optimizer_type'] , format='string' )
        self._num_nodes_dense = self._cast_parameter( cfg['model']['num_nodes_dense'] , format='int' )
        self._batch_size      = self._cast_parameter( cfg['model']['batch_size'] , format='int' )
        self._class_weight    = self._cast_parameter( cfg['model']['class_weight'] , format='dict' )
        self._wcce_weight     = self._cast_parameter( cfg['model']['wcce_weight'] , format='dict' )        
        self._weights_start   = self._cast_parameter( cfg['model']['weights_start'] , format='string' )


        # Assign "FINE-TUNING" attributes to class fields
        self._n_epochs_ft          = self._cast_parameter( cfg['fine_tuning']['num_epochs_ft'] , format='int' )
        self._optimizer_type_ft    = self._cast_parameter( cfg['fine_tuning']['optimizer_type_ft'] , format='string' )
        self._num_layers_freeze_ft = self._cast_parameter( cfg['fine_tuning']['num_layers_freeze_ft'] , format='int' )
          
        
        # Assign "RE-TRAINING" attributed to class fields
        self._num_steps_rt           = self._cast_parameter( cfg['re_training']['num_steps_rt'] , format='int' ) 
        self._n_epochs_rt            = self._cast_parameter( cfg['re_training']['num_epochs_rt'] , format='int' )
        self._num_layers_unfreeze_rt = self._cast_parameter( cfg['re_training']['num_layers_unfreeze_rt'] , format='int' )
 

        # Get number of classes
        self._assess_label()


        # Check mode
        self._check_task_mode_arch()
        
        
        # Check metric to save best model
        self._check_metric()
        
        
        # Check whether start weights exist
        self._check_weights_start()

        
        # Check WCCE weights
        self._check_wcce_weights()
        
        
        # Set number of epochs to 1 if in debug mode
        #if self._debug:
        #    self._n_epochs = self._n_epochs_ft = self._n_epochs_rt = NUM_DEBUG_0
            
        
        # Set optimizer for transfer learning and fine tuning
        self._check_optimizer_type()
        
        self._set_optimizer( cfg , key=None )
        
        if self._mode == 'fine-tuning':
            self._set_optimizer( cfg , key='ft' )

            

    # ===================================
    # Cast parameter
    # ===================================
    
    def _cast_parameter( self , entry , format='int' ):
        if entry is not None and entry != 'None':
            if format == 'int':
                return myint( myfloat( entry ) )    
                
            elif format == 'float':
                return myfloat( entry )
            
            elif format == 'boolean':
                return entry
                
            elif format == 'string':
                return entry
                
            else:
                return None
        
        else:
            return None


    
    # ===================================
    # Assess label
    # ===================================
 
    def _assess_label( self ):
        if self._task == 'classification':
            # Collect all labels
            labels = np.array( self._y_train.tolist() + \
                               self._y_valid.tolist() )


            # Get number of classes
            labels_unique     = np.unique( labels ) 
            self._n_classes   = len( labels_unique )
            self._class_names = labels_unique.copy()

        else:
            self._n_classes   = 0
            self._class_names = None



    # ===================================
    # Check task and mode
    # ===================================
    
    def _check_task_mode_arch( self ):
        # Check task
        if self._task not in TASKS:
             sys.exit( '\nERROR ( CNN -- _check_mode ): selected task ' + self._task + ' is not available!\nSelect among ' + ','.join( TASKS ) + '\n\n' )
       
        
        # Check mode
        if self._mode not in MODES:
            sys.exit( '\nERROR ( CNN -- _check_mode ): selected mode ' + self._mode + ' is not available!\nSelect among ' + ','.join( MODES ) + '\n\n' )

        
        # Check architecture
        if self._model_id not in KERAS_MODELS:
            sys.exit( '\nERROR ( CNN -- _check_mode ): selected architecture ' + self._model_id + ' is not available!\nSelect among ' + ','.join( KERAS_MODELS ) + '\n\n' )

   
    
    # ===================================
    # Check if starting model exists
    # ===================================
    
    def _check_weights_start( self ):
        if self._weights_start != 'imagenet' and self._weights_start is not None:
            if os.path.isfile( self._weights_start ) is False:
                sys.exit( '\nERROR (CNN -- _load_config ): input HDF5 ' + self._weights_start + ' does not exist!\n' )
         
        elif self._weights_start is None:
            self._weights_start = 'imagenet'

      

    # ===================================
    # Check metric to save best model
    # ===================================
    
    def _check_metric( self ):
        if self._metric not in METRICS:
            sys.warning( '\nWarning ( CNN -- _load_config ): metric ' + self._metric + \
                         'is not available among the list ' + ' '.join( METRICS ) + '!' + \
                         '\nValidation accuracy will be used in place of the selected metric' )
            self._metric = 'accuracy'
            
        if self._n_classes > 2 and self._metric in [ 'precision' , 'recall' , 'f1score' , 'auc' , 'sensitivity' , 'specificity' ]:
            sys.warning( '\nWarning ( CNN -- _load_config ): metric ' + self._metric + \
                         ' cannot be used with ' + str( self._n_classes ) + ' classes!' + \
                         '\nCohen kappa will be used in place of the selected metric' )
            self._metric = 'cohen_kappa'
            
    
    
    # ===================================
    # Check WCCE weights
    # ===================================
    
    def _check_wcce_weights( self ):
        self._wcce_weight = None
        if self._wcce_weight is not None:
            if np.array( self._wcce_weight ).shape != ( self._n_classes , self._n_classes ):
                print('Class weight shape ACTUAL: ', np.array( self._wcce_weight ).shape)
                print('Class weight shape SHOULD: ', ( self._n_classes , self._n_classes ))
                sys.exit('\nERROR: Class weights do not match number of classes in training data!\n') 
        else:
            self._wcce_weight = wcce._ones_square_np( self._n_classes )
            
    
    
    # ===================================
    # Check type of optimizer
    # ===================================

    def _check_optimizer_type( self ):
        if self._optimizer_type.lower() not in OPTIMIZERS:
            sys.exit( '\nERROR: selected optimizer for transfer learning "' + \
                        self._optimizer_type + '" is not supported by Keras!\n' + \
                        'Select among' + ','.join( OPTIMIZERS ) + '\n' )
       
        if self._mode == 'fine-tuning':
            if self._optimizer_type_ft.lower() not in OPTIMIZERS:
                sys.exit( '\nERROR: selected optimizer for fine tuning "' + \
                            self._optimizer_type_ft + '" is not supported by Keras!\n' + \
                            'Select among ' + ','.join( OPTIMIZERS ) + '\n' )



    # ===================================
    # Set optimizer
    # ===================================
    
    def _set_optimizer( self , cfg , key=None ):
        # Initialize list for printing
        list_param = []
    

        # Set key and get type of optimizer
        if key != 'ft':
            key_opt        = 'optimizer'
            optimizer_type = cfg['model']['optimizer_type'].lower()
        
        elif key == 'ft':
            key_opt        = 'optimizer_fine_tuning'
            optimizer_type = cfg['fine_tuning']['optimizer_type_ft'].lower()


        # Case optimizer SGD
        if optimizer_type == 'sgd':
            lr       = self._cast_parameter( cfg[key_opt]['param_sgd']['learning_rate'] , format='float' )
            momentum = self._cast_parameter( cfg[key_opt]['param_sgd']['momentum'] , format='float' )
            decay    = self._cast_parameter( cfg[key_opt]['param_sgd']['decay'] , format='float' )
            nesterov = self._cast_parameter( cfg[key_opt]['param_sgd']['nesterov'] , format='float' )
            
            list_param.append( [ lr , momentum , decay , nesterov ] )                                 
            
            from keras.optimizers import SGD
            optimizer = SGD( lr       = lr ,
                             momentum = momentum ,
                             decay    = decay ,
                             nesterov = nesterov )
                             
                             
        # Case optimizer RMSprop
        elif optimizer_type == 'rmsprop':
            lr       = self._cast_parameter( cfg[key_opt]['param_rmsprop']['learning_rate'] , format='float' )
            rho      = self._cast_parameter( cfg[key_opt]['param_rmsprop']['rho'] , format='float' )
            epsilon  = self._cast_parameter( cfg[key_opt]['param_rmsprop']['epsilon'] , format='float' )
            decay    = self._cast_parameter( cfg[key_opt]['param_rmsprop']['decay'] , format='float' )

            list_param.append( [ lr , rho , epsilon , decay ] )
            
            from keras.optimizers import RMSprop
            optimizer = RMSprop( lr      = lr ,
                                 rho     = rho ,
                                 epsilon = epsilon , 
                                 decay   = decay )
                                 

        # Case optimizer Adagrad
        elif optimizer_type == 'adagrad':
            lr      = self._cast_parameter( cfg[key_opt]['param_adagrad']['learning_rate'] , format='float' )
            epsilon = self._cast_parameter( cfg[key_opt]['param_adagrad']['epsilon'] , format='float' )
            decay   = self._cast_parameter( cfg[key_opt]['param_adagrad']['decay'] , format='float' )            
            
            list_param.append( [ lr , epsilon , decay ] )
                            
            from keras.optimizers import Adagrad
            optimizer = Adagrad( lr      = lr ,
                                 epsilon = epsilon ,
                                 decay   = decay )

                                 
        # Case optimizer Adadelta
        elif optimizer_type == 'adadelta':
            lr      = self._cast_parameter( cfg[key_opt]['param_adadelta']['learning_rate'] , format='float' )
            rho     = self._cast_parameter( cfg[key_opt]['param_adadelta']['rho'] , format='float' )
            epsilon = self._cast_parameter( cfg[key_opt]['param_adadelta']['epsilon'] , format='float' )
            decay   = self._cast_parameter( cfg[key_opt]['param_adadelta']['decay'] , format='float' )
        
            list_param.append( [ lr , rho , epsilon , decay ] )

            from keras.optimizers import Adadelta
            optimizer = Adadelta( lr      = lr ,
                                  rho     = rho ,
                                  epsilon = epsilon ,
                                  decay   = decay )


        # Case optimizer Adam
        elif optimizer_type == 'adam':
            lr      = self._cast_parameter( cfg[key_opt]['param_adam']['learning_rate'] , format='float' )
            beta_1  = self._cast_parameter( cfg[key_opt]['param_adam']['beta_1'] , format='float' )
            beta_2  = self._cast_parameter( cfg[key_opt]['param_adam']['beta_2'] , format='float' )
            epsilon = self._cast_parameter( cfg[key_opt]['param_adam']['epsilon'] , format='float' )
            decay   = self._cast_parameter( cfg[key_opt]['param_adam']['decay'] , format='float' )
            
            list_param.append( [ lr , beta_1 , beta_2 , epsilon , decay ] )
            
            from keras.optimizers import Adam    
            optimizer = Adam( lr      = lr , 
                              beta_1  = beta_1 ,
                              beta_2  = beta_2 ,
                              epsilon = epsilon ,
                              decay   = decay )

                              
        # Case optimizer Adamax
        elif optimizer_type == 'adamax':
            lr      = self._cast_parameter( cfg[key_opt]['param_adamax']['learning_rate'] , format='float' )
            beta_1  = self._cast_parameter( cfg[key_opt]['param_adamax']['beta_1'] , format='float' )
            beta_2  = self._cast_parameter( cfg[key_opt]['param_adamax']['beta_2'] , format='float' )
            epsilon = self._cast_parameter( cfg[key_opt]['param_adamax']['epsilon'] , format='float' )
            decay   = self._cast_parameter( cfg[key_opt]['param_adamax']['decay'] , format='float' )

            list_param.append( [ lr , beta_1 , beta_2 , epsilon , decay ] )
            
            from keras.optimizers import Adamax
            optimizer = Adamax( lr      = lr ,
                                beta_1  = beta_1 ,
                                beta_2  = beta_2 ,
                                epsilon = epsilon ,
                                decay   = decay )


        # Case optimizer Nadam
        elif optimizer_type == 'nadam':
            lr             = self._cast_parameter( cfg[key_opt]['param_nadam']['learning_rate'] , format='float' )
            beta_1         = self._cast_parameter( cfg[key_opt]['param_nadam']['beta_1'] , format='float' )
            beta_2         = self._cast_parameter( cfg[key_opt]['param_nadam']['beta_2'] , format='float' )
            epsilon        = self._cast_parameter( cfg[key_opt]['param_nadam']['epsilon'] , format='float' )
            schedule_decay = self._cast_parameter( cfg[key_opt]['param_nadam']['schedule_decay'] , format='float' )

            list_param.append( [ lr , beta_1 , beta_2 , epsilon , schedule_decay ] )

            from keras.optimizers import Nadam
            optimizer = Nadam( lr             = lr ,
                               beta_1         = beta_1 ,
                               beta_2         = beta_2 ,
                               epsilon        = epsilon ,
                               schedule_decay = schedule_decay )


        # Return to class
        if key != 'ft':
            self.optimizer_type          = optimizer_type
            self._optimizer_param_print  = list_param            
            self._optimizer              = optimizer
            self._reduce_lr_rate_min     = lr / 100.0
        
        elif key == 'ft':
            self.optimizer_type_ft         = optimizer_type
            self._optimizer_param_print_ft = list_param            
            self._optimizer_ft             = optimizer  
            self._reduce_lr_rate_min       = lr / 100.0        



    # ===================================
    # Compute class weights
    # ===================================  

    def _compute_class_weights( self ): 
        # Case A: classification
        if self._task != 'regression':
            # Combine labels
            labels = np.array( self._y_train.tolist() + self._y_valid.tolist() )

            # Get number of elements per class
            self._n_class_elements = []

            for i in range( self._n_classes ):
                n_class_elements = len( labels[ labels == self._class_names[i] ] )
                self._n_class_elements.append( n_class_elements )


            # Create dictionary of class weights
            nmax    = np.max( self._n_class_elements )
            weights = nmax * 1.0 / self._n_class_elements
            dict    = {}
        
            for i in range( self._n_classes ):
                dict[i] = weights[i]
            
            self._class_weights = dict


        # Case B: regression
        else:
            self._class_weights = None
    


    # ===================================
    # Get steps per epoch
    # ===================================
    
    def _get_steps_per_epoch( self ):
        # Steps per epoch for training
        if self._debug:
            self._steps_per_epoch_train = NUM_DEBUG_1
        else:
            self._steps_per_epoch_train = np.int( ( self._n_imgs_train * 1.0 ) / self._batch_size )

        # Steps per epoch for validation
        if self._debug:
            self._steps_per_epoch_valid = NUM_DEBUG_1
        else:
            self._steps_per_epoch_valid = self._n_imgs_valid

        # Check
        if self._steps_per_epoch_train < 1:
            self._steps_per_epoch_train = 1
            
        if self._steps_per_epoch_valid < 1:
            self._steps_per_epoch_valid = 1
            


    # ===================================
    # Load model
    # ===================================
    
    def _load_model( self ):
        # Set starting weights
        if self._mode == 'learning':
            weights = None
        else:
            weights = self._weights_start


        # VGG-16
        if self._model_id == 'vgg16':
            from keras.applications.vgg16 import VGG16 , preprocess_input
            self._input_shape = ( 224 , 224 , 3 )
                
            if weights is None or weights == 'imagenet':
                self._model = VGG16( weights     = weights , 
                                     include_top = False   , 
                                     input_shape = self._input_shape )
            else:
                self._model = load_model( weights )
                
            self._preproc = preprocess_input 
                
            
        # VGG-19
        elif self._model_id == 'vgg19':
            from keras.applications.vgg19 import VGG19 , preprocess_input
            self._input_shape = ( 224 , 224 , 3 )
                
            if weights is None or weights == 'imagenet':
                    self._model = VGG19( weights     = weights , 
                                         include_top = False   , 
                                         input_shape = self._input_shape )
            else:
                self._model = load_model( weights )

            self._preproc = preprocess_input
                

        # INCEPTION-V3
        elif self._model_id == 'inception-v3':
            from keras.applications.inception_v3 import InceptionV3 , preprocess_input
            self._input_shape = ( 299 , 299 , 3 )

            if weights is None or weights == 'imagenet':                
                self._model = InceptionV3( weights     = weights , 
                                           include_top = False   ,
                                           input_shape = self._input_shape )
            else:
                self._model = load_model( weights )
                
            self._preproc = preprocess_input
        
            
        # XCEPTION
        elif self._model_id == 'xception':
            from keras.applications.xception import Xception , preprocess_input
            self._input_shape = ( 299 , 299 , 3 )

            if weights is None or weights == 'imagenet':                    
                self._model = Xception( weights     = weights ,
                                        include_top = False , 
                                        input_shape = self._input_shape )
            else:
                self._model = load_model( weights )
                
            self._preproc = preprocess_input
        
            
        # RESNET-50
        elif self._model_id == 'resnet50':
            from keras.applications.resnet50 import ResNet50 , preprocess_input
            self._input_shape = ( 224 , 224 , 3 )
                
            if weights is None or weights == 'imagenet':
                self._model = ResNet50( weights     = weights , 
                                        include_top = False , 
                                        input_shape = self._input_shape )
            else:
                self._model = load_model( weights )
                
            self._preproc = preprocess_input
        
        
        # INCEPTION-RESNET-V2
        elif self._model_id == 'inception-resnet-v2':
            from keras.applications.inception_resnet_v2 import InceptionResNetV2 , preprocess_input
            self._input_shape = ( 299 , 299 , 3 )

            if weights is None or weights == 'imagenet':                
                self._model = InceptionResNetV2( weights     = weights , 
                                                 include_top = False   , 
                                                 input_shape = self._input_shape )
            else:
                self._model = load_model( weights )
                
                self._preproc = preprocess_input
               

        # MOBILENET
        elif self._model_id == 'mobilenet':
            from keras.applications.mobilenet import MobileNet , preprocess_input , relu6 , DepthwiseConv2D
            self._input_shape = ( 224 , 224 , 3 )

            if weights is None or weights == 'imagenet':                
                self._model = MobileNet( weights     = weights , 
                                         include_top = False   , 
                                         input_shape = self._input_shape )        
            else:
                self._model = load_model( weights )
                
            self._preproc     = preprocess_input
            
            
        # MOBILENET-V2
        elif self._model_id == 'mobilenet-v2':
            from keras.applications.mobilenetv2 import MobileNetV2 , preprocess_input
            self._input_shape = ( 224 , 224 , 3 )

            if weights is None or weights == 'imagenet':                
                self._model = MobileNetV2( weights     = weights , 
                                           include_top = False   , 
                                           input_shape = self._input_shape )
            else:
                self._model = load_model( weights )
                
            self._preproc = preprocess_input
           
            
        # DENSENET-121
        elif self._model_id == 'densenet-121':
            from keras.applications.densenet import DenseNet121 ,  preprocess_input
            self._input_shape = ( 224 , 224 , 3 )

            if weights is None or weights == 'imagenet':                
                self._model = DenseNet121( weights     = weights          , 
                                           input_shape = self._input_shape , 
                                           include_top = False )        
            else:
                self._model = load_model( weights )
                
            self._preproc = preprocess_input    

                
        # DENSENET-169
        elif self._model_id == 'densenet-169':
            from keras.applications.densenet import DenseNet169 ,  preprocess_input
            self._input_shape = ( 224 , 224 , 3 )

            if weights is None or weights == 'imagenet':                
                self._model = DenseNet169( weights     = weights , 
                                           input_shape = self._input_shape , 
                                           include_top = False )        
            else:
                self._model = load_model( weights )
                
            self._preproc = preprocess_input    
                
            
        # DENSENET-201
        elif self._model_id == 'densenet-201':
            from keras.applications.densenet import DenseNet201 ,  preprocess_input
            self._input_shape = ( 224 , 224 , 3 )

            if weights is None or weights == 'imagenet':                
                self._model = DenseNet201( weights     = weights , 
                                           input_shape = self._input_shape , 
                                           include_top = False )        
            else:
                self._model = load_model( weights )
                
            self._preproc = preprocess_input  
            
            
        # NASNET
        elif self._model_id == 'nasnet':
            from keras.applications.nasnet import NASNetLarge ,  preprocess_input
            self._input_shape = ( 331 , 331 , 3 )

            if weights is None or weights == 'imagenet':                
                self._model = NASNetLarge( weights     = weights , 
                                           input_shape = self._input_shape , 
                                           include_top = False )        
            else:
                self._model = load_model( weights )
                
            self._preproc = preprocess_input

            
        # NASNET-MOBILE
        elif self._model_id == 'nasnet-mobile':
            from keras.applications.nasnet import NASNetMobile ,  preprocess_input
            self._input_shape = ( 224 , 224 , 3 )

            if weights is None or weights == 'imagenet':                
                self._model = NASNetMobile( weights     = weights , 
                                            input_shape = self._input_shape , 
                                            include_top = False )        
            else:
                self._model = load_model( weights )
                
            self._preproc = preprocess_input
               
        
        # Get model input shape
        self._get_model_input_shape()

            
            
    # ===================================
    # Get model input shape
    # ===================================
    
    def _get_model_input_shape( self ):
        input_shape = self._model.input.shape
        d1          = myint( input_shape[1] )
        d2          = myint( input_shape[2] )
        d3          = myint( input_shape[3] )
        
        self._model_input_shape = ( d1 , d2 , d3 )


    
    # ===================================
    # Learning a model from scratch
    # ===================================
        
    def _learning( self ):
        # Init training monitoring
        self._init_monitoring()
        self._mode_now = 'learning'

        # Create output filenames
        self._create_filenames_best_model()

        # Add new last layer
        self._add_new_last_layer()
        
        # Compile model
        self._compile_model( key='ls' )
         
        # Print summary
        print( self._model.summary() )

        # Run training
        self._train( n_epochs=self._n_epochs )
        
        
        
    # ===================================
    # Transfer-learning
    # ===================================
    
    def _transfer_learning( self ):
        # Init training monitoring
        self._init_monitoring()
        self._mode_now = 'transfer-learning'
        
        # Create output filenames
        self._create_filenames_best_model()

        # Add new last layer
        self._add_new_last_layer()
        
        # Compile model by freezing all layers except the last
        self._compile_model( key='tl' )   

        # Print summary
        print( self._model.summary() )

        # Run training
        self._train( n_epochs=self._n_epochs )
        
        

    # ===================================
    # Fine-tuning
    # ===================================
    
    def _fine_tuning( self ):
        # Compile model by freezing a certain number of layers
        self._compile_model( key='ft' )
        self._mode_now = 'fine-tuning'
        
        # Save transfer learning perfromance data frame
        self._metrics_tl = self._metrics.copy()
        
        # Print summary
        print( self._model.summary() )

        # Run training
        self._train( n_epochs=self._n_epochs_ft )    
    
    
    
    # ===================================
    # Re-training
    # ===================================
    
    def _re_training( self ):
        # Init training monitoring
        self._init_monitoring()
        self._mode_now = 're-training'

        # Create output filenames
        self._create_filenames_best_model()

        # For loop on the re-training steps
        for i in range( self._num_steps_rt ):
            print( '\n>>>>>>>>>>>> RE-TRAINING STEP N.', i )
        
            # Compile model by freezing a certain number of layers
            self._compile_model( key='rt' , step=i )
        
            # Print summary
            print( self._model.summary() )

            # Run training
            self._train( n_epochs=self._n_epochs )    


    
    # ===================================
    # Init training monitoring
    # ===================================
    
    def _init_monitoring( self ):
        # Init testing y_true and y_pred
        self._y_true = self._y_valid.copy()
        self._y_pred = np.zeros( ( len( self._y_true ) , 2 ) , dtype=myfloat )


        # Init monitoring
        if self._metric == 'mse':
            self._metric_monitor = 1e10
        else:
            self._metric_monitor = 0.0


        # Initialize "Metrics" class
        if self._task == 'classification':
            if self._n_classes == 2:
                self._mcalc = me.Metrics( task='class' )
            else:
                self._mcalc = me.Metrics( task='mclass' , n_classes=self._n_classes )
        
        elif self._task == 'regression':
            self._mcalc = me.Metrics( task='regr' )


    
    # ===================================
    # Create filenames best model
    # ===================================
    
    def _create_filenames_best_model( self ):
        self._file_log  = os.path.join( self._pathout , 'logger_' + self._output_hash + '.csv' )
        self._file_hdf5 = os.path.join( self._pathout , 'weights_' + self._output_hash + '.h5' )
 


    # ===================================
    # Add new last layer
    # ===================================
    
    def _add_new_last_layer( self ):
        from keras.layers import Dense, GlobalAveragePooling2D
        from keras.models import Model

        input = self._model.input 
        
        if self._weights_start == 'imagenet':
            x = self._model.output
            x = GlobalAveragePooling2D( name='docking_layer' )( x )
        else:
            x = self._model.layers[-3].output

        x = Dense( self._num_nodes_dense , activation='relu' , name='dense_new_01' )( x ) 
        
        if self._task == 'classification':
            predictions = Dense( self._n_classes , activation='softmax' , name='dense_new_02' )( x )
        
        elif self._task == 'regression':
            predictions = Dense( 1 , activation='linear' , name='dense_new_02' )( x )
        
        model_new   = Model( inputs=input , outputs=predictions )
        self._model = model_new
        
        
        
    # ===================================
    # Compile model
    # ===================================
    
    def _compile_model( self , key='ls' , step=None ):
        # Define WCCE
        wcce_loss = wcce.wrapped_partial( wcce._wcce , 
                                          weights = tf.constant( self._wcce_weight ) )
                
        
        # Set optimizer and layers to freeze
        if key == 'ls':
            optimizer = self._optimizer
            
        elif key == 'tl':
            optimizer = self._optimizer            
            
            for layer in self._model.layers:
                if layer.name not in [ 'dense_new_01' , 'dense_new_02' ]:
                    layer.trainable = False                                 
        
        elif key == 'ft':    
            optimizer = self._optimizer_ft
            
            num_layers_freeze = self._num_layers_freeze_ft
            
            for layer in self._model.layers[:num_layers_freeze]:
                layer.trainable = False
            for layer in self._model.layers[num_layers_freeze:]:
                layer.trainable = True

        elif key == 'rt':
            optimizer = self._optimizer

            num_layers_unfreeze = self._num_layers_unfreeze_rt * ( step + 1 )
            num_layers_freeze   = len( self._model.layers ) - num_layers_unfreeze

            for layer in self._model.layers[:num_layers_freeze]:
                layer.trainable = False
            for layer in self._model.layers[num_layers_freeze:]:
                layer.trainable = True


        # Compile
        if self._loss == 'categorical_crossentropy':
            metrics = [ 'acc' ]
            self._model.compile( optimizer = optimizer ,
                                 loss      = self._loss ,
                                 metrics   = metrics )
        
        elif self._loss == 'wcce':
            metrics = [ 'acc' ]
            self._model.compile( optimizer = optimizer ,
                                 loss      = wcce_loss ,
                                 metrics   = metrics )

        elif self._loss == 'mean_squared_error':
            self._model.compile( optimizer = optimizer ,
                                 loss      = self._loss )            
    
   

    # ===================================
    # Training
    # ===================================
    
    def _train( self , n_epochs=10 ):
        # Create batch generators
        self._gen_train = self._create_batch_generator_train()
        
        
        # Start timing training
        time_start = time.time()
        
        
        # Training on batches
        print( '\nPerforming training on batches ....' )
        self._metrics = self._training_on_batches( n_epochs )        

        
        # Create label to save training outputs
        self._timestr = time.strftime("%Y%m%d-%H%M%S")
        
        
        # Save time elapsed for training
        self._time = time.time() - time_start   
    
        
        
    # ===================================
    # Create batch generators
    # ===================================
    
    def _create_batch_generator_train( self ):
        # Allocate memory for batches storing images and labels 
        batch_img = np.zeros( ( self._batch_size , 
                                self._input_shape[0] , 
                                self._input_shape[1] , 
                                self._input_shape[2] ) )
        
        if self._task == 'classification':
            batch_y = np.zeros( ( self._batch_size , self._n_classes ) )

        elif self._task == 'regression':
            batch_y = np.zeros( ( self._batch_size , 1 ) )
            
        
        # Keep creating batches until receiving stopping signal 
        while 1:
            ind = random.sample( self._ind_train.tolist() , self._batch_size )
        
            for i in range( len( ind ) ):
                # Select random filepath and corresponding y
                filein = self._imgs_train[ind[i]]
                
                if self._task == 'classification':
                    y = to_categorical( self._y_train[ ind[i] ] , 
                                        num_classes=self._n_classes )
                elif self._task == 'regression':
                    y = self._y_train[ind[i]]
                
                # Open image in resized format
                imgp  = kimage.load_img( filein ,
                                         target_size=( self._input_shape[0] , 
                                                       self._input_shape[1] ) )
                img   = kimage.img_to_array( imgp )
                
                # Preprocessing  
                img = self._preproc( img )
            
                # Assign to batch collectors
                batch_img[i] = img
                batch_y[i]   = y

            yield ( batch_img , batch_y )
 


    # ===================================
    # Training on batches
    # ===================================
    
    def _training_on_batches( self , n_epochs ):
        # Number of steps per epoch
        n_steps = self._steps_per_epoch_train
        
    
        # Initialize data frame containing the metrics
        if self._task == 'classification':
            if self._n_classes == 2:
                metrics = pd.DataFrame( columns = [  'mode'           ,
                                                     'loss'           ,
                                                     'accuracy'       ,
                                                     'val_accuracy'   ,
                                                     'val_precision'  ,
                                                     'val_recall'     , 
                                                     'val_f1score'    , 
                                                     'val_auc'        ,
                                                     'val_sensitivity',
                                                     'val_specificity',
                                                     'val_cohen_kappa',
                                                     'val_threshold'   ] )
            else:
                metrics = pd.DataFrame( columns = [ 'mode'           ,
                                                    'loss'           ,
                                                    'accuracy'       , 
                                                    'val_accuracy'   ,
                                                    'val_cohen_kappa' ] )

        elif self._task == 'regression':
            metrics = pd.DataFrame( columns = [ 'mode'    ,
                                                'loss'    , 
                                                'val_mse' , 
                                                'val_r2'  ] )


        # Loop on epochs
        for epoch in range( n_epochs ):
            print( '\nEpoch n.', epoch )
        
            # Initialize vectors
            loss = np.zeros( n_steps )
            acc  = np.zeros( n_steps )
            pbar = ProgressBar().start()
           
            row_epoch = [ self._mode_now ]


            # For loop on steps
            for step in pbar( range( n_steps ) ):
                # Get batch
                gen     = next( self._gen_train )
                X_batch = gen[0]
                Y_batch = gen[1]
                
                
                # Backprogation from single batch
                history = self._model.train_on_batch( X_batch , Y_batch )
                
                
                #  Save loss and accuracy from batch
                if self._task == 'classification':
                    loss[ step ] = history[0]
                    acc[ step ]  = history[1]
                elif self._task == 'regression':
                    loss[ step ] = history

                
                # Update progress bar
                time.sleep( 0.01 )
                           
            
            # Compute mean loss and accuracy over the epoch
            if self._task == 'classification':
                loss_mean = np.mean( loss )
                acc_mean  = np.mean( acc )
                row_epoch.append( loss_mean )
                row_epoch.append( acc_mean )
                print( '\tloss: ', loss_mean, ' --  acc: ' , acc_mean,'\n' )
            
            elif self._task == 'regression': 
                loss_mean = np.mean( loss )
                row_epoch.append( loss_mean )
                print( '\tloss: ', loss_mean, '\n' )
            
            
            # Compute validation metrics on the entire validation dataset
            outputs              = self._validate_end_of_epoch() 
            row_epoch           += outputs
            metrics.loc[ epoch ] = row_epoch 
            
            if os.path.isfile( self._file_log ):
                self._update_logger( row_epoch ) 
            else:
                self._update_logger( epoch , init=metrics )
            
            print( '\tupdated logger file: ', self._file_log )
            
        return metrics
            
        
    
    # ===================================
    # Validate end of epoch
    # ===================================

    def _validate_end_of_epoch( self ):
        # Get validation data 
        if self._debug:
            y_true = self._y_valid.copy()[:NUM_DEBUG_2]
        else:
            y_true = self._y_valid.copy()
        
        
        # Compute predictions for all validation images in order
        y_pred = self._predict_on_validation( self._imgs_valid )

        
        # Calculate metrics with class "Metrics"
        self._mcalc._run( y_true , y_pred )


        # Print --> Case A: Classification
        if self._task == 'classification':
            if self._n_classes == 2:
                output_row = [ self._mcalc._accuracy , self._mcalc._precision , self._mcalc._recall ,
                               self._mcalc._f1score , self._mcalc._auc , self._mcalc._sensitivity , 
                               self._mcalc._specificity , self._mcalc._cohen_kappa , self._mcalc._thres_best ]
            
                print( '\tval_accuracy: %s - val_precision: %s - val_recall: %s - val_f1score: %s - val_cohen_kappa: %s' % \
                        ( self._num2str( self._mcalc._accuracy ) , self._num2str( self._mcalc._precision ) ,
                          self._num2str( self._mcalc._recall ) , self._num2str( self._mcalc._f1score ) , 
                          self._num2str( self._mcalc._cohen_kappa ) ) )
            
                print( '\tval_auc: %s - val_sensitivity( %s ): % s - val_specificity( %s ): %s' % \
                        ( self._num2str( self._mcalc._auc ) , self._num2str( self._mcalc._thres_best ) , 
                          self._num2str( self._mcalc._sensitivity ) , self._num2str( self._mcalc._thres_best ) , 
                          self._num2str( self._mcalc._specificity ) ) )


            # Compute metrics for the multi-class problem
            else:
                output_row = [ self._mcalc._accuracy , self._mcalc._cohen_kappa ]          
            
                print( '\tval_accuracy: %s - val_cohen_kappa: %s ' % \
                        ( self._num2str( self._mcalc._accuracy ) , self._num2str( self._mcalc._cohen_kappa ) ) )        
         

        # Print --> Case B: Regression
        elif self._task == 'regression':
            output_row = [ self._mcalc._mse , self._mcalc._r2 ]
            print( '\tval_mse: %s - val_r2: %s' % ( self._num2str( self._mcalc._mse ) , 
                                                    self._num2str( self._mcalc._r2 ) ) )


        # Save best model
        if self._metric == 'accuracy':
            self._save_best_model( self._mcalc._accuracy , y_true , y_pred , 'val_' + self._metric )
            
        elif self._metric == 'auc':
            self._save_best_model( self._mcalc._auc , y_true , y_pred , 'val_' + self._metric )            
        elif self._metric == 'sensitivity':
            self._save_best_model( self._mcalc._sensitivity , y_true , y_pred , 'val_' + self._metric )
        
        elif self._metric == 'specificity':
            self._save_best_model( self._mcalc._specificity , y_true , y_pred , 'val_' + self._metric )
            
        elif self._metric == 'f1score':
            self._save_best_model( self._mcalc._f1score , y_true , y_pred , 'val_' + self._metric )

        elif self._metric == 'cohen_kappa':
            self._save_best_model( self._mcalc._cohen_kappa , y_true , y_pred , 'val_' + self._metric )

        elif self._metric == 'recall':
            self._save_best_model( self._mcalc._recall , y_true , y_pred , 'val_' + self._metric )
        
        elif self._metric == 'precision':
            self._save_best_model( self._mcalc._precision , y_true , y_pred , 'val_' + self._metric )
         
        elif self._metric == 'mse':
            self._save_best_model( self._mcalc._mse , y_true , y_pred , 'val_' + self._metric )
            
        elif self._metric == 'r2':
            self._save_best_model( self._mcalc._r2 , y_true , y_pred , 'val_' + self._metric )            
        
        return output_row
        

        
    # ===================================
    # Predict on full validation dataset
    # ==================================        

    def _predict_on_validation( self , files ):
        if self._debug:
            n_files = NUM_DEBUG_2
        else:
            n_files = len( files )
        
        y_pred = []       
        
        for i in range( n_files ):
            filein = files[i]
                        
            imgp = kimage.load_img( filein , 
                                    target_size=( self._input_shape[0] , 
                                                  self._input_shape[1] ) )
            img  = kimage.img_to_array( imgp )
            img  = np.expand_dims( img , axis=0 )
            img  = self._preproc( img )

            y_pred.append( self._model.predict( img )[0] )
            
        y_pred = np.array( y_pred )
        
        return y_pred

        

    # ===================================
    # Compute sensitivity and specificity using
    # Youden's operating point
    # =================================== 
    
    def _calc_sens_and_spec( self , tpr , fpr  , thres ):
        # Compute best operating point on ROC curve, define as 
        # the one maximizing the difference ( TPR - FPR ), also called
        # Youden operating point ( https://en.wikipedia.org/wiki/Youden%27s_J_statistic )
        diff       = tpr - fpr
        i_max      = np.argwhere( diff == np.max( diff ) )
        threshold  = thres[ i_max ][0]
        fpr_best   = fpr[i_max]
        tpr_best   = tpr[i_max]
                    
        # Compute sensitivity and specificity at optimal operating point
        sensitivity = tpr[ i_max ][0]
        specificity = 1 - fpr[ i_max ][0]
        
        return sensitivity , specificity , threshold


    
    # ===================================
    # Save model according to selected metric
    # =================================== 
    
    def _save_best_model( self , metric_current , y_true , y_pred , metric_name ):
        # Case A: Metric has improved by increasing
        if metric_current > self._metric_monitor and metric_name != 'mse':
            self._metric_monitor = metric_current
            self._model.save( self._file_hdf5 )
            
            print( '\n\t', metric_name + ' has improved:\n') 
            print( '\tsaving weights to ' , self._file_hdf5 )
            
            self._y_true = y_true.copy()
            self._y_pred = y_pred.copy()


        # Case B: Metric has improved by decreasing
        elif metric_current < self._metric_monitor and metric_name == 'mse':
            self._model.save( self._file_hdf5 )
            
            print( '\n\t', metric_name + ' has improved:\n') 
            print( '\tsaving weights to ' , self._file_hdf5 )
            
            self._y_true = y_true
            self._y_pred = y_pred

        
        # Case C: Metric has not improved
        else:
            print( '\n\t', metric_name + ' has not improved' ) 

            

    # ===================================
    # Return string of floating number 
    # rounded with a certain amount of 
    # digits
    # =================================== 

    def _num2str( self , num ):
        if isinstance( num  , np.ndarray ) or isinstance( num  , list ):
            num = num[0]

        return str( round( num , 4 ) )



    # ===================================
    # Update logger
    # ===================================
    
    def _update_logger( self , outputs , init=None ):
        if init is None:
            df                = pd.read_csv( self._file_log , sep=SEP )
            ind               = df.index.get_values()[-1]
            df.loc[ ind + 1 ] = outputs
            df.to_csv( self._file_log , sep=SEP , index=False )

        else:
            init.to_csv( self._file_log , sep=SEP , index=False )
   


    # ===================================
    # Write all outputs
    # ===================================
    
    def _write_outputs( self , time=None ):
        # Create output label
        self._get_output_label()


        # Save model
        self._save_model()


        # Write history
        self._write_history( time=time )


        # Write config file
        self._write_config()



    # ===================================
    # Get output label
    # ===================================
    
    def _get_output_label( self ):
        label = ''
        if self._add_label is not None:
            label += self._add_label + '_'
            
        label += self._timestr + '_'        
        label += self._output_hash
            
        self._output_label = label
    
    
    
    # ===================================
    # Save model
    # ===================================
    
    def _save_model( self ):
        # Construct label
        fileout  = os.path.join( self._pathout , 'cnn_model_' + self._model_id + '_' )
        fileout += self._output_label + '.h5'


        # Move file_best_model to fileout
        if os.path.isfile( self._file_hdf5 ):
            command_mv = 'mv ' + self._file_hdf5 + ' ' + fileout

            os.system( command_mv )
            print( 'Move command:\n', command_mv  )
                
        else:
            self._model.save( fileout )
            print( 'Saved model at last epoch' )
                
        print( 'Model has been saved in ', fileout )
        
    
    
    # ===================================
    # Write history
    # ===================================
    
    def _write_history( self , time=None ):
        # Construct output filename
        fileout  = os.path.join( self._pathout , 'cnn_history_' + self._model_id + '_' )
        fileout += self._output_label + '.json'
        
        
        # Rename logger and phase if mode == 'fine-tuning'
        file_log = os.path.join( self._pathout , 'cnn_logger_' + self._model_id + '_' + \
                                                 self._output_label + '.csv' )
        command = 'mv ' + self._file_log + ' ' + file_log
        os.system( command )
        print( '\nSaved logger file in:', file_log )
        df_log = pd.read_csv( file_log , sep=SEP )

        
        # Common dictionary
        num_epochs = self._n_epochs
        time       = self._time
        
        dict = {
                'csv_train'            : self._make_json_serializable( self._csv_train )       ,
                'csv_valid'            : self._make_json_serializable( self._csv_valid )       ,
                'col_imgs'             : self._make_json_serializable( self._col_imgs )        ,
                'col_label'            : self._make_json_serializable( self._col_label )       ,
                'pathout'              : self._make_json_serializable( self._pathout )         , 
                'num_train_samples'    : self._make_json_serializable( self._n_imgs_train )    ,
                'num_valid_samples'    : self._make_json_serializable( self._n_imgs_valid )    ,
                'task'                 : self._make_json_serializable( self._task )            ,
                'mode'                 : self._make_json_serializable( self._mode )            ,
                'model'                : self._make_json_serializable( self._model_id )        ,
                'metric'               : self._make_json_serializable( self._metric )          ,
                'batch_size'           : self._make_json_serializable( self._batch_size )      ,
                'optimizer'            : self._make_json_serializable( self._optimizer_type )  ,
                'num_epochs'           : self._make_json_serializable( self._n_epochs )        ,
                'weights_start'        : self._make_json_serializable( self._weights_start )   ,
                'y_true'               : self._make_json_serializable( self._y_true.tolist() ) ,
                'y_pred'               : self._make_json_serializable( self._y_pred.tolist() )
                }


        # Add time elapsed
        if time is not None:
            dict.update( { 'time': time } )


        # Add parameters specific of re-training
        if self._mode == 're-training':
            aux= {  'num_steps'          : self._num_steps_rt ,
                    'num_layers_unfreeze': self._num_layers_unfreeze_rt }
            dict.update( aux )
                
               
        # Case A: Classification
        if self._task == 'classification':
            # Case A.1: Binary classification
            if self._n_classes == 2:
                if self._mode != 'fine-tuning':
                    df_aux = df_log
                else:
                    df_aux = df_log[ df_log[ 'mode' ] == 'transfer-learning' ]

                aux = { 'loss_train'            : self._make_json_serializable( df_aux[ 'loss' ].values )                      ,
                        'accuracy_train'        : self._make_json_serializable( df_aux['accuracy'].values )                    ,
                        'accuracy_valid'        : self._make_json_serializable( df_aux['val_accuracy'].values )                ,
                        'peak_accuracy_valid'   : self._make_json_serializable( np.max( df_aux['val_accuracy'].values ) )      ,                
                        'cohen_kappa_valid'     : self._make_json_serializable( df_aux['val_cohen_kappa'].values )             ,
                        'peak_cohen_kappa_valid': self._make_json_serializable( np.max( df_aux['val_cohen_kappa'].values ) )   ,
                        'precision_valid'       : self._make_json_serializable( df_aux['val_precision'].values )               ,
                        'peak_precision_valid'  : self._make_json_serializable( np.max( df_aux['val_precision'].values ) )     ,
                        'recall_valid'          : self._make_json_serializable( df_aux['val_recall'].values )                  ,
                        'peak_recall_valid'     : self._make_json_serializable( np.max( df_aux['val_recall'].values ) )        ,
                        'f1score_valid'         : self._make_json_serializable( df_aux[ 'val_f1score' ].values )               ,
                        'peak_f1score_valid'    : self._make_json_serializable( np.max( df_aux[ 'val_f1score' ].values ) )     ,
                        'auc_valid'             : self._make_json_serializable( df_aux[ 'val_auc' ].values )                   ,
                        'peak_auc_valid'        : self._make_json_serializable( np.max( df_aux[ 'val_auc' ].values ) )         ,
                        'sensitivity_valid'     : self._make_json_serializable( df_aux[ 'val_sensitivity' ].values )           ,
                        'peak_sensitivity_valid': self._make_json_serializable( np.max( df_aux[ 'val_sensitivity' ].values ) ) ,
                        'specificity_valid'     : self._make_json_serializable( df_aux[ 'val_sensitivity' ].values )           ,
                        'peak_specificity_valid': self._make_json_serializable( np.max( df_aux[ 'val_specificity' ].values ) ) ,                    
                      }
                 
                dict.update( aux )
       

                if self._mode == 'fine-tuning':
                    df_aux = df_log[ df_log[ 'mode' ] == 'fine-tuning' ]
                    key    = '_ft'

                    aux = { 'loss_train'            + key: self._make_json_serializable( df_aux[ 'loss' ].values )                      ,
                            'accuracy_train'        + key: self._make_json_serializable( df_aux['accuracy'].values )                    ,
                            'accuracy_valid'        + key: self._make_json_serializable( df_aux['val_accuracy'].values )                ,
                            'peak_accuracy_valid'   + key: self._make_json_serializable( np.max( df_aux['val_accuracy'].values ) )      ,                
                            'cohen_kappa_valid'     + key: self._make_json_serializable( df_aux['val_cohen_kappa'].values )             ,
                            'peak_cohen_kappa_valid'+ key: self._make_json_serializable( np.max( df_aux['val_cohen_kappa'].values ) )   ,
                            'precision_valid'       + key: self._make_json_serializable( df_aux['val_precision'].values )               ,
                            'peak_precision_valid'  + key: self._make_json_serializable( np.max( df_aux['val_precision'].values ) )     ,
                            'recall_valid'          + key: self._make_json_serializable( df_aux['val_recall'].values )                  ,
                            'peak_recall_valid'     + key: self._make_json_serializable( np.max( df_aux['val_recall'].values ) )        ,
                            'f1score_valid'         + key: self._make_json_serializable( df_aux[ 'val_f1score' ].values )               ,
                            'peak_f1score_valid'    + key: self._make_json_serializable( np.max( df_aux[ 'val_f1score' ].values ) )     ,
                            'auc_valid'             + key: self._make_json_serializable( df_aux[ 'val_auc' ].values )                   ,
                            'peak_auc_valid'        + key: self._make_json_serializable( np.max( df_aux[ 'val_auc' ].values ) )         ,
                            'sensitivity_valid'     + key: self._make_json_serializable( df_aux[ 'val_sensitivity' ].values )           ,
                            'peak_sensitivity_valid'+ key: self._make_json_serializable( np.max( df_aux[ 'val_sensitivity' ].values ) ) ,
                            'specificity_valid'     + key: self._make_json_serializable( df_aux[ 'val_sensitivity' ].values )           ,
                            'peak_specificity_valid'+ key: self._make_json_serializable( np.max( df_aux[ 'val_specificity' ].values ) ) ,                    
                            'optimizer_type'        + key: self._make_json_serializable( self._optimizer_type_ft )  
                         }
                 
                    dict.update( aux )

            
            # Case A.2: Multi-label classification
            else:    
                if self._mode != 'fine-tuning':
                    df_aux = df_log
                else:
                    df_aux = df_log[ df_log[ 'mode' ] == 'transfer-learning' ]

                aux = { 'loss_train'            : self._make_json_serializable( df_aux[ 'loss' ].values )                      ,
                        'accuracy_train'        : self._make_json_serializable( df_aux['accuracy'].values )                    ,
                        'accuracy_valid'        : self._make_json_serializable( df_aux['val_accuracy'].values )                ,
                        'peak_accuracy_valid'   : self._make_json_serializable( np.max( df_aux['val_accuracy'].values ) )      ,                
                        'cohen_kappa_valid'     : self._make_json_serializable( df_aux['val_cohen_kappa'].values )             ,
                        'peak_cohen_kappa_valid': self._make_json_serializable( np.max( df_aux['val_cohen_kappa'].values ) )   ,
                      }
                 
                dict.update( aux )
       

                if self._mode == 'fine-tuning':
                    df_aux = df_log[ df_log[ 'mode' ] == 'fine-tuning' ]
                    key    = '_ft'

                    aux = { 'loss_train'            + key: self._make_json_serializable( df_aux[ 'loss' ].values )                      ,
                            'accuracy_train'        + key: self._make_json_serializable( df_aux['accuracy'].values )                    ,
                            'accuracy_valid'        + key: self._make_json_serializable( df_aux['val_accuracy'].values )                ,
                            'peak_accuracy_valid'   + key: self._make_json_serializable( np.max( df_aux['val_accuracy'].values ) )      ,                
                            'cohen_kappa_valid'     + key: self._make_json_serializable( df_aux['val_cohen_kappa'].values )             ,
                            'peak_cohen_kappa_valid'+ key: self._make_json_serializable( np.max( df_aux['val_cohen_kappa'].values ) )   ,
                          }
                 
                    dict.update( aux )


        # Case B: Regression
        else:
            if self._mode != 'fine-tuning':
                    df_aux = df_log
            else:
                df_aux = df_log[ df_log[ 'mode' ] == 'transfer-learning' ]

            aux = { 'loss_train'      : self._make_json_serializable( df_aux[ 'loss' ].values )            ,
                    'mse_valid'       : self._make_json_serializable( df_aux['val_mse'].values )           ,
                    'lowest_mse_valid': self._make_json_serializable( np.min( df_aux['val_mse'].values ) ) ,                
                    'r2_valid'        : self._make_json_serializable( df_aux['val_r2'].values )            ,
                    'peak_r2_valid'   : self._make_json_serializable( np.max( df_aux['val_r2'].values ) )  ,
                  }
                 
            dict.update( aux )
       

            if self._mode == 'fine-tuning':
                df_aux = df_log[ df_log[ 'mode' ] == 'fine-tuning' ]
                key    = '_ft'

                aux = { 'loss_train'       + key: self._make_json_serializable( df_aux[ 'loss' ].values )            ,
                        'mse_valid'        + key: self._make_json_serializable( df_aux['val_mse'].values )           ,
                        'lowest_mse_valid' + key: self._make_json_serializable( np.min( df_aux['val_mse'].values ) ) ,                
                        'r2_valid'         + key: self._make_json_serializable( df_aux['val_r2'].values )            ,
                        'peak_r2_valid'    + key: self._make_json_serializable( np.max( df_aux['val_r2'].values ) )  ,
                      }
                 
                dict.update( aux )

               
        # Write history to JSON file
        with open( fileout , 'w' ) as fp:
            json.dump( dict , fp , sort_keys=True )
        print( '\nTraining logfile has been saved in ', fileout )
        
        
    
    # ===================================
    # Is JSON serializable?
    # =================================== 

    def _make_json_serializable( self , input ):
        if type( input ) is np.ndarray:
            if len( input ) == 1:
                while type( input ) is np.ndarray:
                    input = input[0]
                input = myfloat2( input )
                
            else:
                input = input.astype( myfloat2 ).tolist()

        elif type( input ) == type( [] ):
            input = np.array( input ).astype( myfloat2 ).tolist()
        
        return input
   

    
    # ===================================
    # Write config file
    # ===================================
    
    def _write_config( self ):
        # Construct output filename
        fileout  = os.path.join( self._pathout , 'cnn_config_' + self._model_id + '_' )
        fileout += self._output_label + '.yml'
        
        
        # Case A: config file is provided
        if self._config_file is not None:
            with open( self._config_file_hashed , 'r' ) as stream:
                dict = yaml.load( stream )
    
        
        # Delete hashed config file
        command = 'rm ' + self._config_file_hashed
        os.system( command )
        
        
        # Write dictionary to YAML
        with open( fileout , 'w' ) as outfile:
            yaml.dump( dict , outfile , default_flow_style=False )
        
        print( '\nConfig file has been saved in ', fileout )     



    

