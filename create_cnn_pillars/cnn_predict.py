'''
Use trained CNN model to make a prediction on a single image
'''


# Author: Filippo Arcadu 
#         AIM4EYE Project
#         10/10/2017



from __future__ import division , print_function
import argparse
import sys
import os
import time
import numpy as np

from keras.preprocessing import image
from keras.models import load_model
from keras import backend as K

import tensorflow as tf

import wcce




# =============================================================================
# Set tensorflow tensor ordering for Keras 
# =============================================================================

K.set_image_dim_ordering( 'tf' )
        


    
# =============================================================================
# List of available architecture 
# =============================================================================

LIST_ARCHS = [ 'vgg16' , 'vgg19' , 'inception-v3' , 'nasnet' , 
               'inception-resnet-v2' , 'xception' ,
               'mobilenet' , 'resnet50', 'squeezenet', 'densenet' ]



               
# =============================================================================
# Types 
# =============================================================================

myint   = np.int
myfloat = np.float32




# =============================================================================
# Parsing input arguments
# =============================================================================

def _examples():
    print( '\n\nEXAMPLES\n\nPredict with CNN model specifying architecture type in the name:\n' \
            '"""\npython cnn_predict.py -i /pstore/data/pio/PreprocessedData/Ride-Rise-F2_crop-fov/CF-50101-2007-12-18-M6-RE-F2-LS_fov_mask_input_and_mask_crop_full.png -o ./ -m /pstore/data/pio/Models/CF/cf_data_curation/identify_out_of_the_eye_images/inception-v3_identify-out-of-the-eye_18-02-09.h5\n"""'\
           '\n\nPredict with CNN model having a generic name:\n' \
            '"""\npython cnn_predict.py -i /pstore/data/pio/PreprocessedData/Ride-Rise-F2_crop-fov/CF-50101-2007-12-18-M6-RE-F2-LS_fov_mask_input_and_mask_crop_full.png -o ./ -m /pstore/data/pio/Tasks/PIO-218/dl/models/weights_best_127003044166031045708553901818539765066.h5 -t inception-v3\n"""' \
           '\n\nPredict bottleneck features with CNN model having a generic name:\n' \
            '"""\npython cnn_predict.py -i /pstore/data/pio/PreprocessedData/Ride-Rise-F2_crop-fov/CF-50101-2007-12-18-M6-RE-F2-LS_fov_mask_input_and_mask_crop_full.png -o ./ -m /pstore/data/pio/Tasks/PIO-218/dl/models/weights_best_127003044166031045708553901818539765066.h5 -t inception-v3 -b\n"""\n\n'           
           )    

          
          
def _get_args():
    parser = argparse.ArgumentParser(    
                                        prog='cnn_predict',
                                        description='Create prediction with CNN model',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                        add_help=False
                                    )
    
    parser.add_argument('-i', '--file_in', dest='file_in',
                        help='Specify input image')
                        
    parser.add_argument('-o', '--path_out', dest='path_out',
                        help='Specify path where to save prediction result')                        

    parser.add_argument('-m', '--model', dest='model',
                        help='Specify HDF5 CNN model to use')  
                        
    parser.add_argument('-n', '--num_classes', dest='num_classes', type=myint, default=2,
                        help='Specify number of classes if compiled loss is WCCE')  
    
    parser.add_argument('-t', '--type-arch', dest='type_arch', 
                        help='Select architecture type' )
                        
    parser.add_argument('-b', '--bottleneck', dest='bottleneck', action='store_true' ,
                        help='Enable computation of bottleneck features' )                        

    parser.add_argument('-h', '--help', dest='help', action='store_true',
                        help='Print help and examples')                            

    args = parser.parse_args()
    
    if args.help is True:
        parser.print_help()
        _examples()
        sys.exit()

    if args.file_in is None:
        parser.print_help()
        sys.exit('\nERROR: Input image not specified!\n')
        
    if os.path.isfile( args.file_in ) is False:
        parser.print_help()
        sys.exit('\nERROR: Input image file does not exist!\n')        

    if args.path_out is None:
        parser.print_help()
        sys.exit('\nERROR: Output path not specified!\n')
        
    if args.model is None:
        parser.print_help()
        sys.exit('\nERROR: Input model not specified!\n')

    if os.path.isfile( args.model ) is False:
        parser.print_help()
        sys.exit('\nERROR: Input model file does not exist!\n')
        
    if args.type_arch is not None and args.type_arch.lower() not in LIST_ARCHS:
        parser.print_help()
        sys.exit('\nERROR: Architecture type is not available, choose among ' + \
                  " ".join( str( x ) for x in LIST_ARCHS ) + '!\n')        
        
    return args



    
# =============================================================================
# Definition of class CNNPredict
# =============================================================================

class CNNPredict:

    # ===================================
    # Init 
    # ===================================

    def __init__( self , 
                  file_model , 
                  type_arch   = None ,
                  num_classes = 2 ,
                  bottleneck  = False ):
        
        # Assign inputs to class attributes
        self._file_model  = file_model
        self._num_classes = num_classes
        self._bottleneck  = bottleneck


        # Get type of architecture
        self._get_type_architecture( type_arch )
        
        
        # Load model
        self._load_model()
        
        
        # Get target size
        self._get_target_size()
        
        
        # Load preprocessing
        self._load_preproc()
        
        
        
    # ===================================
    # Get type of architecture 
    # ===================================

    def _get_type_architecture( self , type_arch ):
        # Initialize class field
        self._type_arch = None
    
    
        # Case architecture type is specified by user
        if type_arch is not None:
            self._type_arch = type_arch.lower()
            
            
        # Case architecture type is not specified in the inputs
        else:
            for arch in LIST_ARCHS:
                if arch in self._file_model:
                    self._type_arch = arch
                    break


        # Check that architecture type was found
        if self._type_arch is None or self._type_arch not in LIST_ARCHS:
            sys.exit( '\nERROR ( CNNPredict -- _get_type_architecture ): ' + \
                      'architecture type not identified!\n' )
            
            
    
    # ===================================
    # Load model
    # ===================================

    def _load_model( self ):
        self._wcce_weight = wcce._ones_square_np( self._num_classes )
        self._model       = load_model( self._file_model , 
                                        custom_objects={ '_wcce': wcce.wrapped_partial(
                                                                  wcce._wcce, 
                                                                  weights = tf.constant( self._wcce_weight )
                                                                ) }  )
        self._num_classes = myint( self._model.output.shape[1] )
    
    
    
    # ===================================
    # Target size
    # ===================================

    def _get_target_size( self ):
        # Get target size
        try:
            size              = self._model.input.shape[1:]
            self._target_size = ( myint( size[0] ) , myint( size[1] ) )
        
        except:
            if self._type_arch == 'vgg16' or self._type_arch == 'vgg19' or \
               self._type_arch == 'resnet50' or self._type_arch == 'mobilenet':
                self._target_size = ( 224  , 224 )
                
            elif self._type_arch == 'inception-v3' or \
                 self._type_arch == 'inception-resnet-v2' or \
                 self._type_arch == 'xception':
                 self._target_size = ( 299 , 299 )
        
        
        # Check if target size corresponds to identified
        # architecture type
        if self._type_arch == 'vgg16' and self._target_size != (224,224):
            sys.exit( '\n\ERROR: architecture type ' + self._type_arch + \
                      'requires target size (224,224) not (' + str( self._target_size[0] ) + \
                      ',' + str( self._target_size[1] ) + ')\n' )
                      
        if self._type_arch == 'vgg19' and self._target_size != (224,224):
            sys.exit( '\n\ERROR: architecture type ' + self._type_arch + \
                      'requires target size (224,224) not (' + str( self._target_size[0] ) + \
                      ',' + str( self._target_size[1] ) + ')\n' ) 

        if self._type_arch == 'resnet50' and self._target_size != (224,224):
            sys.exit( '\n\ERROR: architecture type ' + self._type_arch + \
                      'requires target size (224,224) not (' + str( self._target_size[0] ) + \
                      ',' + str( self._target_size[1] ) + ')\n' )
                      
        if self._type_arch == 'mobilenet' and self._target_size != (224,224):
            sys.exit( '\n\ERROR: architecture type ' + self._type_arch + \
                      'requires target size (224,224) not (' + str( self._target_size[0] ) + \
                      ',' + str( self._target_size[1] ) + ')\n' )

        if self._type_arch == 'inception-v3' and self._target_size != (299,299):
            sys.exit( '\n\ERROR: architecture type ' + self._type_arch + \
                      'requires target size (299,299) not (' + str( self._target_size[0] ) + \
                      ',' + str( self._target_size[1] ) + ')\n' )

        if self._type_arch == 'inception-resnet-v2' and self._target_size != (299,299):
            sys.exit( '\n\ERROR: architecture type ' + self._type_arch + \
                      'requires target size (299,299) not (' + str( self._target_size[0] ) + \
                      ',' + str( self._target_size[1] ) + ')\n' )

        if self._type_arch == 'xception' and self._target_size != (299,299):
            sys.exit( '\n\ERROR: architecture type ' + self._type_arch + \
                      'requires target size (299,299) not (' + str( self._target_size[0] ) + \
                      ',' + str( self._target_size[1] ) + ')\n' )                      
    
        if self._type_arch == 'nasnet' and self._target_size != (224,224):
            sys.exit( '\n\ERROR: architecture type ' + self._type_arch + \
                      'requires target size (224,224) not (' + str( self._target_size[0] ) + \
                      ',' + str( self._target_size[1] ) + ')\n' )                      
    
 

    # ===================================
    # Load preprocessing 
    # ===================================
    
    def _load_preproc( self ):
        if self._type_arch =='vgg16':
            from keras.applications.vgg16 import preprocess_input
            self._preproc = preprocess_input
        
        elif self._type_arch == 'vgg19':
            from keras.applications.vgg19 import preprocess_input
            self._preproc = preprocess_input

        elif self._type_arch == 'inception-v3':
            from keras.applications.inception_v3 import preprocess_input
            self._preproc = preprocess_input
        
        elif self._type_arch == 'resnet50':
            from keras.applications.resnet50 import preprocess_input
            self._preproc = preprocess_input
            
        elif self._type_arch == 'xception':
            from keras.applications.xception import preprocess_input
            self._preproc = preprocess_input
        
        elif self._type_arch == 'inception-resnet-v2':
            from keras.applications.inception_resnet_v2 import preprocess_input
            self._preproc = preprocess_input
            
        elif self._type_arch == 'mobilenet':
            from keras.applications.mobilenet import preprocess_input
            self._preproc = preprocess_input
        
        elif self._type_arch == 'nasnet':
            from keras.applications.nasnet import preprocess_input
            self._preproc = preprocess_input
           
        elif self._type_arch == 'densenet':
            from keras.applications.densenet import preprocess_input
            self._preproc = preprocess_input
           
        else:
            self._preproc = None
    

    
    # ===================================
    # Run prediction
    # ===================================

    def _predict( self , filein ):
        # Assign filename to class
        self._file_in = filein

        
        # Open image
        img = image.load_img( filein , 
                              target_size = self._target_size )
        x   = image.img_to_array( img )

        
        # Transform image in a tensor compatible with Keras predict
        x = np.expand_dims( x , axis=0 )
  
  
        # Apply preprocessing required by the specific CNN model
        if self._preproc is not None:
            x = self._preproc( x )
    
    
        # Run prediction
        self._probs = self._model.predict( x )[0]

    
        # Predicted class
        self._pred_class = np.argwhere( self._probs == np.max( self._probs ) ).flatten()
        
        
        # Compute bottleneck features if enabled
        self._bfeats = None
        
        if self._bottleneck:
            self._bfeats = self._get_bottleneck_features( x )[0].reshape( -1 )


    
    # ===================================
    # Get bottleneck features
    # ===================================

    def _get_bottleneck_features( self , img ):
        last_layer = len( self._model.layers ) - 2 
        get_bfeats = K.function( [ self._model.layers[0].input , K.learning_phase() ] ,
                                 [ self._model.layers[last_layer].output , ] )
        bfeats     = get_bfeats( [ img , 0 ] )
    
        return bfeats
    
    
    
    # ===================================
    # Write result
    # ===================================

    def _write_result( self , pathout ):
        # Re-define output filename
        fileout = self._get_output_filename( pathout )
        
    
        # Open file
        fp = open( fileout , 'w' )
    
    
        # Get MIG url for the image
        filein_mig = self._create_mig_url( self._file_in )
        
        
        # Write header
        fp.write( '%s,%s,%s' % ( 'image' , 'image_mig_url' , 'prediction' ) )
        
        for i in range( len( self._probs ) ):
            fp.write( ',%s' % 'class n.' + str(i) + ' probability' )
            
        if self._bfeats is not None:
            for i in range( len( self._bfeats ) ):
                fp.write( ',%s' % 'bfeat n.' + str(i) )            
    
    
        # Write content
        fp.write( '\n%s,%s,%d' % ( self._file_in , filein_mig , self._pred_class ) )
        
        for i in range( len( self._probs ) ):
            fp.write( ',%.5e' % self._probs[i] )
            
        if self._bfeats is not None:
            for i in range( len( self._bfeats ) ):
                fp.write( ',%.7e' % self._bfeats[i] )            

        
        # Close file
        fp.close()
        
        
        # Print
        print( '\nSaved prediction result in:\n' , fileout,'\n' )
    
    
    
    # ===================================
    # Get output filename
    # ===================================

    def _get_output_filename( self , pathout ):
        if os.path.isdir( pathout ) is False:
            pathout = os.path.dirname( pathout )
        pathout = os.path.abspath( pathout )
        
        basename     = os.path.basename( self._file_in )
        basename , _ = os.path.splitext( basename )
        
        fileout      = os.path.join( pathout , basename + '_prediction.csv' )
        
        return fileout
    

    
    # ===================================
    # Create MIG url
    # ===================================
    
    def _create_mig_url( self , filein ):
        # Get absolute path and remove /pstore/data/pio/
        path_abs = os.path.abspath( filein )    
        path_rel = path_abs.replace( '/pstore/data/pio/' , '' )
    
    
        # Construct MIG urls
        mig_base = 'http://rbalv-a4-dev.bas.roche.com:8090/aim4eye/get-fundus-photo?image='
        mig_url  = mig_base + path_rel + '&width=800'
    
        return mig_url
        
        
        
        
# =============================================================================
# Main 
# =============================================================================

def main():
    # Start timing
    time1 = time.time()
    print( '\n\nCNN PREDICT\n' )


    # Get arguments
    args = _get_args()
    
    
    # Initialize class
    cnn = CNNPredict( args.model ,
                      type_arch   = args.type_arch ,
                      num_classes = args.num_classes , 
                      bottleneck  = args.bottleneck )
    
    
    # Some prints
    print( '\nModel file: ', cnn._file_model )
    print( 'Architecture type: ', cnn._type_arch )
    print( 'Target size: ', cnn._target_size )
    
    if cnn._bottleneck:
        print( 'Enabled computation of bottleneck features' )
    
    if cnn._preproc is None:
        print( 'No preprocessing of the data enabled' )
    else:
        print( 'Preprocessing required by ', cnn._type_arch,' enabled' )
    
    
    # Run predictions
    print( '\nInput file: ', args.file_in )
    print( '\nRun prediction ...' )
    cnn._predict( args.file_in )
    print( '\nProbability vector: ', cnn._probs )
    print( 'Predicted class: ', cnn._pred_class )
    
    
    # Write results to file
    cnn._write_result( args.path_out )
    
    
    # Final print
    diff  = time.time() - time1
    print( '\nTime elapsed: ', diff , '\n\n' )
    
    
    
    
# =============================================================================
# Call to main
# =============================================================================

if __name__ == '__main__':
    main()
