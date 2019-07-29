'''
Package to compute metrics to benchmark ML models
'''


# Author: Filippo Arcadu 
# Date: 14/01/2019
# PHCOSS -- PIOCNN-15



from __future__ import print_function , division
import numpy as np
from sklearn.metrics import mean_squared_error, cohen_kappa_score, \
                            precision_score , recall_score , f1_score , \
                            roc_auc_score , roc_curve , auc , r2_score , \
                            accuracy_score , confusion_matrix




# =============================================================================
# Types 
# =============================================================================

myint    = np.int
myfloat  = np.float32
myfloat2 = np.float64




# =============================================================================
# Types of tasks 
# =============================================================================

TASKS = [ 'class' , 'mclass' , 'regr' , 'segm' ]




# =============================================================================
# Class Metrics
# =============================================================================

class Metrics:
    
    # ===================================
    # Init 
    # ===================================

    def __init__( self , task='class' , n_classes=None ):

        # Assign type of task to class
        # Task can be:
        #     1) "class"   ---> binary classification
        #     2) "mclass"  ---> multi-class classification
        #     3) "regr"    ---> regression
        #     4) "segm"    ---> segmentation
        self._task      = task
        self._n_classes = n_classes

        
        # Check whether task is admissable
        self._is_entry_in_list( TASKS , task )


        # Set to false the flag indicating whether
        # the confidence interval through bootstrapping 
        # has been computed or not
        self._flag_ci = False



    # ===================================
    # Check whether entry  is in list
    # ===================================

    def _is_entry_in_list( self , list_entries , entry ):
        if entry in list_entries:
            flag = False
        else:
            flag = True

        if flag:
            string = '\nERROR ( Metrics -- _is_entry_in_list ): entry "' + str( entry ) + \
                     '" does not correspond to any item of the list [' + ','.join( TASKS ) + ']!\n\n'  
            raise ValueError( string )



    # ===================================
    # Compute metrics
    # ===================================

    def _run( self , y_true , y_pred ):
        # Check that the 2 input vectors are ok    
        self._check_input_vectors( y_true , y_pred )


        # Case 1: binary classification
        if self._task == 'class':
            self._run_class( y_true , y_pred )


        # Case 2: multi-label classification
        elif self._task == 'mclass':
            self._run_mclass( y_true , y_pred )


        # Case 3: regression
        elif self._task == 'regr':
            self._run_regr( y_true , y_pred )


        # Case 4: segmentation
        elif self._task == 'segm':
            self._run_segm( y_true , y_pred )


    
    # ===================================
    # Check input vectors
    # ===================================

    def _check_input_vectors( self , y_true , y_pred ):
        # Check for binary classification
        if self._task == 'class':
            if len( y_pred[0] ) != 2:
                raise Exception( '\nMetrics -- _check_input_vectors: y_pred shape ', y_pred.shape , ' is ' \
                                 'not what expected for binary classification in this case ', ( len( y_true ) , 2 ) , ')!\n' )


        # Checks for multi-label classification
        elif self._task == 'mclass':
            if len( y_pred[0] ) != self._n_classes:
                raise Exception( '\nMetrics -- _check_input_vectors: y_pred shape ' , y_pred.shape , ' is ' \
                                 'not what expected for binary classification in this case ' , ( len( y_true ) , n_labels ) , ')!\n' )

           
        # Checks for multi-label classification
        elif self._task == 'regr':
            if len( y_pred[0] ) != 1:
                raise Exception( '\nMetrics -- _check_input_vectors: y_pred shape ' , y_pred.shape , ' is ' \
                                 'not what expected for binary classification in this case ' , ( len( y_true ) , 1 ) ,  ')!\n' )


   
    # ===================================
    # Case 1: Compute metrics for binary classification
    # ===================================

    def _run_class( self , y_true , y_pred ):
        # Transform probabilities in 1-class prediction
        y_class = y_pred.argmax( axis=1 ).astype( myint )


        # Compute metrics for 2-class problem
        try:
            self._accuracy      = accuracy_score( y_true , y_class )
            self._precision     = precision_score( y_true , y_class )
            self._recall        = recall_score( y_true , y_class )
            self._f1score       = f1_score( y_true , y_class )
            self._cohen_kappa   = cohen_kappa_score( y_true , y_class )
            self._conf_matrix   = confusion_matrix( y_true , y_class )
            
            self._fpr , self._tpr , self._thres = roc_curve( y_true , y_pred[:,1] )
            
            self._auc = auc( self._fpr , self._tpr )
            
            self._sensitivity , self._specificity , self._thres_best , self._fpr_best , self._tpr_best = self._youden_point()
            
            y_thres                                    = np.zeros( len( y_true ) )
            y_thres[ y_pred[:,1] >= self._thres_best ] = 1
            self._conf_matrix_youden                   = confusion_matrix( y_true , y_thres ) 
 
        except:
            self._accuracy    = self._precision   = self._recall = \
            self._f1score     = self._cohen_kappa = self._auc    = \
            self._sensitivity = self._specificity = self._thres_best = 0.0

            
    
    # ===================================
    # Compute Youden's point
    # ===================================

    def _youden_point( self ):
        # Compute best operating point on ROC curve, define as 
        # the one maximizing the difference ( TPR - FPR ), also called
        # Youden operating point ( https://en.wikipedia.org/wiki/Youden%27s_J_statistic )
        diff  = self._tpr - self._fpr
        i_max = np.argwhere( diff == np.max( diff ) )

        thres_best = self._thres[ i_max ][0]

        fpr_best = self._fpr[i_max]
        tpr_best = self._tpr[i_max]
                    
        
        # Compute sensitivity and specificity at optimal operating point
        sensitivity = self._tpr[ i_max ][0]
        specificity = 1 - self._fpr[ i_max ][0]
        
        return sensitivity[0] , specificity[0] , thres_best , fpr_best , tpr_best



    # ===================================
    # Case 2: Compute metrics for multi-label classification
    # ===================================

    def _run_mclass( self , y_true , y_pred ):
        # Transform probabilities in 1-class prediction
        y_class = y_pred.argmax( axis=1 ).astype( myint )

    
        # Compute metrics for multi-label classification
        self._accuracy    = accuracy_score( y_true , y_class )
        self._cohen_kappa = cohen_kappa_score( y_true , y_class )
 


    # ===================================
    # Case 3: Compute metrics for regression
    # ===================================

    def _run_regr( self , y_true , y_pred ):
        # Compute metrics for regression
        self._mse  = mean_squared_error( y_true , y_pred )    
        self._r2   = r2_score( y_true , y_pred )
 

    
    # ===================================
    # Case 4: Compute metrics for segmentation
    #         
    # N.B.: It computes intersection-over-union
    #       and dice coefficient for just one pair
    #       of mask and segmentation, i.e. < y_true >
    #       stores a single mask and < y_pred > a 
    #       single segmentation
    # ===================================

    def _run_segm( self , y_true , y_pred ):
        # Compute metrics for segmentation
        self._iou  = self._jaccard( self , y_true , y_pred )    
        self._dice = self._dice( self , y_true , y_pred )

 
    
    # ===================================
    # Jaccard coefficient or Intersection over union
    # 
    # Definition on Wikipedia: https://en.wikipedia.org/wiki/Jaccard_index
    #
    # J(A,B) = M11 / ( M01 + M10 + M11 )
    # Mij: where A has value i and and B has value j
    #
    # y_true should be binary, i.e. {0,1}
    # y_pred should be real in [0,1]
    # ===================================

    def _jaccard( self , y_true , y_pred ):
        # Binarize y_pred
        y_pred_bin                 = np.zeros( y_pred.shape , dtype=myfloat )
        y_pred_bin[ y_pred > 0.5 ] = 1.0


        # Compute the components
        M11 = len( np.argwhere( ( y_true == 1 ) & ( y_pred_bin == 1 ) ) )
        M01 = len( np.argwhere( ( y_true == 0 ) & ( y_pred_bin == 1 ) ) )
        M10 = len( np.argwhere( ( y_true == 1 ) & ( y_pred_bin == 0 ) ) )


        # Compute Jaccard
        iou = myfloat( M11 ) / myfloat( M01 + M10 + M11 )

        return iou

    
    
    # ===================================
    # Sorensen-Dice coefficient
    # 
    # Definition on Wikipedia: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    #
    # D(A,B) = 2*M11 / ( 2 * M11 + M01 + M10 )
    # Mij: where A has value i and and B has value j
    #
    # y_true should be binary, i.e. {0,1}
    # y_pred should be real in [0,1]
    # ===================================

    def _dice( self , y_true , y_pred ):
        # Binarize y_pred
        y_pred_bin                 = np.zeros( y_pred.shape , dtype=myfloat )
        y_pred_bin[ y_pred > 0.5 ] = 1.0


        # Compute the components
        M11 = len( np.argwhere( ( y_true == 1 ) & ( y_pred_bin == 1 ) ) )
        M01 = len( np.argwhere( ( y_true == 0 ) & ( y_pred_bin == 1 ) ) )
        M10 = len( np.argwhere( ( y_true == 1 ) & ( y_pred_bin == 0 ) ) )


        # Compute dice
        dice = myfloat( 2 * M11 ) / myfloat( M01 + M10 + 2 * M11 )

        return dice



    # ===================================
    # Compute confidence interval through
    # bootstrapping
    # ===================================
    
    def _ci_bstrap( self , y_true , y_pred , level=95 , n_bstraps=1000 ):
        # Define number of bstraps
        if len( y_true ) < n_bstraps:
            self._n_bstraps = len( y_true )
        else:
            self._n_bstraps = n_bootstrap

        
        # Assign confidence level to class
        self._level = level


        # Set flag to true, needed for printing summary
        # that includes the confidence interval
        self._flag_ci = True

       
        # Case 1: Bootstrapping for classification
        if self._task == 'class':
            self._ci_bstrap_class( y_true , y_pred , level=self._level , n_bstraps=self._n_bstraps  )


        # Case 2: Bootstrapping for multi-label classification
        elif self._task == 'mclass':
            self._ci_bstrap_mclass( y_true , y_pred , level=self._level , n_bstraps=self._n_bstraps  )


        # Case 3: Bootstrapping for regression
        elif self._task == 'regr': 
            self._ci_bstrap_regr( y_true , y_pred , level=self._level , n_bstraps=self._n_bstraps  )


    
    # ===================================
    # Case 1: Compute confidence interval through
    #         bootstrapping for the classification
    # ===================================
 
    def _ci_bstrap_class( self , y_true , y_pred , level=95 , n_bstraps=1000 ):
        # Initialize lists
        bstrap_accuracy    = []
        bstrap_auc         = []
        bstrap_sensitivity = []
        bstrap_specificity = []
        bstrap_precision   = []
        bstrap_recall      = []
        bstrap_f1score     = []
        bstrap_cohen_kappa = []
             
        
        # Transform probabilities in 1-class prediction
        y_class = y_pred.argmax( axis=1 ).astype( myint )

       
        # Do bootstrapping
        ind_all = np.arange( len( y_true ) )
        
        for i in range( n_bstraps ):
            ind = np.random.choice( ind_all , len( ind_all ) - 1 )
            
            if 1:
                accuracy          = accuracy_score( y_true[ind] , y_class[ind] )
                precision         = precision_score( y_true[ind] , y_class[ind] )
                recall            = recall_score( y_true[ind] , y_class[ind] )
                f1score           = f1_score( y_true[ind] , y_class[ind] )
                cohen_kappa       = cohen_kappa_score( y_true[ind] , y_class[ind] )
                fpr , tpr , thres = roc_curve( y_true[ind] , y_pred[ind,1] )
                auc_value         = auc( fpr , tpr )
                
                sensitivity , specificity , threshold , _ , _ = self._youden_point_bstrap( tpr , fpr , thres )
 
                bstrap_accuracy.append( accuracy )
                bstrap_precision.append( precision )
                bstrap_recall.append( recall )
                bstrap_f1score.append( f1score )
                bstrap_cohen_kappa.append( cohen_kappa )
                bstrap_auc.append( auc_value )
                bstrap_sensitivity.append( sensitivity )
                bstrap_specificity.append( specificity )
                
            else:
                pass
            
           
        # Convert to array and sort
        bstrap_accuracy = np.array( bstrap_accuracy ).reshape( -1 )
        bstrap_accuracy.sort()

        bstrap_precision = np.array( bstrap_precision ).reshape( -1 )
        bstrap_precision.sort()

        bstrap_recall = np.array( bstrap_recall ).reshape( -1 )
        bstrap_recall.sort()

        bstrap_f1score = np.array( bstrap_f1score ).reshape( -1 )
        bstrap_f1score.sort()

        bstrap_cohen_kappa = np.array( bstrap_cohen_kappa ).reshape( -1 )
        bstrap_cohen_kappa.sort() 

        bstrap_auc = np.array( bstrap_auc ).reshape( -1 )
        bstrap_auc.sort()

        bstrap_sensitivity = np.array( bstrap_sensitivity ).reshape( -1 )
        bstrap_sensitivity.sort()

        bstrap_specificity = np.array( bstrap_specificity ).reshape( -1 )
        bstrap_specificity.sort()

        
        # Get confidence interval
        thres = ( 100 - level ) / 100.0 * 0.5
        n_el  = len( bstrap_auc )
        
        self._accuracy_ci_up   = bstrap_accuracy[ myint( ( 1 - thres ) * n_el ) ]
        self._accuracy_ci_down = bstrap_accuracy[ myint( thres * n_el ) ]

        self._precision_ci_up   = bstrap_precision[ myint( ( 1 - thres ) * n_el ) ]
        self._precision_ci_down = bstrap_precision[ myint( thres * n_el ) ]

        self._recall_ci_up   = bstrap_recall[ myint( ( 1 - thres ) * n_el ) ]
        self._recall_ci_down = bstrap_recall[ myint( thres * n_el ) ]

        self._f1score_ci_up   = bstrap_accuracy[ myint( ( 1 - thres ) * n_el ) ]
        self._f1score_ci_down = bstrap_accuracy[ myint( thres * n_el ) ]

        self._cohen_kappa_ci_up   = bstrap_cohen_kappa[ myint( ( 1 - thres ) * n_el ) ]
        self._cohen_kappa_ci_down = bstrap_cohen_kappa[ myint( thres * n_el ) ]

        self._auc_ci_up   = bstrap_auc[ myint( ( 1 - thres ) * n_el ) ]
        self._auc_ci_down = bstrap_auc[ myint( thres * n_el ) ]

        self._sensitivity_ci_up   = bstrap_sensitivity[ myint( ( 1 - thres ) * n_el ) ]
        self._sensitivity_ci_down = bstrap_sensitivity[ myint( thres * n_el ) ]

        self._specificity_ci_up   = bstrap_specificity[ myint( ( 1 - thres ) * n_el ) ]
        self._specificity_ci_down = bstrap_specificity[ myint( thres * n_el ) ]

    
    
    # ===================================
    # Compute Youden's point for bootstrapping
    # ===================================

    def _youden_point_bstrap( self , tpr , fpr , thres ):
        # Compute best operating point on ROC curve, define as 
        # the one maximizing the difference ( TPR - FPR ), also called
        # Youden operating point ( https://en.wikipedia.org/wiki/Youden%27s_J_statistic )
        diff  = tpr - fpr
        i_max = np.argwhere( diff == np.max( diff ) )

        thres_best = thres[ i_max ][0]

        fpr_best = fpr[i_max]
        tpr_best = tpr[i_max]
                    
        
        # Compute sensitivity and specificity at optimal operating point
        sensitivity = tpr[ i_max ][0]
        specificity = 1 - fpr[ i_max ][0]
        
        return sensitivity[0] , specificity[0] , thres_best , fpr_best , tpr_best

   
    
    # ===================================
    # Case 2: Compute confidence interval through
    #         bootstrapping for multi-label classification
    # ===================================
 
    def _ci_bstrap_mclass( self , y_true , y_pred , level=95 , n_bstraps=1000 ):
        # Initialize lists
        bstrap_accuracy    = []
        bstrap_cohen_kappa = []
             
        
        # Do bootstrapping
        ind_all = np.arange( len( y_true ) )
        
        for i in range( n_bstraps ):
            ind = np.random.choice( ind_all , len( ind_all ) - 1 )
            
            try:
                accuracy          = accuracy_score( y_true[ind] , y_class[ind] )
                cohen_kappa       = cohen_kappa_score( y_true[ind] , y_class[ind] )
                
                bstrap_accuracy.append( accuracy )
                bstrap_cohen_kappa.append( cohen_kappa )
                
            except:
                pass
            
           
        # Convert to array and sort
        bstrap_accuracy = np.array( accuracy ).reshape( -1 )
        bstrap_accuracy.sort()

        bstrap_cohen_kappa = np.array( cohen_kappa ).reshape( -1 )
        bstrap_cohen_kappa.sort() 

        
        # Get confidence interval
        thres = ( 100 - level ) / 100.0 * 0.5
        n_el  = len( bstrap_auc )
        
        self._accuracy_ci_up   = bstrap_accuracy[ myint( ( 1 - thres ) * n_el ) ]
        self._accuracy_ci_down = bstrap_accuracy[ myint( thres * n_el ) ]

        self._cohen_kappa_ci_up   = bstrap_cohen_kappa[ myint( ( 1 - thres ) * n_el ) ]
        self._cohen_kappa_ci_down = bstrap_cohen_kappa[ myint( thres * n_el ) ]

   
    
    # ===================================
    # Case 3: Compute confidence interval through
    #         bootstrapping for regression
    # ===================================
 
    def _ci_bstrap_mclass( self , y_true , y_pred , level=95 , n_bstraps=1000 ):
        # Compute metrics for regression
        self._mse  = mean_squared_error( y_true , y_pred )    
        self._r2   = r2_score( y_true , y_pred )
 

        
        # Initialize lists
        bstrap_mse = []
        bstrap_r2  = []
             
        
        # Do bootstrapping
        ind_all = np.arange( len( y_true ) )
        
        for i in range( n_bstraps ):
            ind = np.random.choice( ind_all , len( ind_all ) - 1 )
            
            try:
                mse = mean_squared_error( y_true[ind] , y_pred[ind] )
                r2  = cohen_kappa_score( y_true[ind] , y_pred[ind] )
                
                bstrap_mse.append( mse )
                bstrap_r2.append( r2 )
                
            except:
                pass
            
           
        # Convert to array and sort
        bstrap_mse = np.array( bstrap_mse ).reshape( -1 )
        bstrap_mse.sort()

        bstrap_r2 = np.array( bstrap_r2 ).reshape( -1 )
        bstrap_r2.sort() 

        
        # Get confidence interval
        thres = ( 100 - level ) / 100.0 * 0.5
        n_el  = len( bstrap_auc )
        
        self._mse_ci_up   = bstrap_mse[ myint( ( 1 - thres ) * n_el ) ]
        self._mse_ci_down = bstrap_mse[ myint( thres * n_el ) ]

        self._r2_ci_up   = bstrap_r2[ myint( ( 1 - thres ) * n_el ) ]
        self._r2_ci_down = bstrap_r2[ myint( thres * n_el ) ]

   
    
    # ===================================
    # Print summary of the metrics
    # ===================================

    def _summary( self ):
        print( '\n' )

        if self._task == 'class':
            self._summary_class()

        elif self._task == 'mclass':
            self._summary_mclass()

        elif self._task == 'regr':
            self._summary_regr()

        elif self._task == 'segm':
            self._summary_segm()

        print( '\n' )


    def _summary_class( self ):
        if self._flag_ci:
            print( '\tAccuracy: ', self._accuracy , ' with CI(', self._level,'%) = (', self._accuracy_ci_down,' , ', self._accuracy_ci_up,')' )
            print( '\tPrecision: ', self._precision,  ' with CI(', self._level,'%) = (', self._precision_ci_down,' , ', self._precision_ci_up,')' )
            print( '\tRecall: ', self._recall,  ' with CI(', self._level,'%) = (', self._recall_ci_down,' , ', self._recall_ci_up,')' )
            print( '\tF1-score: ', self._f1score , ' with CI(', self._level,'%) = (', self._f1score_ci_down,' , ', self._f1score_ci_up,')' )
            print( '\tCohen Kappa: ', self._cohen_kappa,  ' with CI(', self._level,'%) = (', self._cohen_kappa_ci_down,' , ', self._cohen_kappa_ci_up,')' )
            print( '\tAUC: ', self._auc,  ' with CI(', self._level,'%) = (', self._auc_ci_down,' , ', self._auc_ci_up,')' )
            print( '\tYouden threshold: ', self._thres,  ' with CI(', self._level,'%) = (', self._thres_ci_down,' , ', self._thres_ci_up,')' )
            print( '\tSensitivity: ', self._sensitivity,  ' with CI(', self._level,'%) = (', self._sensitivity_ci_down,' , ', self._sensitivity_ci_up,')' )
            print( '\tSpecificity: ', self._specificity, ' with CI(', self._level,'%) = (', self._specificity_ci_down,' , ', self._specificity_ci_up,')' )
            print( '\tConfusion matrix at 0.5:\n' , self._conf_matrix )
            print( '\tConfusion matrix at Youden point:\n' , self._conf_matrix_youden )

        else:
            print( '\tAccuracy: ', self._accuracy )
            print( '\tPrecision: ', self._precision )
            print( '\tRecall: ', self._recall )
            print( '\tF1-score: ', self._f1score )
            print( '\tCohen Kappa: ', self._cohen_kappa )
            print( '\tAUC: ', self._auc )
            print( '\tYouden threshold: ', self._thres )
            print( '\tSensitivity ( ' , self._thres, '): ', self._sensitivity )
            print( '\tSpecificity ( ' , self._thres, '): ', self._specificity )

     
    def _summary_mclass( self ):
        if self._flag_ci:
            print( '\tAccuracy: ', self._accuracy , ' with CI(', self._level,'%) = (', self._accuracy_ci_down,' , ', self._accuracy_ci_up,')' )
            print( '\tCohen Kappa: ', self._cohen_kappa , ' with CI(', self._level,'%) = (', self._accuracy_ci_down,' , ', self._accuracy_ci_up,')' )

        else:
            print( '\tAccuracy: ', self._accuracy )
            print( '\tCohen Kappa: ', self._cohen_kappa )

    
    def _summary_regr( self ):
        if self._flag_ci:
            print( '\tMSE: ', self._mse , ' with CI(', self._level,'%) = (', self._mse_ci_down,' , ', self._mse_ci_up,')' )
            print( '\tR2 score: ', self._r2score , ' with CI(', self._level,'%) = (', self._r2score_ci_down,' , ', self._r2score_ci_up,')' )

        else:
            print( '\tMSE: ', self._mse )
            print( '\tR2 score: ', self._r2score )


    def _summary_segm( self ):
        print( '\tIoU: ', self._iou )
        print( '\tDice: ', self._dice )


