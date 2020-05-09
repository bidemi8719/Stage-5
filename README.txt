# ================= Project python describtion ============

python main.py RegularCNN

"""
the report is titled 'Stage 5 Machine learning project' as pdf file
"""


"""
after preprocessing the data with scripts 'preprocessing' the preprocessed data with dimension 
(2602, 50, 50, 3) was splited and saved as saved with 'y_train.pickle', 'y_test.pickle', 'X_train.pickle', 'X_test.pickle'
the data for support vector machine was reduced using Principal Component Analysis (PCA) feature reduction technique. the reduced data 
was saved with file title 'X_train_pca.pickle', 'X_test_pca.pickle'

"""

""" 
there are three different CNN scripts and one SVM scripts 
1) script with title 'CNN_Model_Regular' is for regular CNN model
2) script with title 'CNN_Model_with_DataAugmentation' for CNN model with Data Augmentation 
3) script with title 'CNN_Model_with_dropout' for CNN with dropout regularized techniques 
4) script with title 'SVM' for support vector machine 

"""

"""
After training the data, the trained model was saved as follows 
'CNN_dropout_Chile_Disease_vs_Normal_15'
'CNN_dropout_Chile_Disease_vs_Normal_30'
'CNN_regular_Chile_Disease_vs_Normal_15'
'CNN_regular_Chile_Disease_vs_Normal_30'
'CNN_DataAugmentation_Chile_Disease_vs_Normal_15'
'CNN_DataAugmentation_Chile_Disease_vs_Normal_30'
"""

