# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:58:33 2020

@author: TEMITAYO
"""

class svm_model(object): 
    
    def __init__(self, X_train='X_train', X_test = 'X_test', y_train='y_train', y_test = 'y_test', gamma = 0.01, kernel= ''):   
                
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test     
        self.kernel = kernel
        self.gamma = gamma
        
        from sklearn.svm import SVC
        from datetime import datetime

        #------------------ (2a) for linear svm ----------------------------
        if  self.kernel == 'linear':
            print('\n For SVM with Kernel = Linear')
            for i in [10, 20, 30, 50, 100, 150, 200]:                
                start = datetime.now()
                svm_lnr = SVC(kernel = 'linear', random_state = 1, gamma = self.gamma, C= i )
                svm_lnr.fit(self.X_train, self.y_train)
                # Now predict the value of the digit:
                y_pred = svm_lnr.predict(self.X_test)
                misClass = (self.y_test != y_pred).sum()
                misClass = (self.y_test != y_pred).sum()
                accuracy = svm_lnr.score(self.X_test, self.y_test) 
                print('C = %d, gamma = %.3f Misclassified samples = %d, Accuracy = %.2f,' % (i, self.gamma, misClass, accuracy), "Run Time:",(datetime.now() - start))
                
        elif self.kernel == 'rbf':
            #------------------ (2b) for svm ------------------------------------
            print('\n For SVM with Kernel = RBF')
            for i in [10, 20, 30, 50, 100, 150, 200]: 
                start = datetime.now()
                svm_rbf = SVC(kernel = 'rbf', random_state = 1, C=i )
                svm_rbf.fit(self.X_train, self.y_train)        
                # Now predict the value of the digit:
                y_pred = svm_rbf.predict(self.X_test)
                misClass = (self.y_test != y_pred).sum()
                misClass = (self.y_test != y_pred).sum()
                accuracy = svm_rbf.score(self.X_test, self.y_test) 
                print('C = %d, Misclassified samples = %d, Accuracy = %.2f,' % (i, misClass, accuracy), "Run Time:",(datetime.now() - start))
                    

