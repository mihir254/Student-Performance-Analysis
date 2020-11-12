import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

class StudentPerformance:
    
    def failed(self, mathScore):
        if(mathScore<50): 
            return 1
        else: 
            return 0
    
    def is_categorical(self, array_like):
        return array_like.dtype.name == 'category'
    
    def readCsv(self, file_name):
        #Importing the datset
        self.dataset = pd.read_csv(file_name);
        self.dataset['math score'].replace('', np.nan, inplace=True)
        self.dataset['reading score'].replace('', np.nan, inplace=True)
        self.dataset['math score'].replace(np.nan, self.dataset['math score'].median(), inplace=True)
        self.dataset['reading score'].replace(np.nan, self.dataset['reading score'].median(),inplace=True)
        self.dataset["gender"] = self.dataset["gender"].astype("category")
        self.dataset["race/ethnicity"] = self.dataset["race/ethnicity"].astype("category")
        self.dataset["parental level of education"] = self.dataset["parental level of education"].astype("category") 
        self.dataset["lunch"] = self.dataset["lunch"].astype("category")
        self.dataset["test preparation course"] = self.dataset["test preparation course"].astype("category") 
        self.dataset["total"] = self.dataset["math score"] +self.dataset["reading score"]+self.dataset["writing score"]
        prepared_Data = self.dataset
        y = list(map(self.failed, prepared_Data["math score"]))
        prepared_Data["y"] = prepared_Data["math score"].apply(lambda x: self.failed(x) )
        prepared_Data = prepared_Data.drop(columns= ["math score","total", "lunch"] )
        
        catFilter = [self.is_categorical(prepared_Data.iloc[:,i])  for i in range(0, len(prepared_Data.columns) )] 
        categoricalCols = prepared_Data.columns[catFilter].tolist()
        cat_vars= categoricalCols
        for var in cat_vars:
            cat_list = "var"+"_" +var
            cat_list = pd.get_dummies(prepared_Data[var],drop_first=True, prefix=var)
            df1= prepared_Data.join(cat_list)
            prepared_Data= df1
        cat_vars= categoricalCols
        data_vars=prepared_Data.columns.values.tolist()
        to_keep=[i for i in data_vars if i not in cat_vars]
        finalDf = prepared_Data[to_keep]       
        
        self.X = finalDf.loc[: , finalDf.columns != "y"]
        self.y = finalDf.loc[: , finalDf.columns == "y"]
        from imblearn.over_sampling import SMOTE
        os = SMOTE(random_state=0)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=0)        
        columns = self.X_train.columns
        self.X,self.y=os.fit_sample(self.X_train, self.y_train.values.ravel()) 
        self.X = pd.DataFrame(data=self.X,columns=columns )
        self.y= pd.DataFrame(data=self.y,columns=['y'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=45)

    def trainLR(self):
        self.logreg = LogisticRegression(max_iter=100000)
        self.logreg.fit(self.X_train, self.y_train.values.ravel())
        
    def trainSVM(self):
        self.classifierSVM = SVC(kernel = 'linear' , C=10)
        self.classifierSVM.fit(self.X_train, self.y_train.values.ravel())
        
    def trainRF(self):
        self.model = RandomForestClassifier()
        self.model.fit(self.X_train, self.y_train.values.ravel())
        
    def LogisticRegression(self):
        self.y_pred = self.logreg.predict(self.X_test)
        cm = confusion_matrix(self.y_test, self.y_pred)
        #print(cm)
        accuracies = cross_val_score(estimator = self.logreg, X = self.X_train, y = self.y_train.values.ravel(), cv = 10, scoring="accuracy")
        #print(accuracies.mean() )
        #print(accuracies.std() )
        return cm,accuracies.mean()*100
    
    def SVM(self):
        self.y_pred2 = self.classifierSVM.predict(self.X_test)
        cm2 = confusion_matrix(self.y_test, self.y_pred2)
        #print(cm2)
        accuraciesSVM = cross_val_score(estimator = self.classifierSVM, X = self.X_train, y = self.y_train.values.ravel(), cv = 10)
        #print(accuraciesSVM.mean())
        #print(accuraciesSVM.std())
        return cm2,accuraciesSVM.mean()*100
    
    def RandomForest(self):
        self.y_pred = self.model.predict(self.X_test)
        cm3 = confusion_matrix(self.y_test, self.y_pred)
        #print(cm3)
        accuraciesRF = cross_val_score(estimator = self.model, X = self.X_train, y = self.y_train.values.ravel(), cv = 10)
        #print(accuraciesRF.mean() )
        #print(accuraciesRF.std() )
        return cm3,accuraciesRF.mean()*100
    
    def predict(self,gender,race,parent,test_prep,reading,writing):
       
        if(gender=="female"):
            g=0
        else:
            g=1
            
        if(race=="group A"):
            b=0
            c=0
            d=0
            e=0
        elif(race=="group B"):
            b=1
            c=0
            d=0
            e=0
        elif(race=="group C"):
            b=0
            c=1
            d=0
            e=0
        elif(race=="group D"):
            b=0
            c=0
            d=1
            e=0
        else:
            b=0
            c=0
            d=0
            e=1
            
        if(parent=="bachelor's degree"):
            bd=1
            hs=0
            md=0
            sc=0
            shs=0
        elif(parent=="associate's degree"):
            bd=0
            hs=0
            md=0
            sc=0
            shs=0
        elif(parent=="high school"):
            bd=0
            hs=1
            md=0
            sc=0
            shs=0
        elif(parent=="master's degree"):
            bd=0
            hs=0
            md=1
            sc=0
            shs=0
        elif(parent=="some college"):
            bd=0
            hs=0
            md=0
            sc=1
            shs=0
        else:
            bd=0
            hs=0
            md=0
            sc=0
            shs=1
            
        if(test_prep=="none"):
            tp=1
        else:
            tp=0
        
        ls = [reading, writing, g, b, c, d, e, bd, hs, md, sc, shs, tp]
        lst = [];
        lst.append(ls);
        
        temp1 = self.model.predict(lst)
        temp1 = temp1.tolist()
        
        temp2 = self.logreg.predict(lst)
        temp2 = temp2.tolist()
        
        temp3 = self.classifierSVM.predict(lst)
        temp3 = temp3.tolist()
        
        res = [temp1, temp2, temp3]
        #print(res)
        return res