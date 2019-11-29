import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score

# def trainDTree(data):
#     feature_cols=['tid1','tid2']
#     X=data[feature_cols]
#     Y=data.label
#     print(X,Y)
#     dTree = DecisionTreeClassifier()
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1) # 70% training and 30% test
#     dTree = dTree.fit(X_train,Y_train)
#     Y_pred = dTree.predict(X_test)
#     print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred),Y_test,Y_pred)
#     pass

def getDatasetPath():
    file=open('dataset.txt','r')
    lines=file.readlines()
    data_dict=dict()
    for line in lines:
        val=line.rstrip().split("=")
        data_dict[val[0]]=val[1]
    return data_dict['train_data'],data_dict['validation_data'],data_dict['test_data']

def trainData(dataset_path):
    train_df=pd.read_csv(dataset_path,sep=",",error_bad_lines=False)
    train_df=train_df.drop(['id', 'tid1', 'tid2','title1_zh','title2_zh'],axis=1)
    train_df=train_df.replace(np.nan, '', regex=True)
    text_vect=TfidfVectorizer()
    text_vect.fit(train_df['title1_en'])
    print(text_vect)
    # text_clf = Pipeline([('vect', TfidfVectorizer()),
    #                   ('clf', MultinomialNB()) ])
    # text_clf.fit(train_df['title1_en']+train_df['title2_en'],train_df['label'])
    print("Classifier trained using data from : "+dataset_path)
    #return text_clf

def validateData(dataset_path,clsf):
    validate_df=pd.read_csv(dataset_path,sep=",",error_bad_lines=False)
    validate_df=validate_df.drop(['tid1', 'tid2','title1_zh','title2_zh'],axis=1)
    validate_df=validate_df.replace(np.nan, '', regex=True)
    predicted = clsf.predict(validate_df['title1_en']+validate_df['title2_en'])
    validate_df['predicted']=predicted
    write_file=open('validate_result.csv','w')
    write_file.write('id,label,prediction\n')
    for i in range(validate_df.shape[0]):
        write_file.write(validate_df['id'][i]+','+validate_df['label'][i]+','+validate_df['predicted'][i]+'\n')
    write_file.close()
    print("Validation completed on data at : "+dataset_path)
    print('Accuracy achieved is ' + str(np.mean(predicted == validate_df['label'])))
    print("Validation results available at : validate_result.csv")

def testData(dataset_path,clsf):
    test_df=pd.read_csv(dataset_path,sep=",",error_bad_lines=False)
    test_df=test_df.drop(['tid1', 'tid2','title1_zh','title2_zh'],axis=1)
    test_df=test_df.replace(np.nan, '', regex=True)
    predicted = clsf.predict(test_df['title1_en']+test_df['title2_en'])
    test_df['label']=predicted
    write_file=open('submission.csv','w')
    write_file.write('id,label\n')
    for i in range(test_df.shape[0]):
        write_file.write(test_df['id'][i]+','+test_df['label'][i]+'\n')
    write_file.close()
    print("Testing Complete. Test results available at : submission.csv")

def main():
    #training_data=pd.read_csv("/home/tarunlolla/SMM/Project2/Project2_Files/train.csv",sep=",",header=1,usecols=["id","tid1","tid2","title1_en","title2_en","label"],error_bad_lines=False)
    train_data,validation_data,test_data=getDatasetPath()
    clsf=trainData(train_data)
    # train_df=pd.read_csv(train_data,sep=",",error_bad_lines=False)
    # train_df=train_df.drop(['id', 'tid1', 'tid2','title1_zh','title2_zh'],axis=1)
    # train_df=train_df.replace(np.nan, '', regex=True)
    # # text_clf = Pipeline([('vect', TfidfVectorizer()),
    #                   ('clf', MultinomialNB()) ])
    #text_clf.fit(train_df['title1_en']+train_df['title2_en'],train_df['label'])
    #validateData(validation_data,clsf)
    # validate_df=pd.read_csv("/home/tarunlolla/SMM/Project2/Project2_Files/validataion.csv",sep=",",error_bad_lines=False)
    # validate_df=validate_df.drop([ 'tid1', 'tid2','title1_zh','title2_zh'],axis=1)
    # validate_df=validate_df.replace(np.nan, '', regex=True)
    # predicted = text_clf.predict(validate_df['title1_en']+validate_df['title2_en'])
    # print(predicted)
    # validate_df['predicted']=predicted
    #stestData(test_data,clsf)
    # write_file=open('submission.csv','w')
    # write_file.write('id,label,prediction\n')
    # for i in range(validate_df.shape[0]):
    #     write_file.write(validate_df['id'][i]+','+validate_df['label'][i]+validate_df['predicted'][i]+'\n')
    # write_file.close()
    # print('Accuracy achieved is ' + str(np.mean(predicted == validate_df['label'])))
    # print(metrics.classification_report(validate_df['label'], predicted)),
    # metrics.confusion_matrix(validate_df['label'], predicted)

if __name__=='__main__':
    main()

