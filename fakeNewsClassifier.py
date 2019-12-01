import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn import metrics

def getDatasetPath():
    file=open('dataset.txt','r')
    lines=file.readlines()
    data_dict=dict()
    for line in lines:
        val=line.rstrip().split("=")
        data_dict[val[0]]=val[1]
    return data_dict['train_data'],data_dict['validation_data'],data_dict['test_data']

def trainData(dataset_path):
    train_df=pd.read_csv(dataset_path,sep=",",error_bad_lines=False,warn_bad_lines=False)
    train_df=train_df.drop(['id', 'tid1', 'tid2','title1_zh','title2_zh'],axis=1)
    labels=['agreed','disagreed','unrelated']
    labels_df=list(train_df['label'].unique())
    junk_labels=list(set(labels_df)-set(labels))
    for lbl in junk_labels:
        if lbl in list(train_df['label']):
            train_df=train_df.drop(train_df[train_df['label'] == lbl].index)
    train_df=train_df.replace(np.nan, '', regex=True)
    text_clf = Pipeline([('vect', TfidfVectorizer()),
                      ('clf', MultinomialNB()) ])
    text_clf.fit(train_df['title1_en']+train_df['title2_en'],train_df['label'])
    print("Classifier trained using data from : "+dataset_path)
    return text_clf

def validateData(dataset_path,clsf):
    validate_df=pd.read_csv(dataset_path,sep=",",error_bad_lines=False,warn_bad_lines=False)
    validate_df=validate_df.drop(['tid1', 'tid2','title1_zh','title2_zh'],axis=1)
    validate_df=validate_df.dropna(subset=['label'])
    labels=['agreed','disagreed','unrelated']
    labels_df=list(validate_df['label'].unique())
    junk_labels=list(set(labels_df)-set(labels))
    for lbl in junk_labels:
        if lbl in list(validate_df['label']):
            validate_df=validate_df.drop(validate_df[validate_df['label'] == lbl].index)
    predicted = clsf.predict(validate_df['title1_en']+validate_df['title2_en'])
    validate_df['predicted']=predicted
    print("Validation completed on data at : "+dataset_path)
    print('Accuracy achieved is ' + str(round(np.mean(predicted == validate_df['label'])*100,2)) + '%')
    print("Classification Report :")
    print(metrics.classification_report(validate_df['label'], predicted))


def testData(dataset_path,clsf):
    test_df=pd.read_csv(dataset_path,sep=",",error_bad_lines=False,warn_bad_lines=False)
    test_df=test_df.drop(['tid1', 'tid2','title1_zh','title2_zh'],axis=1)
    test_df=test_df.replace(np.nan, '', regex=True)
    predicted = clsf.predict(test_df['title1_en']+test_df['title2_en'])
    test_df['label']=predicted
    write_file=open('submission.csv','w')
    write_file.write('id,label\n')
    for i in range(test_df.shape[0]):
        write_file.write(test_df['id'][i]+','+test_df['label'][i]+'\n')
    write_file.close()
    print("Task Complete. Results available at : submission.csv")

def main():
    print("Training the model... ")
    train_data,validation_data,test_data=getDatasetPath()
    clsf=trainData(train_data)
    print("Validating the model... ")
    validateData(validation_data,clsf)
    print("Predicting... ")
    testData(test_data,clsf)

if __name__=='__main__':
    main()

