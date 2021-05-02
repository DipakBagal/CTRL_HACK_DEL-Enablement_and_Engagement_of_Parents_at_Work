### Custom definitions and classes if any ###

def load_model():
    # loading model
    from keras.models import Sequential, model_from_json
    import os
    import math 
    import pickle
    
    Package_path = os.path.dirname(os.path.abspath(__file__))
    
    model = model_from_json(open('/content/drive/MyDrive/CTRL_HACK_DEL/IDEA_Submissions/Enablement_and_Engagement_of_Parents_at_Work/model_architecture.json').read())
    model.load_weights('/content/drive/MyDrive/CTRL_HACK_DEL/IDEA_Submissions/Enablement_and_Engagement_of_Parents_at_Work/model_weights.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def Predict_Support(filename='/content/drive/MyDrive/CTRL_HACK_DEL/IDEA_Submissions/Enablement_and_Engagement_of_Parents_at_Work/Test.csv'):
  
  import numpy as np
  import pandas as pd
  from keras.models import Sequential, model_from_json
  from keras.layers import Dense
  from keras.utils import np_utils
  import os
  from sklearn.preprocessing import LabelEncoder  
  import pickle
  
  Package_path = os.path.dirname(os.path.abspath(__file__))
  filepath = os.path.dirname(os.path.abspath(__file__))

  encoder = LabelEncoder()
  encoder.classes_ = np.load('/content/drive/MyDrive/CTRL_HACK_DEL/IDEA_Submissions/Enablement_and_Engagement_of_Parents_at_Work/Y_classes.npy', allow_pickle=True)
  
  '''
  en_WFH={}
  en_CompanyType={}
  en_Gender={}
  en_Review={}
  en_Partner={}
  en_Parent={}
  en_Insurance={}
  en_Flexible={}
  en_partnerWorking={}
  en_Health={}
  '''
''' 
  a_file = open("/content/drive/MyDrive/CTRL_HACK_DEL/IDEA_Submissions/Enablement_and_Engagement_of_Parents_at_Work/en_WFH.pkl", "rb")
  en_WFH = pickle.load(a_file)
  a_file = open("/content/drive/MyDrive/CTRL_HACK_DEL/IDEA_Submissions/Enablement_and_Engagement_of_Parents_at_Work/en_CompanyType.pkl", "rb")
  en_CompanyType = pickle.load(a_file)
  a_file = open("/content/drive/MyDrive/CTRL_HACK_DEL/IDEA_Submissions/Enablement_and_Engagement_of_Parents_at_Work/en_Gender.pkl", "rb")
  en_Gender = pickle.load(a_file)
  a_file = open("/content/drive/MyDrive/CTRL_HACK_DEL/IDEA_Submissions/Enablement_and_Engagement_of_Parents_at_Work/en_Review.pkl", "rb")
  en_Review = pickle.load(a_file)
  a_file = open("/content/drive/MyDrive/CTRL_HACK_DEL/IDEA_Submissions/Enablement_and_Engagement_of_Parents_at_Work/en_Partner.pkl", "rb")
  en_Partner = pickle.load(a_file)
  a_file = open("/content/drive/MyDrive/CTRL_HACK_DEL/IDEA_Submissions/Enablement_and_Engagement_of_Parents_at_Work/en_Parent.pkl", "rb")
  en_Parent = pickle.load(a_file)
  a_file = open("/content/drive/MyDrive/CTRL_HACK_DEL/IDEA_Submissions/Enablement_and_Engagement_of_Parents_at_Work/en_Insurance.pkl", "rb")
  en_Insurance = pickle.load(a_file)
  a_file = open("/content/drive/MyDrive/CTRL_HACK_DEL/IDEA_Submissions/Enablement_and_Engagement_of_Parents_at_Work/en_Flexible.pkl", "rb")
  en_Flexible = pickle.load(a_file)
  a_file = open("/content/drive/MyDrive/CTRL_HACK_DEL/IDEA_Submissions/Enablement_and_Engagement_of_Parents_at_Work/en_partnerWorking.pkl", "rb")
  en_partnerWorking = pickle.load(a_file)
  a_file = open("/content/drive/MyDrive/CTRL_HACK_DEL/IDEA_Submissions/Enablement_and_Engagement_of_Parents_at_Work/en_Health.pkl", "rb")
  en_Health = pickle.load(a_file)
  
  Orig_Frame = pd.read_csv(filename, index_col=None, header=0)
  test_frame = pd.read_csv(filename, index_col=None, header=0)

  import nltk
  nltk.download('vader_lexicon')
  from nltk.sentiment.vader import SentimentIntensityAnalyzer
  sid = SentimentIntensityAnalyzer()
  #frame.head()  
  result = []
  for value in test_frame["Review Of Company"]:
    sentiment_dict = sid.polarity_scores(value)
    if sentiment_dict['compound'] >= 0.05 :
      result.append("Positive")
  
    elif sentiment_dict['compound'] <= - 0.05 :
      result.append("Negative")
  
    else :
      result.append("Neutral")

  test_frame["Review Of Company"] = result   
  test_frame.head()
  test_frame.drop(['Children Age'], axis = 1,inplace = True)

  # Convert the column values to lower
  test_frame['WFH Setup Available'] = test_frame['WFH Setup Available'].str.lower()
  test_frame['Company Type'] = test_frame['Company Type'].str.lower()          
  test_frame['Gender']  = test_frame['Gender'].str.lower()               
  test_frame['Review Of Company'] = test_frame['Review Of Company'].str.lower()     
  test_frame['Partner']  = test_frame['Partner'].str.lower()               
  test_frame['Is Parent']  = test_frame['Is Parent'].str.lower()             
  test_frame['Used Insurance']  = test_frame['Used Insurance'].str.lower()        
  test_frame['Flexible Working Hours'] = test_frame['Flexible Working Hours'].str.lower() 
  test_frame['Partner Working'] = test_frame['Partner Working'].str.lower()        
  test_frame['Health Conditions']  = test_frame['Health Conditions'].str.lower()

  ## Encode columns in final dataframe

  test_frame['WFH Setup Available'] = test_frame['WFH Setup Available'].map(en_WFH)
  test_frame['Company Type'] = test_frame['Company Type'].map(en_CompanyType)          
  test_frame['Gender']  = test_frame['Gender'].map(en_Gender)               
  test_frame['Review Of Company'] = test_frame['Review Of Company'].map(en_Review)     
  test_frame['Partner']  = test_frame['Partner'].map(en_Partner)               
  test_frame['Is Parent']  = test_frame['Is Parent'].map(en_Parent)             
  test_frame['Used Insurance']  = test_frame['Used Insurance'].map(en_Insurance)        
  test_frame['Flexible Working Hours'] = test_frame['Flexible Working Hours'].map(en_Flexible) 
  test_frame['Partner Working'] = test_frame['Partner Working'].map(en_partnerWorking)        
  test_frame['Health Conditions']  = test_frame['Health Conditions'].map(en_Health)   

  test_frame = test_frame.drop(['Employee ID','Learning New Thing','Hours Worked Last Week','Completed Task Last Week'], axis = 1)
  test_frame['Assigned Work Hours']= test_frame['Assigned Work Hours']/test_frame['Assigned Work Hours'].max()
  test_frame['Assigned Task Last Week']= test_frame['Assigned Task Last Week']/test_frame['Assigned Task Last Week'].max()
  test_frame['Leaves Available']= test_frame['Leaves Available']/test_frame['Leaves Available'].max()
  print(test_frame.head())
  
  test_X=test_frame.astype(float)
  print('X shape : ', test_X.shape)
  model = load_model()
  # predictions
  test_y = model.predict_classes(test_X, verbose=0)
  print(test_y)
  encoder = LabelEncoder()
  encoder.classes_ = np.load('/content/drive/MyDrive/CTRL_HACK_DEL/IDEA_Submissions/Enablement_and_Engagement_of_Parents_at_Work/Y_classes.npy', allow_pickle=True)
  Y = encoder.inverse_transform(test_y)
  
  Orig_Frame['Support']=Y
  Orig_Frame.to_csv('/content/drive/MyDrive/CTRL_HACK_DEL/IDEA_Submissions/Enablement_and_Engagement_of_Parents_at_Work/Test_submit.csv')
  print(Y[0:9])
'''

