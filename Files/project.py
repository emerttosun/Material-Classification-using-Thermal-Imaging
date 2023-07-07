
"""

@authors: Erkani Mert Tosun & Eyüp Enes Aytaç

"""



import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import signal
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torchvision
from torch import nn
from torch import optim
import random
from sklearn.preprocessing import OneHotEncoder
import copy
import time

#%%




def videoToTimeSeries(inpPath, inpRect = [425, 160, 470, 200]): # topleftx toplefty, botrightx, botrighty
    #print(inpPath)
    capture = cv2.VideoCapture(inpPath)
    result = []
    
    while (capture.isOpened()):    
        ret , frame = capture.read()            # ret = true or false, frame has a image info
        if ret == False:
            break
        
        newframe  = cv2.resize(frame, (640,480))   
        newframe = cv2.GaussianBlur(newframe, (5,5), 10)  # take the low pass filter and blur the image with gaussian
        #newframe = cv2.rectangle(newframe, (inpRect[0], inpRect[0]),(inpRect[2], inpRect[3]), (0,255,0), 1) #draw the rectangle on the image
        roi = newframe[inpRect[1]:inpRect[3], inpRect[0]:inpRect[2]]
        result.append(np.average(roi))

    return result
    

 #%%   
# Extract feature from timeseries with exponentially
def extractFeature(timeSeries) :
    
    Feature = []
    lenofInt = 300
    maxPoint = np.argmax(timeSeries)
    for idx in range(lenofInt) : 
        tempFeature = []
        startPoint = maxPoint - int(lenofInt/2) + idx
        for k in range(8) : 
            tempFeature.append(timeSeries[startPoint] 
                               - timeSeries[startPoint - 2**k ]) 
        Feature.append(tempFeature)
    Feature = np.array(Feature,dtype = object)
    return Feature




#%%

#Generate a train and test set for general case, USE WHEN DeepLearning
def GenerateSets(df):
    X = []
    y = []
    
    for i in range(len(df)) : 
        Label = df.iloc[i][1]
        X.append(extractFeature(df.iloc[i][6]))
        if Label == "Metal" : 
            Tempy = np.ones(300)*0
            y.append(Tempy)
            
        elif Label == "Plastic" : 
            Tempy = np.ones(300)*1
            y.append(Tempy)
        elif Label == "Carton" :
            
            Tempy = np.ones(300)*2
            y.append(Tempy)
    
    
    X = np.concatenate(X)
    y = np.concatenate(y)
    

    return X,y
        

#%%
#Prediction Model for SVM
def predictionSVM(test_array,clf,label) : 
    
    pred = 0
    
    #clf.fit(X_train, y_train)
    
    prediction = clf.predict(test_array)
    
    
    pred1 = sum(prediction == 0)/len(prediction)
    pred2 = sum(prediction == 1)/len(prediction)
    pred3 = sum(prediction == 2)/len(prediction)

    maxpred = [pred1,pred2,pred3]

    max_value = max(maxpred)
    label = np.argmax(maxpred)
        
    if label == 0 :
        label_Name = "Metal"
        
    elif label == 1 :
        label_Name = "Plastic"
        
    elif label == 2 :
        label_Name = "Carton"
    
    #print("Prediction : " + label_Name + " % " + str(max_value * 100) )
    print(label_Name)

def predictionDPL(test_array,model,label) : 
    test_array = test_array.astype(np.float32)
    maxpred = []
    torch_Sample_X = torch.from_numpy(test_array)
    
    with torch.no_grad() : 
        prediction_sample = model(torch_Sample_X)
    
    prediction_sample = prediction_sample.argmax(dim=1)
    
    prediction_sample = prediction_sample.detach().numpy()
    
    pred1 = sum(prediction_sample == 0)/len(prediction_sample)
    pred2 = sum(prediction_sample == 1)/len(prediction_sample)
    pred3 = sum(prediction_sample == 2)/len(prediction_sample)

    maxpred = [pred1,pred2,pred3]

    max_value = max(maxpred)
    label = np.argmax(maxpred)
        
    if label == 0 :
        label_Name = "Metal"
        
    elif label == 1 :
        label_Name = "Plastic"
        
    elif label == 2 :
        label_Name = "Carton"
    
    print(label_Name)
    
    
#%%   
# Model for DeepLearning
def Model_seq(inp_length) : 
    seq_model = nn.Sequential(
        nn.Linear(inp_length, 32),
        nn.Sigmoid(),
        nn.Linear(32, 64),
        nn.Sigmoid(),
        nn.Linear(64, 256),
        nn.Sigmoid(),
        nn.Linear(256, 512),
        nn.Sigmoid(),
        nn.Linear(512, 3))
        
    return seq_model
# Another Model 
def Model_new(inp_length):
    model = nn.Sequential(
        nn.Linear(inp_length, 16),
        nn.ReLU(),
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 3)
    )
    return model

#%%

def DeepPrediction(model,X_train,y_train,X_test,y_test,
                   ) : 
    
    best_acc = - np.inf   # init to negative infinity
    best_weights = None
    
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []
    
    
    optimiser = optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 200
    batch_size = 16
    loss_fn = nn.CrossEntropyLoss()

    
    
    for epoch in range(n_epochs):
        epoch_loss = []
        epoch_acc = []
        
        #trainLossEpoch = 0
        model.train()
        for i in range(len(X_train)//batch_size) : 
            start = i * batch_size
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            output_train = model(X_batch) # forwards pass
            loss_train = loss_fn(output_train, y_batch) # calculate loss

            optimiser.zero_grad() # set gradients to zero
            loss_train.backward() # backwards pass
        
            optimiser.step() # update model parameters
            acc = (torch.argmax(output_train,1) == torch.argmax(y_batch,1)).float().mean()
            
            
            epoch_loss.append(float(loss_train))
            epoch_acc.append(float(acc))
            
            
        model.eval()
        y_pred = model(X_test)
        ce = loss_fn(y_pred, y_test)
        acc = (torch.argmax(y_pred,1) == torch.argmax(y_test,1)).float().mean()
        ce = float(ce)
        acc = float(acc)
        train_loss_hist.append(np.mean(epoch_loss))
        train_acc_hist.append(np.mean(epoch_acc))
        test_loss_hist.append(ce)
        test_acc_hist.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc*100:.4f}%")
        
    model.load_state_dict(best_weights)
    
    
    

    return train_loss_hist,test_loss_hist,train_acc_hist,test_acc_hist


def GetPredictionUnit(df_s,SampleNumber):
    
    windowSize = 5
    
    time=videoToTimeSeries(df_s.iloc[SampleNumber,0],
                           inpRect= [df_s.iloc[SampleNumber][2],df_s.iloc[SampleNumber][3],
                                     df_s.iloc[SampleNumber][4],df_s.iloc[SampleNumber][5]])
    win = signal.windows.hann(windowsize)
    filtered = signal.convolve(time, win, mode='same') / sum(win)
    
    
    win = signal.windows.hann(windowSize)
    
    
    features = signal.convolve(filtered, win, mode='same') / sum(win)
    X = extractFeature(features)
    
    return X


  #%%  
# main 
if __name__ == "__main__": 
    
    
    
    Time_Series = []
    
    df = pd.DataFrame(pd.read_excel(r"C:\Users\BHtosun\Desktop\dataf.xlsx" ,
                                    dtype = {'FileName': str, 'Label': str, 'topleftx ': int , 'toplefty ': int,
                                             'botrightx ': int, 'botrighty': int}))
    
    windowsize = 5
    a = 0
    for a in range(len(df)) :
        time=videoToTimeSeries(df.iloc[a,0],inpRect= [df.iloc[a][2],df.iloc[a][3],df.iloc[a][4],df.iloc[a][5]])
        win = signal.windows.hann(windowsize)
        filtered = signal.convolve(time, win, mode='same') / sum(win)
        Time_Series.append(filtered)
        
    df["TimeSeries"] = Time_Series
    
 #%%    For SVM case
    """
    #### FOR PLOTTING THE GRAPH 
    for a in range(len(df)) : 
        if df.iloc[a,1] == "Metal" : 
            colored = "green"
        elif df.iloc[a,1] == "Plastic" : 
            colored = "blue"
        else :
            colored = "red"
        plt.plot(df.iloc[a,6],color = colored,label = df.iloc[a,1])
        
            
    
    
    
    red_patch = mpatches.Patch(color='red', label='Carton')
    blue_patch = mpatches.Patch(color='blue', label='Plastic')
    green_patch = mpatches.Patch(color='green', label='Metal')
    plt.legend(handles=[red_patch,blue_patch,green_patch])
    """

 #%%

    X,y = GenerateSets(df)
    
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
    
#%%
    # SVM Process 
        
    df_s = pd.DataFrame(pd.read_excel( r"C:\Users\BHtosun\Desktop\dataPrediction.xlsx" ,
                                    dtype = {'FileName': str, 'Label': str, 'topleftx ': int , 'toplefty ': int,
                                             'botrightx ': int, 'botrighty': int}))
    
#%%
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train,y_train).score(X_test,y_test)
    
    
#%%# tree

    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train,y_train).score(X_test,y_test)

#%%
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    k = 5  # Adjust the value of k as per your needs
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train).score(X_test,y_test)
    

#%%

    
#%%
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=50000)
    clf.fit(X, y).score(X_test,y_test)
    
#%%

    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    
    classifier = OneVsRestClassifier(LogisticRegression())
    classifier.fit(X_train, y_train).score(X_test,y_test)

#%%

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train).score(X_test,y_test)  
    
#%%
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    
    gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    gb_classifier.fit(X_train, y_train).score(X_test,y_test)
    
#%%

    
    
    #%%
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    svm_classifier = SVC(kernel='rbf', C=1.0, random_state=42)
    svm_classifier.fit(X_train, y_train).score(X_test,y_test)    
#%%
    
    for a in range(len(df_s)) :
        Sample_X = GetPredictionUnit(df_s,a)
        if df_s.iloc[a][1] == "Metal" : 
                        
            predictionSVM(Sample_X, rf_classifier,0)
            
        elif df_s.iloc[a][1] == "Plastic" : 
            
            predictionSVM(Sample_X, rf_classifier,1)
        # Sample_X = Test Array ; clf = clf ; 0,1,2 choice 
        else : 
            predictionSVM(Sample_X, rf_classifier,2)





 #%%

    inp_length = 8
    model = Model_seq(inp_length)
    
    # one hat encoding
    
    y = np.array(y).reshape(-1,1)
    ohe = OneHotEncoder(sparse_output=False).fit(y)
    y = ohe.transform(y)
        
    X = X.astype(np.float32)
    X = torch.from_numpy(X).float()
    
    y = torch.from_numpy(y).float()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
    
    train_loss_hist,test_loss_hist,train_acc_hist,test_acc_hist = DeepPrediction(model,X_train,y_train,X_test,y_test)
    
        
    plt.plot(train_loss_hist, label="train")
    plt.plot(test_loss_hist, label="test")
    plt.xlabel("epochs")
    plt.ylabel("cross entropy")
    plt.legend()
    plt.show()
     
    plt.plot(train_acc_hist, label="train")
    plt.plot(test_acc_hist, label="test")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
    
    #%%
    for i in range(len(df_s)) : 
        
        Sample_X = GetPredictionUnit(df_s,i)
    
        
        if df_s.iloc[i][1] == "Metal" : 
                        
            predictionDPL(Sample_X, model,0)
            
        elif df_s.iloc[i][1] == "Plastic" : 
            
            predictionDPL(Sample_X, model,1)
        # Sample_X = Test Array ; clf = clf ; 0,1,2 choice 
        else : 
            predictionDPL(Sample_X, model,2)

