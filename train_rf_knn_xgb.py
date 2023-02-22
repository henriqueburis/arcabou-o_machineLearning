import os
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns
import glob
import cv2


#label_list = ['nuvem','Solo_exposto','Cultivo']
#label_list = ['nuvem','noNuvem']
label_list = ['Solo_exposto','Cultivo']

#label_list = ['nuvem','cpi','cp','se','crop','cultivo']

seed ="{:%d-%m-%Y_%H-%M-%S}".format(datetime.now())
n_classe = len(label_list)
save_dir = "results"

#train_images = np.load('../Data/talhao_com_mask/train_24_01_23_mask_talhao_images.npy') #(9000, 128, 128, 3)
#train_labels_ = np.load('../Data/talhao_com_mask/train_24_01_23_mask_talhao_labels_c.npy') #(9000,1)

#train_images = np.load('../Data/amostras_pre_processadas/imagem_com_mascara/sem_normalizacao/nuvem_nonuvem_train_sem_norm_com_mask_talhao_images.npy') 
#train_labels_ = np.load('../Data/amostras_pre_processadas/imagem_com_mascara/sem_normalizacao/nuvem_nonuvem_train_sem_norm_com_mask_talhao_labels.npy')

train_images = np.load('../Data/amostras_pre_processadas/imagem_sem_mascara/nuvem_nonuvem_train_sem_norm_sem_mask_talhao_images.npy')
train_labels_ = np.load('../Data/amostras_pre_processadas/imagem_sem_mascara/nuvem_nonuvem_train_sem_norm_sem_mask_talhao_labels.npy')

#train_images = np.load('../Data/amostras_pre_processadas/imagem_sem_mascara/se_cultivo_train_sem_norm_sem_mask_talhao_images.npy') #===> #8000 amostras
#train_labels_ = np.load('../Data/amostras_pre_processadas/imagem_sem_mascara/se_cultivo_train_sem_norm_sem_mask_talhao_labels.npy')

inference = np.load('Dataloader/TALHAO_130_sem_norm_sem_mask_talhao_images.npy')


#test_images = np.load('../Data/talhao_com_mask/test_24_01_23_mask_talhao_images.npy') #(9000, 128, 128, 3)
#test_labels = np.load('../Data/talhao_com_mask/test_24_01_23_mask_talhao_labels_c.npy') #(9000,1)


print("before=$%;  ",train_images.shape) #(9000, 128, 128, 3)

t, w, h, c = train_images.shape
dataTrain = train_images.reshape(t,c,w,h)

print("After=$%;  ",dataTrain.shape) # (9000, 3, 128, 128)

treino_data, teste_data, treino_label, teste_label = train_test_split(dataTrain, train_labels_, test_size=0.20, random_state=42)

def norm_input(inputs):
    t, w, h, c = inputs.shape
    inputs = inputs.reshape(t,c*w*h)

    return inputs

def RF_(treino_data, treino_label,teste_data):
    # Random Forest
    RF_model = RandomForestClassifier(
                                  n_estimators = 2000, 
                                  random_state = 42, 
                                  bootstrap = False, 
                                  max_depth = 10, 
                                  max_features = 'auto',
                                  min_samples_leaf = 2,
                                  min_samples_split = 5,
                                  )


    RF_model.fit(treino_data, treino_label)

    prediction_RF = RF_model.predict(teste_data)
    return prediction_RF


def KNN_(treino_data, treino_label,teste_data):
    knn = KNeighborsClassifier(n_neighbors = 9)
    knn.fit(treino_data, treino_label)

    prediction_knn = knn.predict(teste_data)
    return prediction_knn


def SVM_(treino_data, treino_label,teste_data):
    clf = svm.SVC()
    clf.fit(treino_data, treino_label)
    prediction_svm = clf.predict(teste_data)
    return prediction_svm

def XGB_(treino_data, treino_label,teste_data):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    clf.fit(treino_data, treino_label)
    prediction = clf.predict(teste_data)
    return prediction

def metricBlock(true_label,pre):

    confusion = confusion_matrix(true_label, pre)
    print(confusion)

    print ("Accuracy = ", metrics.accuracy_score(true_label, pre))
    print ("Recall = ", metrics.recall_score(true_label, pre, average = None))
    print ("Precision = ", metrics.precision_score(true_label, pre, average = None))
    print ("F1 = ", metrics.f1_score(true_label, pre, average = None))




if __name__ == "__main__":

    
    treino_data = norm_input(treino_data) 

    #teste_data = norm_input(teste_data)

    inference = norm_input(inference)

    #t, w, h, c = test_images.shape
    #test_ = test_images.reshape(t,c*w*h)

    pred_rf = RF_(treino_data, treino_label,inference)
    #pred_knn = KNN_(treino_data, treino_label,teste_data)
    #pred_svm = SVM_(treino_data, treino_label,teste_data)
    #pred = XGB_(treino_data, treino_label,teste_data)

    #metricBlock(teste_label,pred_rf)
    print(pred_rf)



