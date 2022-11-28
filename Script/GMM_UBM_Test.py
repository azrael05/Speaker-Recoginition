import librosa
from glob import glob
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn import mixture
from sklearn.metrics import accuracy_score

def compute_eer(label, pred, positive_label=1):

    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred)
    fnr = 1 - tpr

    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    eer = (eer_1 + eer_2) / 2
    return eer

base_path='/home/pi/speaker_recognition/SR'
test_path=  base_path + '/' + 'Test_Folder'

cl_sz=64

test_files_1 = glob(test_path+"/*/*_AH01OENC*")
test_files_2 = glob(test_path+"/*/*_AH01MENC*")
test_files = test_files_1 + test_files_2
test_files.sort()

true = []
for i in range(len(test_files)):
  true.append(test_files[i].split('/')[7])
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
true = le.fit_transform(true)

Score = []
speaker=[]
pred = []
prob=[]
for i in range(len(test_files)):
 audio_data, fs = librosa.load(test_files[i],sr=8000)

 audio_data=audio_data-np.mean(audio_data)
 audio_data=audio_data/(1.01*(max(abs(audio_data))))

 MFCC=librosa.feature.mfcc(y=audio_data,sr=fs, n_mfcc=14, lifter=1,dct_type=3,hop_length=int(0.01*fs), win_length=int(0.02*fs),window='hann', n_mels=24)
 MFCC_d  = librosa.feature.delta(MFCC, order=1)
 MFCC_dd = librosa.feature.delta(MFCC, order=2)

 MFCC=MFCC.T
 MFCC_d=MFCC_d.T
 MFCC_dd=MFCC_dd.T

 MFCC=MFCC[:,1:14]
 MFCC_d=MFCC_d[:,1:14]
 MFCC_dd=MFCC_dd[:,1:14]

 X = np.concatenate((MFCC, MFCC_d), axis=1)
 X = np.concatenate((X, MFCC_dd), axis=1)

 for j in range(len(SPK)):
  gmm_name = "Speaker_"+str(j)
  loaded_gmm = mixture.GaussianMixture(n_components = cl_sz, covariance_type='diag')
  means = np.load("/content/drive/MyDrive/Speaker Recognition Data/GMM UBM Models/"+gmm_name + '_means.npy')
  covar = np.load("/content/drive/MyDrive/Speaker Recognition Data/GMM UBM Models/"+gmm_name + '_covariances.npy')
  loaded_gmm.weights_ = np.load("/content/drive/MyDrive/Speaker Recognition Data/GMM UBM Models/"+gmm_name + '_weights.npy')
  loaded_gmm.precisions_cholesky_ = np.load("/content/drive/MyDrive/Speaker Recognition Data/GMM UBM Models/"+gmm_name + '_precisions_cholesky.npy')
  loaded_gmm.means_ = means
  loaded_gmm.covariances_ = covar
  log_likelihood_gmm = loaded_gmm.score(X)

  loaded_ubm = mixture.GaussianMixture(n_components = cl_sz, covariance_type='diag')
  means_ubm = np.load("/content/drive/MyDrive/Speaker Recognition Data/GMM UBM Models/"+'UBM_means.npy')
  covar_ubm = np.load("/content/drive/MyDrive/Speaker Recognition Data/GMM UBM Models/"+'UBM_covariances.npy')
  loaded_ubm.weights_ = np.load("/content/drive/MyDrive/Speaker Recognition Data/GMM UBM Models/"+'UBM_weights.npy')
  loaded_ubm.precisions_cholesky_ = np.load("/content/drive/MyDrive/Speaker Recognition Data/GMM UBM Models/"+'UBM_precisions_cholesky.npy')
  loaded_ubm.means_ = means
  loaded_ubm.covariances_ = covar
  log_likelihood_ubm = loaded_ubm.score(X)

  Score.append(log_likelihood_gmm-log_likelihood_ubm)
 pred_label = np.argmax(Score)
 if(pred_label == true[i]):
   pred.append(1)
 else:
   pred.append(0)  
 prob.append(abs(max(Score)))  
 speaker.append(pred_label)
 Score=[]
 print("Testing of Utterence "+str(i)+" completed")

eer = compute_eer(pred, prob)
print('The equal error rate is {:.3f}'.format(eer))

print("The Accuracy is ", accuracy_score(true, speaker))