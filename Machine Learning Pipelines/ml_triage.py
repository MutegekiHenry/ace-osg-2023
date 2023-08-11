#!/usr/bin/env python3

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the data set
dataset = pd.read_csv('./triage_data.csv')
print("I have read the dataset...")
#Check data set shape
dataset.shape

#Sample columns
import random
random.choices(dataset.columns, k=20)

# Sample dataset
dataset.head()

# Drop index column
dataset.drop('Unnamed: 0', inplace=True, axis=1)

# Drop columns with more than 5% missing values
five_pct_missing = [column for column in dataset.columns if (sum(dataset[column].isna() * 1)/dataset.shape[0]) * 100 > 5.0]
dataset.drop(five_pct_missing, inplace=True, axis=1)

dataset.shape

# Take care of missing values
if dataset.isna().any().any():
  dataset.dropna(inplace=True)

dataset.isna().any().any()

dataset.shape


# Decompose dataset for better analysis
# Features that are related are grouped together

demographics_features = list(set(dataset.columns).intersection(set(["age",  "gender",  "ethnicity",  "race",  "lang",  "religion",  "maritalstatus",  "employstatus",  "insurance_status"])))
triage_features = list(set(dataset.columns).intersection(set(["dep_name",  "arrivalmode",  "arrivalmonth",  "arrivalday",  "arrivalhour_bin",  "triage_vital_hr",  "triage_vital_sbp",  
                                                              "triage_vital_dbp",  "triage_vital_rr",  "triage_vital_o2",  "triage_vital_o2_device",  "triage_vital_temp"])))
hospital_usage_features = list(set(dataset.columns).intersection(set(["disposition", "previousdispo",  "n_edvisits",  "n_admissions",  "n_surgeries"])))
imaging_features = list(set(dataset.columns).intersection(set(["cxr_count",  "echo_count",  "ekg_count",  "headct_count",  "mri_count",  "otherct_count",  "otherimg_count",  "otherus_count",  
                                                               "otherxr_count"])))
pmh_features = list(set(dataset.columns).intersection(set(["2ndarymalig",  "abdomhernia",  "abdomnlpain",  "abortcompl",  "acqfootdef",  "acrenlfail",  "acutecvd",  "acutemi",  "acutphanm",  
                                                           "adjustmentdisorders",  "adltrespfl",  "alcoholrelateddisorders",  "allergy",  "amniosdx",  "analrectal",  "anemia",  "aneurysm",  
                                                           "anxietydisorders",  "appendicitis",  "artembolism",  "asppneumon",  "asthma",  "attentiondeficitconductdisruptivebeha",  "backproblem",  
                                                           "biliarydx",  "birthasphyx",  "birthtrauma",  "bladdercncr",  "blindness",  "bnignutneo",  "bonectcncr",  "bph",  "brainnscan",  "breastcancr",  
                                                           "breastdx",  "brnchlngca",  "bronchitis",  "burns",  "cardiaarrst",  "cardiacanom",  "carditis",  "cataract",  "cervixcancr",  "chestpain",  
                                                           "chfnonhp",  "chrkidneydisease",  "coaghemrdx",  "coloncancer",  "comabrndmg",  "complicdevi",  "complicproc",  "conduction",  "contraceptiv",  
                                                           "copd",  "coronathero",  "crushinjury",  "cysticfibro",  "deliriumdementiaamnesticothercognitiv",  "developmentaldisorders",  "diabmelnoc",  
                                                           "diabmelwcm",  "disordersusuallydiagnosedininfancych",  "diverticulos",  "dizziness",  "dminpreg",  "dysrhythmia",  "earlylabor",  
                                                           "ecodesadverseeffectsofmedicalcare",  "ecodesadverseeffectsofmedicaldrugs",  "ecodescutpierce",  "ecodesdrowningsubmersion",  "ecodesfall",  
                                                           "ecodesfirearm",  "ecodesfireburn",  "ecodesmachinery",  "ecodesmotorvehicletrafficmvt",  "ecodesnaturalenvironment",  
                                                           "ecodesotherspecifiedandclassifiable",  "ecodesotherspecifiednec",  "ecodespedalcyclistnotmvt",  "ecodesplaceofoccurrence",  "ecodespoisoning",  
                                                           "ecodesstruckbyagainst",  "ecodessuffocation",  "ecodestransportnotmvt",  "ecodesunspecified",  "ectopicpreg",  "encephalitis",  "endometrios",  
                                                           "epilepsycnv",  "esophcancer",  "esophgealdx",  "exameval",  "eyeinfectn",  "fatigue",  "femgenitca",  "feminfertil",  "fetaldistrs",  "fluidelcdx",  
                                                           "fuo",  "fxarm",  "fxhip",  "fxleg",  "fxskullfac",  "gangrene",  "gasduoulcer",  "gastritis",  "gastroent",  "giconganom",  "gihemorrhag",  "giperitcan",  
                                                           "glaucoma",  "goutotcrys",  "guconganom",  "hdnckcancr",  "headachemig",  "hemmorhoids",  "hemorrpreg",  "hepatitis",  "hivinfectn",  "hodgkinsds",  
                                                           "hrtvalvedx",  "htn",  "htncomplicn",  "htninpreg",  "hyperlipidem",  "immunitydx",  "immunizscrn",  "impulsecontroldisordersnec",  "inducabortn",  
                                                           "infectarth",  "influenza",  "infmalegen",  "intestinfct",  "intobstruct",  "intracrninj",  "jointinjury",  "kidnyrnlca",  "lateeffcvd",  "leukemias",  
                                                           "liveborn",  "liveribdca",  "longpregncy",  "lowbirthwt",  "lungexternl",  "lymphenlarg",  "maintchemr",  "malgenitca",  "maligneopls",  "malposition",  
                                                           "meningitis",  "menopausldx",  "menstrualdx",  "miscellaneousmentalhealthdisorders",  "mooddisorders",  "mouthdx",  "ms",  "multmyeloma",  "mycoses",  
                                                           "nauseavomit",  "neoplsmunsp",  "nephritis",  "nervcongan",  "nonepithca",  "nonhodglym",  "nutritdefic",  "obrelatedperintrauma",  "opnwndextr",  
                                                           "opnwndhead",  "osteoarthros",  "osteoporosis",  "otacqdefor",  "otaftercare",  "otbnignneo",  "otbonedx",  "otcirculdx",  "otcomplbir",  "otconganom",  
                                                           "otconntiss",  "otdxbladdr",  "otdxkidney",  "otdxstomch",  "otendodsor",  "otfemalgen",  "othbactinf",  "othcnsinfx",  "othematldx",  "othercvd",  
                                                           "othereardx",  "otheredcns",  "othereyedx",  "othergidx",  "othergudx",  "otherinjury",  "otherpregnancyanddeliveryincludingnormal",  "otherscreen",  
                                                           "othfracture",  "othheartdx",  "othinfectns",  "othliverdx",  "othlowresp",  "othmalegen",  "othnervdx",  "othskindx",  "othveindx",  "otinflskin",  
                                                           "otitismedia",  "otjointdx",  "otnutritdx",  "otperintdx",  "otpregcomp",  "otprimryca",  "otrespirca",  "otupprresp",  "otuprspin",  "ovariancyst",  
                                                           "ovarycancer",  "pancreascan",  "pancreasdx",  "paralysis",  "parkinsons",  "pathologfx",  "pelvicobstr",  "perintjaund",  "peripathero",  "peritonitis",  
                                                           "personalitydisorders",  "phlebitis",  "pid",  "pleurisy",  "pneumonia",  "poisnnonmed",  "poisnotmed",  "poisonpsych",  "precereoccl",  "prevcsectn",  
                                                           "prolapse",  "prostatecan",  "pulmhartdx",  "rctmanusca",  "rehab",  "respdistres",  "retinaldx",  "rheumarth",  "schizophreniaandotherpsychoticdisorde",  
                                                           "screeningandhistoryofmentalhealthan",  "septicemia",  "septicemiaexceptinlabor",  "sexualinfxs",  "shock",  "sicklecell",  "skininfectn",  "skinmelanom",  
                                                           "sle",  "socialadmin",  "spincorinj",  "spontabortn",  "sprain",  "stomchcancr",  "substancerelateddisorders",  "suicideandintentionalselfinflictedin",  
                                                           "superficinj",  "syncope",  "teethdx",  "testiscancr",  "thyroidcncr",  "thyroiddsor",  "tia",  "tonsillitis",  "tuberculosis",  "ulceratcol",  "ulcerskin",  
                                                           "umbilcord",  "unclassified",  "urinstone",  "urinyorgca",  "uteruscancr",  "uti",  "varicosevn",  "viralinfect",  "whtblooddx"])))
historical_labs_features = list(set(dataset.columns).intersection(set(["bloodua_last",  "glucoseua_last",  "ketonesua_last",  "leukocytesua_last",  "nitriteua_last",  "pregtestur_last",  "proteinua_last",  
                                                              "bloodculture,routine_last",  "urineculture,routine_last",  "bloodua_npos",  "glucoseua_npos",  "ketonesua_npos",  "leukocytesua_npos",  
                                                              "nitriteua_npos",  "pregtestur_npos",  "proteinua_npos",  "bloodculture,routine_npos",  "urineculture,routine_npos",  "bloodua_count",  
                                                              "glucoseua_count",  "ketonesua_count",  "leukocytesua_count",  "nitriteua_count",  "pregtestur_count",  "proteinua_count",  
                                                              "bloodculture,routine_count",  "urineculture,routine_count"])))


demographics = dataset[demographics_features]
triage_variables = dataset[triage_features]
hospital_usage = dataset[hospital_usage_features]
chief_complaint = dataset[[column for column in dataset.columns if column.startswith('cc')]]
past_medical_history = dataset[pmh_features]
outpatient_medications = dataset[[column for column in dataset.columns if column.startswith('meds')]]
imaging = dataset[imaging_features]
response_variable = dataset['esi']
historical_labs = dataset[historical_labs_features]

chief_complaint['n_cc'] = chief_complaint.sum(axis=1)
past_medical_history['n_pmh'] = past_medical_history.sum(axis=1)
outpatient_medications['n_meds'] = outpatient_medications.sum(axis=1)
imaging['total_count'] = imaging.sum(axis=1)

print("I have reached data processing...")
## Data Preprocessing

# One-hot encoding categorical features
independent_variables = pd.concat([demographics, triage_variables, hospital_usage, past_medical_history, chief_complaint, imaging, outpatient_medications, historical_labs], axis=1)
dataset_enc = pd.get_dummies(independent_variables, columns=[column for column in independent_variables if independent_variables[column].dtype==object])

# Memory optimization
import gc
del([demographics, triage_variables, hospital_usage, past_medical_history, chief_complaint, imaging, outpatient_medications, historical_labs])
del(independent_variables)
del(dataset)
gc.collect()

# Define independent (predictors) and dependent (target) variables
X = dataset_enc.values
y = response_variable.values.astype(np.int64)

print(X.shape, y.shape)

# Split train and test set
import sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Feature selection
from sklearn.feature_selection import SelectKBest, VarianceThreshold
thresholder = VarianceThreshold(threshold=0.5)
X_train = thresholder.fit_transform(X_train)
X_test = thresholder.transform(X_test)

print(X_train.shape, X_test.shape)

threshold_features = dataset_enc.columns[thresholder.get_support()]
feature_selector = SelectKBest(k=int(X_train.shape[1] * 0.5))
X_train = feature_selector.fit_transform(X_train, y_train)
X_test = feature_selector.transform(X_test)

selected_features = threshold_features[feature_selector.get_support()]

# Check class distribution before resampling
y_train_df = pd.DataFrame(data=y_train.reshape(-1,1), columns=['classes'])
y_train_dist = y_train_df.classes.value_counts()

# Plot class distribution before resampling

plt.figure(figsize=(8,8))
sns.barplot(y=y_train_dist.values, x=y_train_dist.index, color='g')
plt.title('Class distribution in response variable before resampling')
plt.xlabel('Classes')
plt.ylabel('Number of samples')
plt.show()

# Resampling to cater for class imbalance
from imblearn.combine import SMOTETomek
resampler = SMOTETomek(sampling_strategy="auto", random_state=42)
X_train, y_train = resampler.fit_resample(X_train, y_train)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Check class distribution after resampling
y_train_df = pd.DataFrame(data=y_train.reshape(-1,1), columns=['classes'])
y_train_dist = y_train_df.classes.value_counts()

# Plot class distribution after resampling

plt.figure(figsize=(8,8))
sns.barplot(y=y_train_dist.values, x=y_train_dist.index, color='g')
plt.title('Class distribution in response variable after resampling')
plt.xlabel('Classes')
plt.ylabel('Number of samples')
plt.savefig('./resampling_distribution.png', format='png')


# Supervised Methods

# Import Libraries
from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
# from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from imblearn.metrics import classification_report_imbalanced

def cm(y_true, y_pred, labels=None):
  '''
  Generates and displays a confusion matrix
  '''
  matrix = confusion_matrix(y_true, y_pred)

  cm_display = ConfusionMatrixDisplay(confusion_matrix = matrix, display_labels=labels)

  plt.figure(figsize=(8,8))
  plt.title(f"Confusion Matrix")
  cm_display.plot(ax=plt.subplot())
  plt.show()
  plt.savefig('./confusion_matrix.png', format='png')

  # Instantiate classifier
dt_classifier = DecisionTreeClassifier(max_depth=10, random_state=42)

# Train classifer
dt_classifier.fit(X_train, y_train)

# Run inference on hold-out test set
y_pred = dt_classifier.predict(X_test)

# Evaluate results
print(classification_report(y_test, y_pred))

print(classification_report_imbalanced(y_test, y_pred))

# Evaluate results
plt.figure(figsize=(15,10))
plt.plot([0,1],[0,1],'k:',label='Random')

y_pred_proba = dt_classifier.predict_proba(X_test)
y_test_dummies = pd.get_dummies(y_test).values

for label in range(5):
  fpr,tpr,thresholds = roc_curve(y_test_dummies[:,label], y_pred_proba[:,label])
  roc_auc = roc_auc_score(y_test_dummies[:,label], y_pred_proba[:,label])

  # Plot the results
  plt.plot(fpr, tpr, label='ROC curve for class {0} (area = {1:0.2f})'.format(label, roc_auc))
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.grid()
  plt.legend()
  plt.title(f'Class-wise metrics')

  # Plot confusion matrix
sns.set_theme(style="white", palette=None)
cm(y_test, y_pred)






