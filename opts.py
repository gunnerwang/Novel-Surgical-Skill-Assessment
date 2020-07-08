# random seed
randomseed = 0

# directory for the entire dataset involving three surgical tasks
dataset_dir = './data'

# name of surgical task in {Knot_Tying, Needle_Passing, Suturing} for data fetching
surg_task = 'Knot_Tying'

# directory tp store train/test split lists and annotations
dataset_task_dir = dataset_dir  + '/' + surg_task

# directory containing extracted features in the form of tensor
task_features_dir = dataset_task_dir + '/features'

# directory containing the respective features
features_C3D_4096_path = task_features_dir + '/features-C3D-4096/'
features_C3D_8192_path = task_features_dir + '/features-C3D-8192/'  # for fine-tune purpose
features_C3D_4096_upsampled_path = task_features_dir + '/features-C3D-4096-upsampled/'  # for gesture recognition
features_C3D_8192_upsampled_path = task_features_dir + '/features-C3D-8192-upsampled/'
features_ResNet_path = task_features_dir + '/features-ResNet/'

# directory and files containing the annotations
gt_path = dataset_task_dir + '/groundTruth/'
impre_path = dataset_task_dir + '/transcriptions/'
meta_file = dataset_task_dir + '/meta_file_' + surg_task + '.txt'

# file storing the gesture vocabulary
mapping_file = dataset_task_dir + '/mapping.txt'

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])
num_classes = len(actions_dict)

# score std to normalize the score within a range
final_score_std = 30

# whether to enable the auxiliary task in MTL-VF
with_gesture_recognition = True
with_skill_classification = True

# hyper-params for gesture recognition
num_stages = 4
num_layers = 10
num_f_maps = 64

# dimension of extracted video descriptor, 4096 for C3D, 1000 for ResNet101
features_dim = 4096

# whether to fine-tune the backbone feature extractor (C3D/ResNet)
finetune_backbone = False

# number of gesture types within the gesture vocabulary
if surg_task == 'Knot_Tying':
    num_gesture_types = 6
elif surg_task == 'Needle_Passing' or surg_task == 'Suturing':
    num_gesture_types = 10

# number of training epochs
max_epochs = 100

# at what epoch interval to save the model
model_ckpt_interval = 1

# default learning rates for different tasks and task uncertainty
SA_learning_rate = 0.00001
SEG_learning_rate = 0.0005
CLS_learning_rate = 0.00001
ETA_learning_rate = 0.00001

# use which model as backbone, C3D or resnet-101
feature_extractor = 'C3D'

if feature_extractor == 'C3D':
    if finetune_backbone:
        features_path = features_C3D_8192_path
        features_full_path = features_C3D_8192_upsampled_path
    else:
        features_path = features_C3D_4096_path
        features_full_path = features_C3D_4096_upsampled_path
elif feature_extractor == 'resnet-101':
    features_path = features_ResNet_path
    features_full_path = None

# mapping of self-proclaimed skill levels
sp_skill_map = {'N':0, 'I':1, 'E':2}

# implement which framework, MTL-VF or IMTL-AGF
mode = 'MTL-VF'
