FOLDER_NAME = 'ariadne1_gtexpert'
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
SUMMARY_WINDOW = 32
LOAD_MODEL = False # do you want to load the model trained before
SAVE_IMG_GAP = 100

N_AGENTS = 4

CELL_SIZE = 0.4  # meter
NODE_RESOLUTION = 3.2  # meter
DOWNSAMPLE_SIZE = NODE_RESOLUTION // CELL_SIZE

SENSOR_RANGE = 20  # meter
UTILITY_RANGE = 0.8 * SENSOR_RANGE
MIN_UTILITY = 1
FRONTIER_CELL_SIZE = 4 * CELL_SIZE

LOCAL_MAP_SIZE = 40  # meter
EXTENDED_LOCAL_MAP_SIZE = 6 * SENSOR_RANGE * 1.05

LOCAL_K_SIZE = 25  # the number of neighboring nodes
LOCAL_NODE_PADDING_SIZE = 360  # the number of nodes will be padded to this value

MAX_EPISODE_STEP = 128

LOCAL_NODE_INPUT_DIM = 5
EMBEDDING_DIM = 128

REPLAY_SIZE = 20000
MINIMUM_BUFFER_SIZE = 5000
BATCH_SIZE = 256
LR = 1e-5
GAMMA = 0.95
EXPERT = 'ground_truth'  # 'tare' or 'ground_truth'

USE_GPU = False  # do you want to collect training data using GPUs
USE_GPU_GLOBAL = True # do you want to train the network using GPUs
NUM_GPU = 0
NUM_META_AGENT = 32

USE_WANDB = False
