TEST_METHOD = 'ground_truth'  # 'tare', 'ground_truth', 'rl'
TEST_N_AGENTS = 4

INPUT_DIM = 5
EMBEDDING_DIM = 128
K_SIZE = 20  # the number of neighbors

USE_GPU = False  # do you want to use GPUS?
NUM_GPU = 0
NUM_META_AGENT = 16  # the number of processes
FOLDER_NAME = 'ariadne1_4_agent_ir'
model_path = f'model/{FOLDER_NAME}'
gifs_path = f'results/{TEST_METHOD}/gifs'

NUM_TEST = 100
NUM_RUN = 1
SAVE_GIFS = False  # do you want to save GIFs
