#################RELOAD####################
RELOAD = False
LOADING_MODEL_NUMBER = ''#'20171012-165719'
LOADING_DIR = ''#'../20171012-165719/'
START_ITER = 0
###########################################
REMOVE_ALL_ZERO = True
ENCODING = 'utf-8'

CRF_ITER = 200 #FULL is 2147483647 #this

TRAIN_FILE = 'etry_NEcorpus_onlyA_ej_train.txt'
TEST_FILE = 'etry_NEcorpus_onlyA_ej_test.txt'
UNLABELED_FILE = 'p_to_t_unlabeled20170908_10000.txt'

BOOTSTRAP_SAMPLE_SIZE = 2500
BOOT_ITER_LIMIT = 10

BAGGING = True
NUM_BAGGING_MODEL = 5
VOTE_ON_DIST = True

FIXED_MIN_SEQ_PROB = 0.75
FIXED_MIN_MARGINAL_PROB = 0.9
FIXED_MIN_PERC_OF_VOTES = None #when VOTE_ON_DIST is False

TAG = ['B-DT','B-LC', 'B-OG','B-PS','B-TI','I-DT','I-LC','I-OG','I-PS','I-TI', 'O']
NE_CLASSES = ['DT','LC', 'OG','PS','TI']
ft_idx = {'word':1,'word.isdigit':2,'postag':3,'postag[:2]':4,'position':5, 'len_word':6}

def print_setting() :
    if RELOAD is True:
        print('LOADING_MODEL_NUMBER = ' + LOADING_MODEL_NUMBER)
        print('LOADING_DIR = ' + LOADING_DIR)
    print('START_ITER = ' + str(START_ITER))
    print('REMOVE_ALL_ZERO = ' + str(REMOVE_ALL_ZERO))
    print('ENCODING = ' + str(ENCODING))

    print('CRF_ITER = ' + str(CRF_ITER))

    print('TRAIN_FILE = ' + str(TRAIN_FILE))
    print('TEST_FILE = ' + str(TEST_FILE))
    print('UNLABELED_FILE = ' + str(UNLABELED_FILE))

    print('BOOTSTRAP_SAMPLE_SIZE = ' +str(BOOTSTRAP_SAMPLE_SIZE))
    print('BOOT_ITER_LIMIT = ' +str(BOOT_ITER_LIMIT))

    print('BAGGING = ' + str(BAGGING))
    print('NUM_BAGGING_MODEL = ' + str(NUM_BAGGING_MODEL))
    print('VOTE_ON_DIST = ' + str(VOTE_ON_DIST))

    print('FIXED_MIN_SEQ_PROB = ' + str(FIXED_MIN_SEQ_PROB))
    print('FIXED_MIN_MARGINAL_PROB = ' + str(FIXED_MIN_MARGINAL_PROB))

