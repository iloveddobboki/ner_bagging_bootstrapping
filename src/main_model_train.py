import time
from ner_evaluation import myTagger
from ner_basicmodel import BasicModel
from ner_baggingmodel import BaggingModel
from ner_load import Loader
import ner_utils as utils
import ner_settings as st
import os.path
if st.RELOAD is False :
    MODEL_NUMBER = time.strftime("%Y%m%d-%H%M%S")
    BASE_MODEL_PATH = '../'+ MODEL_NUMBER + '/base_model/'
    BASE_DATA_SAVE_PATH = '../'+ MODEL_NUMBER + '/base_data/'
    BAGGING_MODEL_PATH = '../'+ MODEL_NUMBER + '/bagging_model/'
    BASE_LINE_NAME = MODEL_NUMBER + '_base'
    BAGGING_MODEL_NAME = MODEL_NUMBER + '_bagging'
#    START_ITER = 0
else :
    MODEL_NUMBER = st.LOADING_MODEL_NUMBER
    BASE_MODEL_PATH = st.LOADING_DIR+ 'base_model/'
    BASE_DATA_SAVE_PATH = st.LOADING_DIR + 'base_data/'
    st.LOG = '../' + MODEL_NUMBER + '/train.log'
    ACTIVE_DIR = '../' + MODEL_NUMBER + '/active_learning/'
    BAGGING_MODEL_PATH = st.LOADING_DIR + 'bagging_model/'
#    START_ITER = st.START_ITER
    BASE_LINE_NAME = st.LOADING_MODEL_NUMBER + '_base'
    BAGGING_MODEL_NAME = st.LOADING_MODEL_NUMBER + '_bagging'

LOAD_MODEL_N = st.START_ITER

def print_name() :
    print('MODEL_NUMBER = ' + MODEL_NUMBER)
    print('BASE_MODEL_PATH = ' + BASE_MODEL_PATH)
    print('BASE_DATA_SAVE_PATH = ' + BASE_DATA_SAVE_PATH)
    print('BASE_LINE_NAME = ' + BASE_LINE_NAME)
    print('BAGGING_MODEL_PATH = ' + BAGGING_MODEL_PATH)
    print('BAGGING_MODEL_NAME = ' + BAGGING_MODEL_NAME+'\n')

def main() :
    if not os.path.exists('../'+ MODEL_NUMBER):
        os.makedirs('../'+ MODEL_NUMBER)
    if st.RELOAD == True :
        print('---------reload---------')
    st.print_setting()
    print_name()
    print ('Reading files')
    test_sents, X_test, y_test = utils.read_labeled_text_data('../data/' + st.TEST_FILE, encoding=st.ENCODING)##20170912
    train_sents, X_train, y_train = utils.read_labeled_text_data('../data/' + st.TRAIN_FILE, encoding=st.ENCODING)##20170912
    unlabeled_sents, X_unlabeled, y_unlabeled = utils.read_labeled_text_data('../data/' + st.UNLABELED_FILE, encoding=st.ENCODING)
    #unlabeled_sents, X_unlabeled, y_unlabeled, _ = utils.read_labeled_pickle_data('../data/' + st.UNLABELED_FILE) ##20170912

    if st.RELOAD is False :
        X_basiccrf = X_train
        y_basiccrf = y_train
        y_basiccrf_mar_p = utils.generate_all(y_basiccrf, 1)
    else :
        loader = Loader(BASE_DATA_SAVE_PATH, LOAD_MODEL_N)
        X_basiccrf = loader.get_X()
        y_basiccrf = loader.get_y()
        y_basiccrf_mar_p = loader.get_yprob()

    X_unlabeled_devied = utils.split_set_w_size(X_unlabeled,st.BOOTSTRAP_SAMPLE_SIZE,st.BOOT_ITER_LIMIT)
    tester = myTagger(X_test=X_test, y_test=y_test, test_sents=test_sents)
    basic_CRF = BasicModel(BASE_MODEL_PATH, BASE_LINE_NAME, BASE_DATA_SAVE_PATH, st.START_ITER)
    bagging_model = BaggingModel(BAGGING_MODEL_PATH, BAGGING_MODEL_NAME, num_of_comp_mds=st.NUM_BAGGING_MODEL,
                                 boot_sample_size=st.BOOTSTRAP_SAMPLE_SIZE / st.NUM_BAGGING_MODEL, X_labeled=X_train,
                                 y_labeled=y_train, save_path=None, start_iter = st.START_ITER)


    for boot_iter in range(st.START_ITER, len(X_unlabeled_devied)):
        print ('boot : ' + str(boot_iter) + '/' + str(len(X_unlabeled_devied)))
        print('Training Basic CRF')

        X_unlabeled_now = X_unlabeled_devied[boot_iter]

        basic_CRF.add_n_train_CRF(X_basiccrf, y_basiccrf,y_basiccrf_mar_p)

        tester.eval_prediction(basic_CRF.make_prediction(tester.X_test,remove_all_o = False)[0])
        y_pred_u = basic_CRF.make_prediction(X_unlabeled_now, remove_all_o=st.REMOVE_ALL_ZERO, min_conf=st.FIXED_MIN_SEQ_PROB)[0]

        if st.BAGGING is False :
            X_basiccrf = X_unlabeled_now
            y_basiccrf = y_pred_u
            y_basiccrf_mar_p = None
            continue
        print('Training Bagging CRFs')
        bagging_model.set_selflabeled_data_n_train(X_unlabeled_now, y_pred_u)
        y_pred_test, _ = bagging_model.make_prediction(tester.X_test, remove_all_o=False)
        tester.eval_prediction(y_pred_test)
        X_basiccrf = X_unlabeled_now
        y_basiccrf, y_basiccrf_mar_p = bagging_model.make_prediction(X_basiccrf,
                                                                     remove_all_o=st.REMOVE_ALL_ZERO,
                                                                     min_conf=st.FIXED_MIN_MARGINAL_PROB if st.VOTE_ON_DIST is True else st.FIXED_MIN_PERC_OF_VOTES)

    boot_iter += 1
    print('boot : ' + str(boot_iter) + '/' + str(len(X_unlabeled_devied)))
    print('Training Basic CRF')

    basic_CRF.add_n_train_CRF(X_basiccrf, y_basiccrf, y_basiccrf_mar_p)

    tester.eval_prediction(basic_CRF.make_prediction(tester.X_test, remove_all_o = False)[0])
if __name__ == "__main__" :
    main()
