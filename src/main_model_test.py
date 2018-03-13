from ner_evaluation import myTagger
import ner_utils as utils
import os, os.path
import sys
import pycrfsuite
import operator

MODEL_NUMBER = '20180313-162854'
MODEL_DIR_PATH = '../'+MODEL_NUMBER+'/'
BASE_MODEL_PATH = MODEL_DIR_PATH+ 'base_model/'
BASE_LINE_NAME = MODEL_NUMBER + '_base'
TEST_FILE = '../data/etry_NEcorpus_onlyA_ej_test.txt'
OUT_PATH = './tmpdir'

tagger = pycrfsuite.Tagger()

def make_prediction(X_test, full_model_name):  # not used when this is component classifier
    tagger.open(full_model_name)
    y_preds = []
    for xseq in X_test:
        y_pred_ori = tagger.tag(xseq)
        y_pred_post, _ = utils._remove_I_without_B(y_pred_ori)
        y_preds.append(y_pred_post)
    tagger.close()
    return y_preds

def print_name() :
    print('MODEL_NUMBER = ' + MODEL_NUMBER)
    print('BASE_MODEL_PATH = ' + BASE_MODEL_PATH)
    print('BASE_LINE_NAME = ' + BASE_LINE_NAME)
    print('TEST_FILE = ' + TEST_FILE)

def write_result(test_sents, y_pred,file_full_name) :
    with open(file_full_name, 'w') as wf:
        for s_idx in range(len(test_sents)):
            for tup_idx in range(len(test_sents[s_idx])):
                mark = ''
                if test_sents[s_idx][tup_idx][3] != y_pred[s_idx][tup_idx] :
                    mark = '$$'
                out_str = mark + test_sents[s_idx][tup_idx][0] + ' ' + test_sents[s_idx][tup_idx][1] + ' '+test_sents[s_idx][tup_idx][2] + ' ' \
                          + y_pred[s_idx][tup_idx] +'('+test_sents[s_idx][tup_idx][3]+')'+ '\n'
                wf.write(out_str.encode('utf-8'))
            wf.write('\n'.encode('utf-8'))

def main(argvs) :
    f1scores = []
    bio_save_dir = argvs[0]
    print_name()
    print('Reading files')
    test_sents, X_test, y_test = utils.read_labeled_text_data(TEST_FILE, encoding='utf-8')

    tester = myTagger(X_test=X_test, y_test=y_test, test_sents=test_sents)
    num_of_mds = len([name for name in os.listdir(BASE_MODEL_PATH) if os.path.isfile(os.path.join(BASE_MODEL_PATH, name))])
    if not os.path.exists(bio_save_dir):
        os.makedirs(bio_save_dir)
    for boot_iter in range(num_of_mds):
        print('\nboot : ' + str(boot_iter) + '/' + str(num_of_mds-1))
        if os.path.exists(BASE_MODEL_PATH + BASE_LINE_NAME + str(boot_iter) + '.crfsuite') is False :
            continue
        print('model : '+ BASE_MODEL_PATH + BASE_LINE_NAME + str(boot_iter) + '.crfsuite')

        y_pred = make_prediction(tester.X_test,BASE_MODEL_PATH + BASE_LINE_NAME + str(boot_iter) + '.crfsuite', )
        print('test : ' + TEST_FILE)
        f1scores.append(tester.eval_prediction(y_pred, tag_conf_table = False,log=False))
        write_result(test_sents, y_pred,bio_save_dir+'/'+str(boot_iter)+'.bio')
        max_index, max_f1 = max(enumerate(f1scores), key=operator.itemgetter(1))
        print('max score : ' + str(max_index) + 'st, ' + str(max_f1))

    out_str = ''
    for f in f1scores :
        out_str+= str(f)+'\t'
    out_str += '\n'
    print out_str
if __name__ == "__main__" :
    main([OUT_PATH])