# -*- coding: utf-8 -*-
"""
Created on Fri July 7

@author: gazkune
"""

#USAGE: python action2vec.py actions.txt actions.model actions.vector
 
import logging
import os.path
import sys
import multiprocessing
 
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

DIR = "action2vec/"
IFILE1 = "continuous_complete_numeric.txt"
IFILE2 = "continuous_complete_ranges_2.txt"
IFILE3 = "continuous_no_t.txt"
IFILE4 = "lined_complete_numeric.txt"
IFILE5 = "lined_complete_ranges_2.txt"
IFILE6 = "lined_no_t.txt"

IFILES = [IFILE1, IFILE2, IFILE3, IFILE4, IFILE5, IFILE6]

VEC_SIZE = [50, 200, 400]
WIN_SIZE = [5, 10]

 
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
 
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
 
    # check and process input arguments
    """
    if len(sys.argv) < 4:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    inp, outp1, outp2 = sys.argv[1:4]
    """
    for filename in IFILES:
        for vec in VEC_SIZE:
            for win in WIN_SIZE:
                model = Word2Vec(LineSentence(DIR+filename), size=vec, window=win, min_count=3,
                         workers=multiprocessing.cpu_count())
 
                # trim unneeded model memory = use(much) less RAM
                #model.init_sims(replace=True)
                outp1 = DIR + filename.split(".")[0] + "_" + str(vec) + "_" + str(win) + ".model"
                outp2 = DIR + filename.split(".")[0] + "_" + str(vec) + "_" + str(win) + ".vector"
                model.save(outp1)
                model.save_word2vec_format(outp2, binary=False)

                print outp1, outp2, "processed and stored"

        
    print 'FIN'  
