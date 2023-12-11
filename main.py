import os
import threading
from time import sleep

import notebooks.AdaBoosterRegressor_8 as ABR
import notebooks.Basic_NN_11 as NN
import notebooks.Chunkfile_0 as loader
import notebooks.DataVisuals_2 as visuals
import notebooks.GradientBoostingRegressor_7 as GBR
import notebooks.GridSearchCV_6 as GridSearch
import notebooks.K_Nearest_Neighbor_3 as KNN
import notebooks.MLP_Regressor_10 as MLP
import notebooks.Preprocessing_1 as preproc
import notebooks.Principle_Component_Analysis_4 as PCA
import notebooks.RandomForestRegressor_9 as RFR
import notebooks.XGBoost_5 as XGB


def main():
    rootpath = os.path.dirname(__file__)
    loader.main(rootpath)
    preproc.main(rootpath)

    threads = []
    max_threads = 1  # Adjust this number to reduce the amount of memory usage

    threads.append(threading.Thread(target=visuals.main, args=(rootpath,)))
    threads.append(threading.Thread(target=KNN.main, args=(rootpath,)))
    threads.append(threading.Thread(target=PCA.main, args=(rootpath,)))
    threads.append(threading.Thread(target=XGB.main, args=(rootpath,)))
    threads.append(threading.Thread(target=GridSearch.main, args=(rootpath,)))
    threads.append(threading.Thread(target=GBR.main, args=(rootpath,)))
    threads.append(threading.Thread(target=ABR.main, args=(rootpath,)))
    threads.append(threading.Thread(target=RFR.main, args=(rootpath,)))
    threads.append(threading.Thread(target=MLP.main, args=(rootpath,)))
    threads.append(threading.Thread(target=NN.main, args=(rootpath,)))

    while len(threads) > 0:
        if threading.active_count() < max_threads:
            threads.pop().start()
            sleep(1)


if __name__ == '__main__':
    try:
        # Steps 1 and 2 in main is preprocessing and pickle files will be saved.
        # This process takes up to 30 minutes to run and is highly recommended to be run only 1x.
        # The pickle file is used for all subsequent steps
        # The preprocessing step requires ~25GB of RAM.
        # Please ensure you have enough RAM before running these steps.
        # If you do not have enough RAM, please run the notebooks individually.
        main()
        print('All threads completed')
    except KeyboardInterrupt:
        print('Program terminated by user')
        exit(-1)
    except Exception as e:
        print(e)
        exit(-1)
