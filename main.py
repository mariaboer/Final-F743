import logging
import os

import notebooks.AdaBoosterRegressor_8 as ABR
import notebooks.Chunkfile_0 as loader  # noqa
import notebooks.Chunkfile_0 as loader  # noqa
import notebooks.DataVisuals_2 as visuals
import notebooks.GradientBoostingRegressor_7 as GBR
import notebooks.GridSearchCV_6 as GridSearch
import notebooks.K_Nearest_Neighbor_3 as KNN
import notebooks.MLP_Regressor_10 as MLP
import notebooks.Preprocessing_1 as preproc  # noqa
import notebooks.Principle_Component_Analysis_4 as PCA
import notebooks.RandomForestRegressor_9 as RFR
import notebooks.XGBoost_5 as XGB


def configure_logging(level=logging.INFO, log_path=None):
    if log_path is None:
        log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    log_file = os.path.join(log_path, f"{os.path.dirname(os.path.realpath(__file__)).split(os.sep)[-1]}.log")
    if level == logging.INFO or logging.NOTSET:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    elif level == logging.DEBUG or level == logging.ERROR:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(filename)s function:%(funcName)s()\t[%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )


def str_or_none(value):
    return value if value is None else str(value)


def main():
    rootpath = os.path.dirname(os.path.realpath(__file__))
    visuals.main(rootpath, 'Memory')
    KNN.main(rootpath, 'Memory')
    PCA.main(rootpath, 'Memory')
    XGB.main(rootpath, 'Memory')
    GridSearch.main(rootpath, 'Memory')
    GBR.main(rootpath, 'Memory')
    ABR.main(rootpath, 'Memory')
    MLP.main(rootpath, 'Memory')
    RFR.main(rootpath, 'Memory')


#     loader.main(rootpath)
#     preproc.main(rootpath, 'Memory')
#
#    threads = []
#    active_threads = []
#    max_threads = 2  # Adjust this number to reduce the amount of memory usage
#
#    threads.append(threading.Thread(target=visuals.main, args=(rootpath, 'Memory')))
#    threads.append(threading.Thread(target=KNN.main, args=(rootpath, 'Memory')))
#    threads.append(threading.Thread(target=PCA.main, args=(rootpath, 'Memory')))
#    threads.append(threading.Thread(target=XGB.main, args=(rootpath, 'Memory')))
#    threads.append(threading.Thread(target=GridSearch.main, args=(rootpath, 'Memory')))
#    threads.append(threading.Thread(target=GBR.main, args=(rootpath, 'Memory')))
#    threads.append(threading.Thread(target=ABR.main, args=(rootpath, 'Memory')))
#    threads.append(threading.Thread(target=RFR.main, args=(rootpath, 'Memory')))
#    threads.append(threading.Thread(target=MLP.main, args=(rootpath, 'Memory')))
#    threads.append(threading.Thread(target=NN.main, args=(rootpath,'Memory')))
#
#    while len(threads) > 0:
#        if threading.active_count() < max_threads:
#            threads.pop().start()
#            logging.debug(f"Thread started. Active threads: {threading.active_count()}")
#            sleep(1)


if __name__ == '__main__':
    try:
        # Steps 1 and 2 in main is preprocessing and pickle files will be saved.
        # This process takes up to 30 minutes to run and is highly recommended to be run only 1x.
        # The pickle file is used for all subsequent steps
        # The preprocessing step requires ~25GB of RAM.
        # Please ensure you have enough RAM before running these steps.
        # If you do not have enough RAM, please run the notebooks individually.
        configure_logging(logging.INFO)
        main()
        print('All threads completed')
    except KeyboardInterrupt:
        print('Program terminated by user')
        exit(-1)
    except Exception as e:
        print(e)
        exit(-1)
