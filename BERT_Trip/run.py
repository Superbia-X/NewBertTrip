import numpy as np
import pandas as pd
import time
from util import log
from model.tester import ResultManager
from sklearn.model_selection import KFold
from data.weeplaces import PretrainFileProcessor
from datetime import datetime
from model.pretrain import TripTrainer
from util import get_model, get_kfold_data, get_data_dir
#from model.pretrain import TripPretrainer       model     model               model model
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold

if __name__ == "__main__":
    mlm_probability = 0.15
    batch_size = 32
    hidden_size = 768
    num_hidden_layers = 8
    num_attention_heads = 8

    pretrain_file_processor = PretrainFileProcessor()
    #datasets = [ 'melb', 'edin', 'toro', 'glas', 'osaka']
    datasets = [
    ('Osak', 47 * 5),
    #('melb', 442 * 2),
    ('Glas', 112 * 10),
    # ('Edin', 634 * 2),
    # ('Toro', 335 * 5),
    #('weeplaces_poi_12800_length_3-15-numtraj_147287', 147287),
    #('weeplaces_poi_6400_length_3-15-numtraj_90248', 90248 * 3),
    #('weeplaces_poi_3200_length_3-15-numtraj_53901', 53901 * 3),
    #('weeplaces_poi_1600_length_3-15-numtraj_31545', 31545),
    #('weeplaces_poi_801_length_3-15-numtraj_19385', 19385 * 3),
    #('weeplaces_poi_400_length_3-15-numtraj_12288', 12288),
    # ('weeplaces_poi_200_length_3-15-numtraj_7790', 7790),
    #('weeplaces_poi_100_length_3-15-numtraj_4497', 4497),
    #('weeplaces_poi_50_length_3-15-numtraj_2134', 2134),
    #('weeplaces_poi_25_length_3-15-numtraj_765', 765),# * 3
    ]
    models = ['bert_trip']
    for (dataset, expected_size) in datasets:
        data_dir = f'{get_data_dir()}/{dataset}'
        data_file_path = f'{data_dir}/data.csv'
        pretrain_data_file_path = f'{data_dir}/pretrain_data.csv'
        share_pretrain_data_file_path = f'{data_dir}/share_pretrain_data.csv'
        train_data_file_path = f'{data_dir}/train.csv'
        test_data_file_path = f'{data_dir}/test.csv'

        df = pd.read_csv(data_file_path, sep = '|', header = None, ).rename(columns = {0: 'user' , 1: 'traj', 7: 'timestamp'})
        date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
        results = ResultManager()
        kf = KFold(n_splits=len(df), random_state=None, shuffle=True)
        index = 0
        f1_list = []
        self_loop_list = []
        repetition_list = []
        pairs_f1_list = []
        distance_shift_list = []
        start_ratio_list = []
        end_ratio_list = []
        for train_index, test_index in kf.split(df):
            train_data = df.iloc[train_index]
            test_data = df.iloc[test_index]
            # m = TripTrainer(base_model, **kw)
            # m.reset()
        # for fold in range(5):
        #     train_data, test_data, random_state  = get_kfold_data(df, fixed_random_state = fixed_random_state)
            train_data.to_csv(train_data_file_path, sep = '|', header = False, index = None)
            test_data.to_csv(test_data_file_path, sep = '|', header = False, index = None)
            first = True
            results = ResultManager()
            for model in models:
                print(dataset, model, 'hidden_size:', hidden_size)
                base_model, evaluator, config = get_model(model)
                kw = {
                    'dataset': dataset,
                    'mlm_probability': mlm_probability,
                    'model_type': model,
                    'hidden_size': hidden_size,
                    'intermediate_size': hidden_size * 4,
                    'num_hidden_layers': num_hidden_layers,
                    'num_attention_heads': num_attention_heads,
                }
                kw = kw | config
                m = TripTrainer(base_model, **kw)
                tester = evaluator(m.config)
                # reset network weights
                m.reset()
                # create pretrain_file using transition matrix by train data and then save it
                # only useful for the Flickr datasets or for the WeePlaces datasets with lower numbers of POIs.

                """
                if first:
                    print("pretrain file")
                    pretrain_df, g = pretrain_file_processor.create_pretrain_file(dataset = dataset, train_data = m.config.train_data, train_data2 = m.config.train_data2, poi_data = m.config.poi_vocab_file, sep = m.config.feature_sep, expected_size = expected_size)
                    pretrain_df.to_csv(share_pretrain_data_file_path, sep = m.config.feature_sep, header = None, index = False)
                    first = False
                for epoch in range(0, 5):
                    m.train(share_pretrain_data_file_path, batch_size = batch_size, epochs = 5, save_model = False)
                    result = tester.run(m.config.test_data, model = m.model, epoch = epoch, strategy = 'one_by_one', verbose = False)
                    results.insert_fold_records(model, m.config, random_state, fold, epoch, result, 'one_by_one')
                    result = tester.run(m.config.test_data, model = m.model, epoch = epoch, strategy = 'all_in_once', verbose = False)
                    results.insert_fold_records(model, m.config, random_state, fold, epoch, result, 'all_in_once')
                """
                random_state = 2024
                N = int(expected_size / len(train_data.index))
                train_df = pd.concat([train_data] * N, ignore_index = True)
                for epoch in range(50):
                    train_df = train_df.sample(frac = 1, random_state = random_state + epoch)
                    train_df.to_csv(m.config.pretrain_data, sep = m.config.feature_sep, header = None, index = False)
                    m.train(m.config.pretrain_data, batch_size = batch_size, epochs = 1, save_model = True)
                    result = tester.run(m.config.test_data, dataset, model = m.model, epoch = epoch, strategy = 'one_by_one', verbose = False)
                    results.insert_fold_records(model, dataset, m.config, random_state, index, epoch, result, 'one_by_one')
                    print("+++++++++++++++++++++++++++++++++++++++++++=")
                    print(epoch)
                    print("+++++++++++++++++++++++++++++++++++++++++++=")
                    # results.insert_fold_records(model, m.config, fold, epoch, result, 'one_by_one')
                    # results.print_best_results()
                best_f1_result, best_pairs_f1_result, best_corresponding_sf, best_corresponding_rp, distance_shift, start_ratio, end_ratio = results.print_best_results(dataset)
                with open(f'distance_count.txt', 'a') as file:
                    file.write(f'{dataset} {index} {distance_shift}\n')
                distance_shift_list.append(distance_shift)
                f1_list.append(best_f1_result)
                pairs_f1_list.append(best_pairs_f1_result)
                self_loop_list.append(best_corresponding_sf)
                repetition_list.append(best_corresponding_rp)
                start_ratio_list.append(start_ratio)
                end_ratio_list.append(end_ratio)
                results.save_fold_result(model, m.config, dataset, index, date, expected_size, mlm_probability, random_state, len(train_data.index), len(test_data.index))
                # results.save_fold_result(model, m.config, dataset, index, date, expected_size, mlm_probability, len(train_index), len(test_index))
                index += 1
        with open(f'Four_result.txt', 'a') as file:
            file.write(f'{dataset} f1: {np.mean(f1_list):.3f} pairs_f1: {np.mean(pairs_f1_list):.3f} self_loop: {np.mean(self_loop_list):.3f} repetition: {np.mean(repetition_list):.3f}\n, distance_shift: {np.mean(distance_shift_list):.3f} start_ratio:{np.mean(start_ratio_list):.3f} end_ratio: {np.mean(end_ratio_list):.3f}\n')
