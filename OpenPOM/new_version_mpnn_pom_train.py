import deepchem as dc
from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
from openpom.utils.data_utils import get_class_imbalance_ratio
from openpom.models.mpnn_pom import MPNNPOMModel
from datetime import datetime
import pandas

TASKS = [
'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal',
'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy',
'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt',
'buttery', 'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery',
'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean',
'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked', 'cooling',
'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy',
'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit skin',
'fruity', 'garlic', 'gassy', 'geranium', 'grape', 'grapefruit', 'grassy',
'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth',
'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery',
'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 'metallic',
'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 'nutty',
'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone',
'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn',
'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted',
'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy',
'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet',
'tea', 'terpenic', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable',
'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody'
]
print("No of tasks: ", len(TASKS))

# The curated dataset can also found at `openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv` in the repo.

# input_file = '/mnt/mmlab_shared/AnnikaS/openpom-main/openpom/data/curated_datasets/All_Food_Molecules_4_recath_Key_shorted_OHE_OpenPOM3.csv' # or new downloaded file path
input_file_train = '/mnt/mmlab_shared/AnnikaS/openpom-main/openpom/data/curated_datasets/OpenPOM_train_dataset_All_Food_6_shortend_recat.csv'
input_file_test = '/mnt/mmlab_shared/AnnikaS/openpom-main/openpom/data/curated_datasets/OpenPOM_test_dataset_All_Food_6_shortend_recat.csv'
# get dataset

featurizer = GraphFeaturizer()
smiles_field = 'SMILES'
loader = dc.data.CSVLoader(tasks=TASKS,
                   feature_field=smiles_field,
                   featurizer=featurizer)
train_dataset = loader.create_dataset(inputs=[input_file_train])
test_dataset = loader.create_dataset(inputs=[input_file_test])
n_tasks = len(train_dataset.tasks)
n_tasks = len(test_dataset.tasks)


# get train valid test splits

# randomstratifiedsplitter = dc.splits.RandomStratifiedSplitter()
# train_dataset, test_dataset, valid_dataset = randomstratifiedsplitter.train_valid_test_split(dataset, frac_train = 0.8, frac_valid = 0.1, frac_test = 0.1, seed = 1)
# #%%
print("train_dataset: ", len(train_dataset))
# print("valid_dataset: ", len(valid_dataset))
print("test_dataset: ", len(test_dataset))


train_ratios = get_class_imbalance_ratio(train_dataset)
assert len(train_ratios) == n_tasks


#learning_rate = 0.0001 #normal Neural networks start at 0.0001 learning rate
opt_value=50
max_value=150
test_scores_all=[]
# initialize model

for learning_rate in [0.0001,0.0005, 0.001, 0.005, 0.01]:
  for epoch_number in [100,150,200,250]:
    model = MPNNPOMModel(n_tasks = n_tasks,
                            batch_size=128,
                            learning_rate=learning_rate,
                            class_imbalance_ratio = train_ratios, #None in google collab verion
                            loss_aggr_type = 'sum',
                            node_out_feats = 100,
                            edge_hidden_feats = 75,
                            edge_out_feats = 100,
                            num_step_message_passing = 5,
                            mpnn_residual = True,
                            message_aggregator_type = 'sum',
                            mode = 'classification',
                            number_atom_features = GraphConvConstants.ATOM_FDIM,
                            number_bond_features = GraphConvConstants.BOND_FDIM,
                            n_classes = 1,
                            readout_type = 'set2set',
                            num_step_set2set = 3,
                            num_layer_set2set = 2,
                            ffn_hidden_list= [392, 392],
                            ffn_embeddings = 256,
                            ffn_activation = 'relu',
                            ffn_dropout_p = 0.12,
                            ffn_dropout_at_input_no_act = False,
                            weight_decay = 1e-5,
                            self_loop = False,
                            optimizer_name = 'adam',
                            log_frequency = 32,#10 in google collab verion
                            model_dir = f'./results_NEW_{learning_rate}_{epoch_number}_new_6', #None in google collab verion {learning_rate}
                            device_name='cuda')

    nb_epoch = epoch_number #opt_value #change to 500-1000

    metric = dc.metrics.Metric(dc.metrics.f1_score)#want it to be (average='micro')


    start_time = datetime.now()
    for epoch in range(1, nb_epoch+1):
        loss = model.fit(
              train_dataset,
              nb_epoch=1,
              max_checkpoints_to_keep=1,
              deterministic=False,
              restore=epoch>1)
        train_scores = model.evaluate(train_dataset, [metric])['f1_score']
      #   valid_scores = model.evaluate(valid_dataset, [metric])['roc_auc_score']
        print(f"New_6_testrun_{learning_rate}_{epoch_number}_epoch {epoch}/{nb_epoch} ; loss = {loss}; train_scores (f1_score) = {train_scores}") # ; valid_scores = {valid_scores}")
    model.save_checkpoint()
    end_time = datetime.now()
# copied from google collab version to get the prediction for every category
    result = model.predict(test_dataset)
    df_odor_prediction = pandas.DataFrame(result, columns = TASKS)
    df_odor_prediction.to_csv(f'OpenPOM_prediction_{learning_rate}_{nb_epoch}_new.csv', sep = ';', index=False)

#-----
    test_scores = model.evaluate(test_dataset, [metric])['f1_score']
    test_scores_all.append(test_scores)
    print("time_taken: ", str(end_time-start_time))
    print("test_score (f1_score): ", test_scores)

    opt_value=opt_value+100
    print(f"NEW_6_{learning_rate}_{epoch_number}_all f1 test scores:", test_scores_all)
# print("all f1 test scores:", test_scores_all)