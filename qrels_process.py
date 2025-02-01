from datasets import load_dataset
from tqdm import tqdm
import pickle
import dataset
from datasets import disable_progress_bar
import re
# 진행 상태 표시 비활성화
disable_progress_bar()

#########################
dataset_list = ["msmarco", "nq", "hotpotqa", "arguana", "webis-touche2020", "dbpedia-entity", "fever"]
dataset_name = "fiqa"

#########################

print(f"data set: {dataset_name}")

query = dataset.DATASET_BUILDERS["BeIR/"+dataset_name +"/queries"].build(load_dataset("BeIR/" + dataset_name, "queries"))
qrels = dataset.DATASET_BUILDERS["BeIR/"+dataset_name +"-qrels/train"].build(load_dataset("BeIR/" + dataset_name + "-qrels", "default"))

qrels_set = {}
for q in tqdm(qrels):
    if q["query-id"] not in qrels_set.keys():
        qrels_set[q["query-id"]] = []
    qrels_set[q["query-id"]].append((q["doc-id"]))
print("lenth of qrels_set: ", len(qrels_set))

pre_test_set = {}
for key, value in qrels_set.items():
    # if max(value) <= 530496 and len(value) > 1:  ## msmarco 에만 적용
    if len(value) > 3:                              ## 나머지 다 적용 
    # if filter_numbers(value) and len(value) > 1:
        pre_test_set[key] = value

test_set = {}
for q_id, doc_list in tqdm(pre_test_set.items()):
    q_text = query.filter(lambda row : row["query-id"] == str(q_id))["text"]
    test_set[q_text[0]] = doc_list  

print("lengh of test_set: ",len(test_set) )


path = dataset_name + "_qrels_mt3.pickle"
with open(path, 'wb') as f:
    pickle.dump(test_set, f)

