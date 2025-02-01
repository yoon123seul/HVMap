from index_builder.faiss_hnsw.module import FaissHNSW
import faiss
import pickle
import numpy as np
import torch
import tqdm as tqdm
from mapping_function import *
from pymilvus import model
import dataset as dataset
from datasets import load_dataset, concatenate_datasets, disable_progress_bar
import torch.nn.functional as F
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import random

class CustomSentenceTransformer:
    def __init__(self, model_name):
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def encode_queries(self, sentences, batch_size=8):
        self.model.eval()
        embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Mean pooling over the token embeddings
            # Assumes `last_hidden_state` is the token embeddings from the model
            token_embeddings = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
            attention_mask = inputs['attention_mask']
            
            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask
            
            embeddings.append(sentence_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    def encode_documents(self, sentences, batch_size=8):
        self.model.eval()
        embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Mean pooling over the token embeddings
            # Assumes `last_hidden_state` is the token embeddings from the model
            token_embeddings = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
            attention_mask = inputs['attention_mask']
            
            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask
            
            embeddings.append(sentence_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_model_bio = model.dense.SentenceTransformerEmbeddingFunction(
        model_name="juanpablomesa/bge-base-bioasq-matryoshka",  # 모델 이름 지정
        device=device  # 디바이스 지정 ('cpu' 또는 'cuda:0')
    )

embedding_model_fin = CustomSentenceTransformer("rbhatia46/financial-rag-matryoshka")

# hetero embedding model load
embedding_model_sci = model.dense.SentenceTransformerEmbeddingFunction(
    model_name="pritamdeka/S-PubMedBert-MS-MARCO-SCIFACT", 
    device=device  
)

domain_from = "fin"
domain_to = "bio"
query_option = "all"
query_option = "500"
# query_option = "include_to"
model_option = "_big"
# model_option = ""
golden_data = ""
# golden_data = "_woG"


if domain_from =="fin" and domain_to == "bio":
    m_DB = "fin_bio"
    embedding_model_from = embedding_model_fin
    embedding_model_to = embedding_model_bio
elif domain_from =="fin" and domain_to == "sci":
    m_DB = "fin_sci"
    embedding_model_from = embedding_model_fin
    embedding_model_to = embedding_model_sci
elif domain_from =="bio" and domain_to == "fin":
    m_DB = "fin_bio"
    embedding_model_from = embedding_model_bio
    embedding_model_to = embedding_model_fin
elif domain_from =="bio" and domain_to == "sci":
    m_DB = "bio_sci"
    embedding_model_from = embedding_model_bio
    embedding_model_to = embedding_model_sci
elif domain_from =="sci" and domain_to == "fin":
    m_DB = "fin_sci"
    embedding_model_from = embedding_model_sci
    embedding_model_to = embedding_model_fin
elif domain_from =="sci" and domain_to == "bio":
    m_DB = "bio_sci"
    embedding_model_from = embedding_model_sci
    embedding_model_to = embedding_model_bio
else:
    print("error")

DB_path = "/faiss_DB/" + domain_from + "/" + domain_from + "_model.bin"
print("loading data from.. ",DB_path)
index_from = FaissHNSW.load_index("angular", {"M": 32, "efConstruction": 128}, DB_path)
index_from.set_query_arguments(128)

DB_path = "/faiss_DB/" + domain_to + "/" + domain_to + "_model.bin"
print("loading data from.. ",DB_path)
index_to = FaissHNSW.load_index("angular", {"M": 32, "efConstruction": 128}, DB_path)
index_to.set_query_arguments(128)


DB_path = "/faiss_DB/" + m_DB + "/" + domain_from + "_model.bin"
print("loading data from.. ",DB_path)
index_migrate_from = FaissHNSW.load_index("angular", {"M": 32, "efConstruction": 128}, DB_path)
index_migrate_from.set_query_arguments(128)


DB_path = "/faiss_DB/" + m_DB + "/" + domain_to + "_model.bin"
print("loading data from.. ",DB_path)
index_migrate_to = FaissHNSW.load_index("angular", {"M": 32, "efConstruction": 128}, DB_path)
index_migrate_to.set_query_arguments(128)


# path = dataset_name + "_qrels_all_mt2" + ".pickle"
path = m_DB + golden_data + "_qrels" + ".pickle"
with open(path, 'rb') as f:
    combine_set = pickle.load(f)

if query_option == "all":
    print("using all kind of query")
    path = domain_from + "_qrels" + ".pickle"
    with open(path, 'rb') as f:
        from_qrels = pickle.load(f)
    if len(from_qrels) > 250:
        random_keys = random.sample(list(from_qrels.keys()), 250)
        from_set = {key: from_qrels[key] for key in random_keys}
    else :
        from_set = from_qrels

    path = domain_to + "_qrels" + ".pickle"
    with open(path, 'rb') as f:
        to_qrels = pickle.load(f)
    if len(to_qrels) > 250:
        random_keys = random.sample(list(to_qrels.keys()), 250)
        to_set = {key: to_qrels[key] for key in random_keys}
    else :
        to_set = to_qrels
elif query_option == "include_to":
    from_set = {}
    path = domain_to + "_qrels" + ".pickle"
    with open(path, 'rb') as f:
        to_qrels = pickle.load(f)
    if len(to_qrels) > 250:
        random_keys = random.sample(list(to_qrels.keys()), 250)
        to_set = {key: to_qrels[key] for key in random_keys}
    else :
        to_set = to_qrels
else :
    print("using only intagrated query")
    from_set = {}
    to_set = {}

test_set = combine_set | from_set | to_set


q_text = list(test_set.keys())
q_answer = list(test_set.values())
test_number = len(q_text)
print(test_number)


from_query = embedding_model_from.encode_queries(q_text)
to_query = embedding_model_to.encode_queries(q_text)
print(f"query number: {len(from_query)}")



## converter load
input_size = len(from_query[0])
output_size =len(to_query[0])
converter = Big_MLP(input_size, output_size)
converter2 = Big_MLP(output_size, input_size)


converter_path = "test_models/" + domain_from + '_model--TO--' + domain_to + '_model/trainset'+ model_option + '_model.pth'
converter.load_state_dict(torch.load(converter_path))

print(f"input dim: {len(from_query[0])}")
print(f"output dim: {len(to_query[0])}")
print(f"load {converter_path}")


converter_path = "test_models/" + domain_to + '_model--TO--' + domain_from + '_model/trainset'+ model_option + '_model.pth'
converter2.load_state_dict(torch.load(converter_path))

print(f"input dim: {len(to_query[0])}")
print(f"output dim: {len(from_query[0])}")
print(f"load {converter_path}")


to_query_map = converter(to_query)

k=10
res_from = []
res_to = []
res_to_map = []
res_migrate_to = []
res_migrate_from = []
for i in tqdm.tqdm(range(len(q_text))):
    res_from.append(index_from.query(from_query[i], k))
    res_to.append(index_to.query(to_query[i], k))
    res_to_map.append(index_to.query(to_query_map[i], k))
    res_migrate_to.append(index_migrate_to.query(to_query[i], k*2))
    res_migrate_from.append(index_migrate_from.query(from_query[i], k*2))




##sorting without converting 

res_from_wo = []
for i, res in tqdm.tqdm(enumerate(res_from)):
    temp = {}
    temp["doc-id"] = res[0]
    temp["vectors"] = torch.from_numpy(res[1])
    temp["distance"] = F.cosine_similarity(temp["vectors"], torch.from_numpy(from_query[i]), dim=-1)
    res_from_wo.append(temp)
    

res_to_wo = []
for i, res in tqdm.tqdm(enumerate(res_to)):
    temp = {}
    temp["doc-id"] = res[0]
    temp["vectors"] = torch.from_numpy(res[1])
    temp["distance"] = F.cosine_similarity(temp["vectors"], torch.from_numpy(to_query[i]), dim=-1)
    res_to_wo.append(temp)

def sorted_indices_desc(list1, list2):
    combined = torch.cat((list1, list2), dim=0)
    sorted_idx = sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)
    return sorted_idx
    
just_sort = []
for r1, r2 in zip(res_from_wo, res_to_wo):
    indx = sorted_indices_desc(r1["distance"],r2["distance"])
    combine = np.concatenate((r1["doc-id"], r2["doc-id"]), axis=0)
    temp = []
    for i in indx:
        temp.append(combine[i])
    just_sort.append(temp)


##space converting 
converting_parm = 0.75
c_p = converting_parm
######################
res_from_dic = []
for i, res in tqdm.tqdm(enumerate(res_from)):
    temp = {}
    temp["doc-id"] = res[0]
    temp["vectors1"] = torch.from_numpy(res[1])
    temp["distance1"] = F.cosine_similarity(temp["vectors1"], torch.from_numpy(from_query[i]), dim=-1)
    temp["vectors"] = converter(torch.from_numpy(res[1]))
    temp["distance2"] = F.cosine_similarity(temp["vectors"], torch.from_numpy(to_query[i]), dim=-1) 
    temp["distance"] = ((1-c_p) * temp["distance2"] + c_p * temp["distance1"])
    res_from_dic.append(temp)
    
res_to_dic = []
for i, res in tqdm.tqdm(enumerate(res_to_map)):
    temp = {}
    temp["doc-id"] = res[0]
    temp["vectors1"] = torch.from_numpy(res[1])
    temp["distance1"] = F.cosine_similarity(temp["vectors1"], torch.from_numpy(to_query[i]), dim=-1)
    temp["vectors2"] = converter2(torch.from_numpy(res[1]))
    temp["distance2"] = F.cosine_similarity(temp["vectors2"], torch.from_numpy(from_query[i]), dim=-1)
    temp["distance"] = ((1-c_p) * temp["distance2"] + c_p * temp["distance1"])
    res_to_dic.append(temp)

print("space convert done")

def sorted_indices_desc(list1, list2):
    combined = torch.cat((list1, list2), dim=0)
    sorted_idx = sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)
    return sorted_idx
    
sort_convert = []
for r1, r2 in zip(res_from_dic, res_to_dic):
    indx = sorted_indices_desc(r1["distance"],r2["distance"])
    combine = np.concatenate((r1["doc-id"], r2["doc-id"]), axis=0)
    temp = []
    for i in indx:
        temp.append(combine[i])
    sort_convert.append(temp)



#################################################### sort with convert
pred_convert = {}
pred_convert["answer"] = q_answer
pred_convert["homo"] = sort_convert

#################################################### sort without convert
pred_just_sort = {}
pred_just_sort["answer"] = q_answer
pred_just_sort["homo"] = just_sort


################################################ migrate TO
pred_migrate_to = {}
pred_migrate_to["answer"] = q_answer
pred_i = []
for r in res_migrate_to:
    temp = []
    for a in r[0]:
        temp.append(a)
    pred_i.append(temp)
pred_migrate_to["homo"] = pred_i


################################################ migrate FROM
pred_migrate_from = {}
pred_migrate_from["answer"] = q_answer
pred_i = []
for r in res_migrate_from:
    temp = []
    for a in r[0]:
        temp.append(a)
    pred_i.append(temp)
pred_migrate_from["homo"] = pred_i

################################################# one by one
pred_one_by_one={}
pred_one_by_one["answer"] = q_answer
pred_homo = []
for r1, r2 in zip(res_from, res_to):
    temp = []
    for a1, a2 in zip(r1[0], r2[0]):
        temp.append(a1)
        temp.append(a2)
    pred_homo.append(temp)
pred_one_by_one["homo"] = pred_homo


# ##################################################### random
kk = 0
# ################
import random
pred_random = {}
pred_random["answer"] = q_answer
res_random = []
for r in just_sort:
    random_numbers = random.sample(range(kk, k*2), k*2-kk)
    temp = []
    for a in r[:kk]:
        temp.append(a)
    for i in random_numbers:
        temp.append(r[i])
    res_random.append(temp)
pred_random["homo"] = res_random


recall_result = {}
ndcg_result = {}
result_list = [pred_one_by_one, pred_migrate_from, pred_convert, pred_just_sort, pred_migrate_to, pred_random]
for option in range(len(result_list)):
    result = result_list[option]

    temp = [[str(num) for num in sublist] for sublist in result["answer"]]
    result["answer"] = temp
    temp = [[str(num) for num in sublist] for sublist in result["homo"]]
    result["homo"] = temp
    print(result["answer"])
    # print(result["homo"])
    from metrics import *

    number = len(result["answer"])
    recall_list = []
    ndcg_list = []
    x_list = range(1, k*2 + 1)
    for i in x_list:
        # recall@k
        print(f"result for recall at {i} ")
        r_h = 0
        r_m = 0
        r_f = 0
        p = 0
        for a, h in zip(result["answer"], result["homo"]):
            r_h += recall(a,h[:i])
        print("homo recall: " , r_h/number)
        recall_list.append(r_h/number)
        # ndcg@k
        print(f"result for ndcg at {i} ")
        ndcg_h = 0
        ndcg_m = 0
        ndcg_f = 0
        for a, h in zip(result["answer"], result["homo"]):
            ndcg_h += nDCG_score(a,h[:i])
        print("homo ndcg: " , ndcg_h/number)
        ndcg_list.append(ndcg_h/number)
    print("="*70)
    recall_result[option] = recall_list
    ndcg_result[option] = ndcg_list



import matplotlib.pyplot as plt
linewidth = 1.5

# x 값이 10일 때의 대표 값을 저장
x_target = k
color_list = ["blue","black","red","orange","gray", "green"]
plt.figure(figsize=(12, 8))

# 각 라인 그리기
plt.plot(x_list, recall_result[0], label='recall one by one', color='blue', linewidth=linewidth)
plt.plot(x_list, recall_result[1], label='recall migrate on FROM', color='black', linewidth=linewidth)
plt.plot(x_list, recall_result[2], label='recall convert', color='red', linewidth=linewidth)
plt.plot(x_list, recall_result[3], label='recall just sort', color='orange', linewidth=linewidth)
plt.plot(x_list, recall_result[4], label='recall migrate on TO', color='gray', linewidth=linewidth)
plt.plot(x_list, recall_result[5], label='recall random', color='green', linewidth=linewidth)

# x_target (x=10)에서 y 값을 가져와 표시
for i, _ in enumerate(recall_result):
    if x_target in x_list:  # x_target이 x_list에 존재하는지 확인
        idx = x_list.index(x_target)

        y_value = recall_result[i][idx]
        plt.text(k*2 - 1, i*0.05, f"{y_value:.4f}", fontsize=10, color=color_list[i])  # y 값을 텍스트로 표시

# 그래프 제목 및 라벨
plt.title(f"{domain_from} TO {domain_to} ON {m_DB}")
plt.xlabel('k value')
plt.xticks(x_list)
plt.yticks([i / 10 for i in range(0, 11)])  # Set y-ticks to show from 0.0 to 1.0 in intervals of 0.1
plt.ylabel('performance')
plt.grid(True)
plt.legend()

# 그래프 표시
plt.show()


linewidth = 1.5

# x 값이 10일 때의 대표 값을 저장
x_target = k
color_list = ["blue","black","red","orange","gray", "green"]
plt.figure(figsize=(12, 8))

# 각 라인 그리기
plt.plot(x_list, ndcg_result[0], label='ndcg one by one', color='blue', linewidth=linewidth)
plt.plot(x_list, ndcg_result[1], label='ndcg migrate on FROM', color='black', linewidth=linewidth)
plt.plot(x_list, ndcg_result[2], label='ndcg convert', color='red', linewidth=linewidth)
plt.plot(x_list, ndcg_result[3], label='ndcg just sort', color='orange', linewidth=linewidth)
plt.plot(x_list, ndcg_result[4], label='ndcg migrate on TO', color='gray', linewidth=linewidth)
plt.plot(x_list, ndcg_result[5], label='ndcg random', color='green', linewidth=linewidth)

# x_target (x=10)에서 y 값을 가져와 표시
for i, _ in enumerate(ndcg_result):
    if x_target in x_list:  # x_target이 x_list에 존재하는지 확인
        idx = x_list.index(x_target)

        y_value = ndcg_result[i][idx]
        plt.text(k*2 - 1, i*0.05, f"{y_value:.4f}", fontsize=10, color=color_list[i])  # y 값을 텍스트로 표시

# 그래프 제목 및 라벨
plt.title(f"{domain_from} TO {domain_to} ON {m_DB}")
plt.xlabel('k value')
plt.xticks(x_list)
plt.yticks([i / 10 for i in range(0, 11)])  # Set y-ticks to show from 0.0 to 1.0 in intervals of 0.1
plt.ylabel('performance')
plt.grid(True)
plt.legend()

# 그래프 표시
plt.show()
