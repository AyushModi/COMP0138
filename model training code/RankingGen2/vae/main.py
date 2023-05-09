import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm, trange
from transformers import BertTokenizer

from eval import eval_vae
from trainer import VAETrainer
from models import DiscreteVAE
from torch.utils.data import DataLoader, Dataset, TensorDataset
from squad_utils import (InputFeatures, convert_examples_to_harv_features,convert_examples_to_features_answer_id,
                         read_examples,SquadExample,read_squad_examples,_improve_answer_span,_check_is_max_context)
from transformers.data.processors.squad import whitespace_tokenize
import re
import json
import csv
import pickle
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from enum import Enum

from transformers import pipeline
Rank_mode = Enum('Rank_mode', 'givenAnswer noAnswer')
Train_mode = Enum('Train_mode', 'train test')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

def loadOrginalDataset(dataFile):
  with open(dataFile) as f:
    data = json.load(f)
  return data

def re_id(data, chosen_rank_mode, chosen_train_mode):
  data2 = dict()
  data2['data'] = list()
  qid = 100000
  cid = 200000
  if chosen_train_mode == Train_mode.test:
    topic_set = data['data']
  elif chosen_train_mode == Train_mode.train:
    topic_set = np.random.choice(data['data'],size=45,replace=False)
  else:
    raise ValueError('Incorrect train mode')

  for topic in topic_set:
    data2['data'].append(dict())
    data2['data'][-1]['title'] = topic['title']
    data2['data'][-1]['paragraphs'] = list()
    for paragraph in topic['paragraphs']:
      data2['data'][-1]['paragraphs'].append(dict())
      data2['data'][-1]['paragraphs'][-1]['context'] = paragraph['context']
      data2['data'][-1]['paragraphs'][-1]['cid'] = cid
      cid += 1
      data2['data'][-1]['paragraphs'][-1]['qas'] = list()
      if chosen_rank_mode == Rank_mode.givenAnswer:
        for qa in paragraph['qas']: #i in range(5):
            data2['data'][-1]['paragraphs'][-1]['qas'].append(dict())
            data2['data'][-1]['paragraphs'][-1]['qas'][-1]['id'] = qid
            qid += 1
            data2['data'][-1]['paragraphs'][-1]['qas'][-1]['question'] = qa['question']
            data2['data'][-1]['paragraphs'][-1]['qas'][-1]['answers'] = qa['answers']
      elif chosen_rank_mode == Rank_mode.noAnswer:
        for i in range(5): # max(5, len(paragraph['qas']))):
            data2['data'][-1]['paragraphs'][-1]['qas'].append(dict())
            data2['data'][-1]['paragraphs'][-1]['qas'][-1]['id'] = qid
            qid += 1
            if i >= len(paragraph['qas']):
              blank_ans = [{"answer_start":0,"text":paragraph['context'].split()[0]}]
              data2['data'][-1]['paragraphs'][-1]['qas'][-1]['question'] = "blank"
              data2['data'][-1]['paragraphs'][-1]['qas'][-1]['answers'] = blank_ans
            else:
              qa = paragraph['qas'][i]
              data2['data'][-1]['paragraphs'][-1]['qas'][-1]['question'] = qa['question']
              data2['data'][-1]['paragraphs'][-1]['qas'][-1]['answers'] = qa['answers']
      else:
        raise ValueError('Incorrect rank_mode')
  with open(f'../ranking/{chosen_rank_mode.name}/{chosen_train_mode.name}/query_passage_data.json','w') as f:
    json.dump(data2, f, ensure_ascii=False)

def selectQueries(data, chosen_rank_mode, chosen_train_mode ,n=15):
  questions = []
  qid2cid = dict()
  contexts = []
  index = 0
  for topic in data['data']:
    for paragraph in topic['paragraphs']:
      contexts.append(paragraph['context'])
      for qa in paragraph['qas']:
        qid2cid[qa['id']] = paragraph['cid']
        if qa['question'] != "blank":
          questions.append((qa['id'],qa['question'],topic['title']))
        
  questions = np.array(questions)
  selected_indices = np.random.choice(len(questions), n, replace=False)
  selected_queries = questions[selected_indices,:]
  with open(f'../ranking/{chosen_rank_mode.name}/{chosen_train_mode.name}/selected_queries.csv', 'w', newline='') as f:
    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
    for sq in selected_queries:
      wr.writerow(sq)
  with open(f'../ranking/{chosen_rank_mode.name}/{chosen_train_mode.name}/qid2cid.txt','w') as f:
    json.dump(qid2cid, f, ensure_ascii=False)
  with open(f'../ranking/{chosen_rank_mode.name}/{chosen_train_mode.name}/contexts.txt','w') as f:
    #print(type(contexts[0]))
    f.write('\n'.join(contexts))
  return selected_queries, qid2cid, contexts

def loadQueries(chosen_rank_mode,chosen_train_mode):
  with open(f'../ranking/{chosen_rank_mode.name}/{chosen_train_mode.name}/selected_queries.csv', 'r', newline='') as f:
    reader = csv.reader(f, quoting=csv.QUOTE_ALL)
    selected_queries = list(reader)
  with open(f'../ranking/{chosen_rank_mode.name}/{chosen_train_mode.name}/qid2cid.txt','r') as f:
    qid2cid = json.load(f)
  with open(f'../ranking/{chosen_rank_mode.name}/{chosen_train_mode.name}/contexts.txt','r') as f:
    contexts = f.read().splitlines()
  return selected_queries, qid2cid, contexts

def prepData(chosen_rank_mode, chosen_train_mode):
    if chosen_train_mode == Train_mode.test:
        filepath = "../data/squad/my_test.json"
    elif chosen_train_mode == Train_mode.train:
        filepath = '../data/squad/train-v1.1.json'
    else:
        raise ValueError('Incorrect train mode')
   
    if not os.path.isfile(f'../ranking/{chosen_rank_mode.name}/{chosen_train_mode.name}/query_passage_data.json'):
        re_id(loadOrginalDataset(filepath), chosen_rank_mode, chosen_train_mode)
        data = loadOrginalDataset(f'../ranking/{chosen_rank_mode.name}/{chosen_train_mode.name}/query_passage_data.json')
        selected_queries, qid2cid, contexts = selectQueries(data, chosen_rank_mode, chosen_train_mode)
    
    selected_queries, qid2cid, contexts = loadQueries(chosen_rank_mode,chosen_train_mode) 
    return selected_queries, qid2cid, contexts

def rankContexts(genPairsForQid, contexts, mode):
  total = 0
  context_score = np.zeros((len(contexts),2))
  context_score[:,0] = -1
  # q = None
  for (cid, score_list) in genPairsForQid.items():
    context_score[cid - 200000][0] = cid
    for score_tuple in score_list:
      q_rec_loss,a_rec_loss,zq_kl_loss,za_kl_loss = score_tuple
      if mode=='all':
        scoreA = np.exp(-(q_rec_loss+a_rec_loss+zq_kl_loss+za_kl_loss))
      elif mode=='rec':
        scoreA = np.exp(-(q_rec_loss+a_rec_loss))
      context_score[cid - 200000][1] += scoreA
    
  context_score[:,1] /= np.sum(context_score[:,1])
  # len(contexts)
  top50 = context_score[context_score[:, 1].argsort()[::-1]][:50]
  return top50.tolist()

def getGeneration(qid2cid,data_loader,true_q_ids, qa_index, tokenizer, vae, pbar,generate=False):   
  genPairsForQid = dict()
  for batch in data_loader:
      questionIndex = batch[0]
      c_ids = batch[1].to(device)
      true_a_ids = batch[2].to(device)
      true_start_positions = batch[3].to(device)
      true_end_positions = batch[4].to(device)
      # print("Q: ", true_q_ids.unsqueeze(0).shape)
      # print("A: ", true_a_ids[0].unsqueeze(0).shape)
      if generate:
        with torch.no_grad():
          zq_mu, zq_logvar, zq, za_prob, za = vae.prior_encoder(c_ids)
          pred_q_ids, pred_start_positions, pred_end_positions = vae.generate(zq, za, c_ids)

      for i in range(c_ids.size(0)):
          if generate:
            pred_a_ids = torch.zeros_like(c_ids[i])
            pred_a_ids[pred_start_positions[i]: pred_end_positions[i]+1] = 1            
            loss, loss_q_rec, loss_a_rec, loss_zq_kl, loss_za_kl, loss_info = vae(c_ids[i].unsqueeze(0), true_q_ids, pred_a_ids.unsqueeze(0), pred_start_positions[i].unsqueeze(0), pred_end_positions[i].unsqueeze(0))
          else:
            loss, loss_q_rec, loss_a_rec, loss_zq_kl, loss_za_kl, loss_info = vae(c_ids[i].unsqueeze(0), true_q_ids, true_a_ids[i].unsqueeze(0), true_start_positions[i].unsqueeze(0), true_end_positions[i].unsqueeze(0))
          
        #   _c_ids = c_ids[i].cpu().tolist()
          q_rec = loss_q_rec.cpu().item()
          a_rec = loss_a_rec.cpu().item()
          zq_kl = loss_zq_kl.cpu().item()
          za_kl = loss_za_kl.cpu().item()

          cid = qid2cid[str(questionIndex[i].item())]
          if cid in genPairsForQid:
            genPairsForQid[cid].append((q_rec,a_rec,zq_kl,za_kl))
          else:
            genPairsForQid[cid] = [(q_rec,a_rec,zq_kl,za_kl)]
      pbar.update(1)
  return genPairsForQid

def loadModel(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    args.tokenizer = tokenizer

    device = torch.cuda.current_device()
    checkpoint = torch.load(args.orig_checkpoint, map_location="cpu")
    vae = DiscreteVAE(checkpoint["args"])
    vae.load_state_dict(checkpoint["state_dict"])
    vae.eval()
    vae = vae.to(device)
    
    examples = read_squad_examples(args.data_file, is_training=True, debug=False)
    return tokenizer, vae, examples

def loadDataset(args,tokenizer,examples,query_list):

    # for query in query_list:
    #   for eg in examples:
    #     eg.question_text = question_text
      features = convert_examples_to_features_answer_id(examples,
                                                    tokenizer=tokenizer,
                                                    max_seq_length=args.max_c_len,
                                                    max_query_length=args.max_q_len,
                                                    max_ans_length=args.max_q_len,
                                                    doc_stride=128,
                                                    is_training=True)
      all_index_ids = torch.tensor([f.unique_id for f in features], dtype=torch.long)
      
      all_q_ids = torch.tensor([f.q_ids for f in features], dtype=torch.long)
      # unique_qa_ids = np.unique([f.unique_id for f in features])
      #chosen_qa_ids = np.random.choice(unique_qas, size=15, replace=False)
      chosen_q_ids = []
      for query in query_list:
        index = query[0]
        index_in_all = (all_index_ids == int(index)).nonzero(as_tuple=True)[0] # Reference: https://stackoverflow.com/questions/47863001/how-pytorch-tensor-get-the-index-of-specific-value
        chosen_q_ids.append((all_q_ids[index_in_all,:],index))
      all_c_ids = torch.tensor([f.c_ids for f in features], dtype=torch.long)
      # print("C_count: ", len(features))
      # print("#egs: ", len(examples))
      all_tag_ids = torch.tensor([f.tag_ids for f in features], dtype=torch.long)
      all_a_ids = (all_tag_ids != 0).long()
      all_start_positions = torch.tensor([f.noq_start_position for f in features], dtype=torch.long)
      all_end_positions = torch.tensor([f.noq_end_position for f in features], dtype=torch.long)
      data = TensorDataset(all_index_ids,all_c_ids,all_a_ids, all_start_positions, all_end_positions)
      data_loader = DataLoader(data, shuffle=False, batch_size=args.batch_size)

      return data_loader, chosen_q_ids

def genRankings(chosen_rank_mode, chosen_train_mode):
    args.data_file = f'../ranking/{chosen_rank_mode.name}/{chosen_train_mode.name}/query_passage_data.json'
    selected_queries, qid2cid, contexts = prepData(chosen_rank_mode, chosen_train_mode)
    tokenizer, vae, examples = loadModel(args)
    data_loader, chosen_q_ids = loadDataset(args,tokenizer,examples,selected_queries)
    chosen_q_ids_left = [qidInfo for qidInfo in chosen_q_ids if not os.path.isfile(f'../ranking/{chosen_rank_mode.name}/{chosen_train_mode.name}/rec/{qidInfo[1]}.pickle')]
    if len(chosen_q_ids) == 0:
       return
    with tqdm(total=len(data_loader) * (len(chosen_q_ids_left))) as pbar:
        for i, (true_q_ids, qa_index) in enumerate(chosen_q_ids_left):
            print(f"Index {i+1}. qid: {qa_index}")
            true_q_ids = true_q_ids.to(device)
            q_text = tokenizer.decode(true_q_ids[0], skip_special_tokens=True)

            output_dict = dict()
            output_dict['question'] = q_text
            output_dict['qid'] = qa_index
            output_dict['true_cid'] = qid2cid[qa_index]

            genPairsForQid = getGeneration(qid2cid,data_loader,true_q_ids, qa_index, tokenizer, vae,pbar,generate=(chosen_rank_mode==Rank_mode.noAnswer))

            output_dict['top50docs'] = rankContexts(genPairsForQid,contexts,mode='all')
            with open(f'../ranking/{chosen_rank_mode.name}/{chosen_train_mode.name}/all/{qa_index}.pickle', 'wb') as f:
                pickle.dump(output_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

            output_dict['top50docs'] = rankContexts(genPairsForQid,contexts,mode='rec')
            with open(f'../ranking/{chosen_rank_mode.name}/{chosen_train_mode.name}/rec/{qa_index}.pickle', 'wb') as f:
                pickle.dump(output_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    

def mergeRanks(chosen_rank_mode, chosen_train_mode):
    with open(f'../ranking/{chosen_rank_mode.name}/{chosen_train_mode.name}/selected_queries.csv', 'r', newline='') as f:
        reader = csv.reader(f, quoting=csv.QUOTE_ALL)
        selected_queries = list(reader)
    with open(f'../ranking/{chosen_rank_mode.name}/{chosen_train_mode.name}/qid2cid.txt','r') as f:
        qid2cid = json.load(f)

    for score_mode in ['rec', 'all']:
        final_dict = dict()
        final_dict['qid2cid'] = qid2cid
        final_dict['queries'] = list()
        for qidInfo in selected_queries:
            qid = qidInfo[0]
            with open(f'../ranking/{chosen_rank_mode.name}/{chosen_train_mode.name}/{score_mode}/{qid}.pickle', 'rb') as h:
                qidTop50Info = pickle.load(h)
            final_dict['queries'].append(qidTop50Info)
        with open(f'../ranking/{chosen_rank_mode.name}/{chosen_train_mode.name}/{score_mode}/ranking_output.json','w') as f:
            json.dump(final_dict,f)

def addBaseline(rank_mode,score_mode,train_mode):
    with open(f'../ranking/{rank_mode}/{train_mode}/query_passage_data.json') as f:
      data = json.load(f)
    with open(f'../ranking/{rank_mode}/{train_mode}/{score_mode}/ranking_output.json') as f:
      ranking_output = json.load(f)

    cid2context = dict()
    cid2topic = dict()
    if rank_mode == 'givenAnswer':
      cid2answers = dict()
    queries = []
    qids = []
    ranks = dict()
    for topic in data['data']:
      for paragraph in topic['paragraphs']:
        if paragraph['qas'] == []:
          continue
        cid = paragraph['cid']
        cid2context[cid] = paragraph['context']
        cid2topic[cid] = topic['title']    
        if rank_mode == 'givenAnswer':
            cid2answers[cid] = [qa['answers'][0]['text'] for qa in paragraph['qas']]
    for queryInfo in ranking_output['queries']:
      queries.append(queryInfo['question'])
      qids.append(queryInfo['qid'])
      ranks[queryInfo['qid']] = dict()
      ranks[queryInfo['qid']]['model_output'] = [docInfo[0] for docInfo in queryInfo['top50docs']]
      ranks[queryInfo['qid']]['true_cid'] = queryInfo['true_cid']

    tfidfVectorizer = TfidfVectorizer()
    context_tfidfs = tfidfVectorizer.fit_transform(cid2context.values())
    if rank_mode == 'noAnswer':
      query_tfids = tfidfVectorizer.transform(queries)
      cosine_similarities = linear_kernel(query_tfids, context_tfidfs)
      cos_sim_list = cosine_similarities.tolist()
    else:
      # print(context_tfidfs.shape)
      cos_sim_list = np.zeros((len(queries),len(cid2context)))
      # print(len(queries))
      # print(len(cid2context))
      for i in range(len(cid2context)):
        new_queries = [query + " " + " ".join(cid2answers[cid]) for query in queries]
        query_tfids = tfidfVectorizer.transform(new_queries)
        # print(context_tfidfs[i].shape)
        cosine_similarities_sub = linear_kernel(query_tfids, context_tfidfs[i])
        # print("done")
        cos_sim_list[:,i] = cosine_similarities_sub.flatten()
      cos_sim_list = cos_sim_list.tolist()
    for i, query_scores in enumerate(cos_sim_list):
      #Reference: https://stackoverflow.com/questions/7851077
      query_scores_sorted = sorted(range(len(cid2context.keys())), key=lambda k: -query_scores[k])
      
      top50cids = np.array(list(cid2context.keys()))[query_scores_sorted[:50]]
      ranks[qids[i]]['tfidf_output'] = top50cids.tolist()

    for queryInfo in ranking_output['queries']:
      qid = queryInfo['qid']
      queryInfo['tfidf_top50'] = [[cid] for cid in ranks[qid]['tfidf_output']]
    with open(f'../ranking/{rank_mode}/{train_mode}/{score_mode}/ranking_output2.json','w') as f:
      json.dump(ranking_output,f)

def generateSyntheticRelevancy(question_answering, rank_mode,score_mode,train_mode):
    with open(f'../ranking/{rank_mode}/{train_mode}/query_passage_data.json') as f:
      data = json.load(f)
    with open(f'../ranking/{rank_mode}/{train_mode}/{score_mode}/ranking_output2.json') as f:
      ranking_output = json.load(f)

    numContexts = 0
    for topic in data['data']:
      for paragraph in topic['paragraphs']:
        numContexts += 1

    questionList = []
    for queryInfo in ranking_output['queries']:
      questionList.append((queryInfo['qid'],queryInfo['question']))
    print(questionList)
    # inputList = []
    with tqdm(total=numContexts*len(ranking_output['queries'])) as pbar:
      for queryInfo in ranking_output['queries']:
        qid = queryInfo['qid']
        question = queryInfo['question']
        for topic in data['data']:
          for paragraph in topic['paragraphs']:
            context = paragraph['context']
            cid = paragraph['cid']
            if not 'relevant_cids' in queryInfo:
              queryInfo['relevant_cids'] = []
            answers = [ansInfo['text'] for qa in paragraph['qas'] for ansInfo in qa['answers']]
            if answers == []:
              pbar.update(1)
              continue
              
            result = question_answering(question=question, context=context)
            if result['answer'] in answers or queryInfo['true_cid']==cid:
              queryInfo['relevant_cids'].append(cid)
            pbar.update(1)

    with open(f'../ranking/{rank_mode}/{train_mode}/{score_mode}/ranking_output3.json','w') as f:
        json.dump(ranking_output, f)

def main(args):
    print("MODEL DIR: " + args.model_dir)
    print("\nGENERATING RANKS\n")
    for chosen_rank_mode in Rank_mode:
        for chosen_train_mode in Train_mode:
            genRankings(chosen_rank_mode, chosen_train_mode)
    print("\nMERGING RANKS\n")
    for chosen_rank_mode in Rank_mode:
        for chosen_train_mode in Train_mode:
            mergeRanks(chosen_rank_mode, chosen_train_mode)
    print("\nGenerating tf-idf baseline\n")
    for score_mode in ['rec','all']:
        for chosen_rank_mode in Rank_mode:
            for chosen_train_mode in Train_mode:
                addBaseline(chosen_rank_mode.name,score_mode,chosen_train_mode.name)
    print("\nGenerating synthetic relevancy\n")
    question_answering = pipeline("question-answering",device= torch.cuda.current_device(),model="vasudevgupta/bigbird-roberta-natural-questions", handle_impossible_answer=True)
    for score_mode in ['rec','all']:
        for chosen_rank_mode in Rank_mode:
            for chosen_train_mode in Train_mode:
                generateSyntheticRelevancy(question_answering,chosen_rank_mode.name,score_mode,chosen_train_mode.name)
    print("DONE")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1004, type=int)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--train_dir', default='../data/squad/train-v1.1.json')
    parser.add_argument('--dev_dir', default='../data/squad/my_dev.json')
    parser.add_argument("--orig_checkpoint", default="../save/vae-checkpoint/original_model.pt", type=str, help="checkpoint for vae model")
    
    parser.add_argument("--max_c_len", default=384, type=int, help="max context length")
    parser.add_argument("--max_q_len", default=64, type=int, help="max query length")

    parser.add_argument("--model_dir", default="../save/vae-checkpoint", type=str)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--lr", default=1e-3, type=float, help="lr")
    parser.add_argument("--batch_size", default=64, type=int, help="batch_size")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay")
    parser.add_argument("--clip", default=5.0, type=float, help="max grad norm")

    parser.add_argument("--bert_model", default='bert-base-uncased', type=str)
    parser.add_argument('--enc_nhidden', type=int, default=300)
    parser.add_argument('--enc_nlayers', type=int, default=1)
    parser.add_argument('--enc_dropout', type=float, default=0.2)
    parser.add_argument('--dec_a_nhidden', type=int, default=300)
    parser.add_argument('--dec_a_nlayers', type=int, default=1)
    parser.add_argument('--dec_a_dropout', type=float, default=0.2)
    parser.add_argument('--dec_q_nhidden', type=int, default=900)
    parser.add_argument('--dec_q_nlayers', type=int, default=2)
    parser.add_argument('--dec_q_dropout', type=float, default=0.3)
    parser.add_argument('--nzqdim', type=int, default=50)
    parser.add_argument('--nza', type=int, default=20)
    parser.add_argument('--nzadim', type=int, default=10)
    parser.add_argument('--lambda_kl', type=float, default=0.1)
    parser.add_argument('--lambda_info', type=float, default=1.0)

    args = parser.parse_args()

    if args.debug:
        args.model_dir = "./dummy"
    # set model dir
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    args.model_dir = os.path.abspath(model_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)
