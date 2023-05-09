import collections
import json
import os

import torch
from transformers import BertTokenizer
from tqdm import tqdm

from qgevalcap.eval import eval_qg
from squad_utils3D import evaluate, write_predictions
from utils3D import batch_to_device

def to_string(index, tokenizer):
    tok_tokens = tokenizer.convert_ids_to_tokens(index)
    tok_text = " ".join(tok_tokens)

    # De-tokenize WordPieces that have been split off.
    tok_text = tok_text.replace("[PAD]", "")
    tok_text = tok_text.replace("[SEP]", "")
    tok_text = tok_text.replace("[CLS]", "")
    tok_text = tok_text.replace(" ##", "")
    tok_text = tok_text.replace("##", "")

    # Clean whitespace
    tok_text = tok_text.strip()
    tok_text = " ".join(tok_text.split())
    return tok_text

class Result(object):
    def __init__(self,
                 context,
                 real_question,
                 posterior_question,
                 prior_question,
                 real_distractor1,
                 posterior_distractor1,
                 prior_distractor1,
                 real_distractor2,
                 posterior_distractor2,
                 prior_distractor2,
                 real_distractor3,
                 posterior_distractor3,
                 prior_distractor3,
                 real_answer,
                 posterior_answer,
                 prior_answer,
                 posterior_zq_prob,
                 prior_zq_prob,
                 posterior_zd1_prob,
                 prior_zd1_prob,
                 posterior_zd2_prob,
                 prior_zd2_prob,
                 posterior_zd3_prob,
                 prior_zd3_prob):
        self.context = context
        self.real_question = real_question
        self.posterior_question = posterior_question
        self.prior_question = prior_question
        self.real_distractor1 = real_distractor1
        self.posterior_distractor1 = posterior_distractor1
        self.prior_distractor1 = prior_distractor1
        self.real_distractor2 = real_distractor2
        self.posterior_distractor2 = posterior_distractor2
        self.prior_distractor2 = prior_distractor2
        self.real_distractor3 = real_distractor3
        self.posterior_distractor3 = posterior_distractor3
        self.prior_distractor3 = prior_distractor3
        self.real_answer = real_answer
        self.posterior_answer = posterior_answer
        self.prior_answer = prior_answer
        self.posterior_zq_prob = posterior_zq_prob
        self.prior_zq_prob = prior_zq_prob
        self.posterior_zd1_prob = posterior_zd1_prob
        self.prior_zd1_prob = prior_zd1_prob
        self.posterior_zd2_prob = posterior_zd2_prob
        self.prior_zd2_prob = prior_zd2_prob
        self.posterior_zd3_prob = posterior_zd3_prob
        self.prior_zd3_prob = prior_zd3_prob


def eval_vae(epoch, args, trainer, eval_data):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits"])

    eval_loader, eval_examples, eval_features = eval_data

    all_results = []
    qa_results = []
    qg_results = {}
    res_dict = {}
    qd1_results = {}
    res_d1_dict = {}
    qd2_results = {}
    res_d2_dict = {}
    qd3_results = {}
    res_d3_dict = {}
    example_index = -1

    for batch in tqdm(eval_loader, desc="Eval iter", leave=False, position=4):
        c_ids, q_ids, a_ids, start, end, d1_ids,d2_ids,d3_ids = batch_to_device(batch, args.device)
        batch_size = c_ids.size(0)
        batch_c_ids = c_ids.cpu().tolist()
        batch_q_ids = q_ids.cpu().tolist()
        batch_d1_ids = d1_ids.cpu().tolist()
        batch_d2_ids = d2_ids.cpu().tolist()
        batch_d3_ids = d3_ids.cpu().tolist()
        batch_start = start.cpu().tolist()
        batch_end = end.cpu().tolist()

        batch_posterior_q_ids, \
        batch_posterior_start, batch_posterior_end, posterior_zq_prob, \
            batch_posterior_d1_ids,batch_posterior_d2_ids,batch_posterior_d3_ids,\
                posterior_zd1_prob, posterior_zd2_prob, posterior_zd3_prob = trainer.generate_posterior(c_ids, q_ids, a_ids,d1_ids,d2_ids,d3_ids)

        batch_start_logits, batch_end_logits \
        = trainer.generate_answer_logits(c_ids, q_ids, a_ids, d1_ids,d2_ids,d3_ids)

        batch_posterior_q_ids = batch_posterior_q_ids.cpu().tolist()
        batch_posterior_d1_ids = batch_posterior_d1_ids.cpu().tolist()
        batch_posterior_d2_ids = batch_posterior_d2_ids.cpu().tolist()
        batch_posterior_d3_ids = batch_posterior_d3_ids.cpu().tolist()
        batch_posterior_start = batch_posterior_start.cpu().tolist()
        batch_posterior_end = batch_posterior_end.cpu().tolist()
        posterior_zq_prob = posterior_zq_prob.cpu()
        posterior_zd1_prob = posterior_zd1_prob.cpu()
        posterior_zd2_prob = posterior_zd2_prob.cpu()
        posterior_zd3_prob = posterior_zd3_prob.cpu()

        batch_prior_q_ids, \
        batch_prior_start, batch_prior_end, prior_zq_prob, \
            batch_prior_d1_ids, batch_prior_d2_ids, batch_prior_d3_ids, \
                prior_zd1_prob, prior_zd2_prob, prior_zd3_prob = trainer.generate_prior(c_ids)


        batch_prior_q_ids = batch_prior_q_ids.cpu().tolist()
        batch_prior_d1_ids = batch_prior_d1_ids.cpu().tolist()
        batch_prior_d2_ids = batch_prior_d2_ids.cpu().tolist()
        batch_prior_d3_ids = batch_prior_d3_ids.cpu().tolist()
        batch_prior_start = batch_prior_start.cpu().tolist()
        batch_prior_end = batch_prior_end.cpu().tolist()
        prior_zq_prob = prior_zq_prob.cpu()
        prior_zd1_prob = prior_zd1_prob.cpu()
        prior_zd2_prob = prior_zd2_prob.cpu()
        prior_zd3_prob = prior_zd3_prob.cpu()

        for i in range(batch_size):
            example_index += 1
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index]
            unique_id = int(eval_feature.unique_id)

            context = to_string(batch_c_ids[i], tokenizer)

            real_question = to_string(batch_q_ids[i], tokenizer)
            posterior_question = to_string(batch_posterior_q_ids[i], tokenizer)
            prior_question = to_string(batch_prior_q_ids[i], tokenizer)

            real_distractor1 = to_string(batch_d1_ids[i], tokenizer)
            posterior_distractor1 = to_string(batch_posterior_d1_ids[i], tokenizer)
            prior_distractor1 = to_string(batch_prior_d1_ids[i], tokenizer)

            real_distractor2 = to_string(batch_d2_ids[i], tokenizer)
            posterior_distractor2 = to_string(batch_posterior_d2_ids[i], tokenizer)
            prior_distractor2 = to_string(batch_prior_d2_ids[i], tokenizer)

            real_distractor3 = to_string(batch_d3_ids[i], tokenizer)
            posterior_distractor3 = to_string(batch_posterior_d3_ids[i], tokenizer)
            prior_distractor3 = to_string(batch_prior_d3_ids[i], tokenizer)

            real_answer = to_string(batch_c_ids[i][batch_start[i]:(batch_end[i] + 1)], tokenizer)
            posterior_answer = to_string(batch_c_ids[i][batch_posterior_start[i]:(batch_posterior_end[i] + 1)], tokenizer)
            prior_answer = to_string(batch_c_ids[i][batch_prior_start[i]:(batch_prior_end[i] + 1)], tokenizer)

            all_results.append(Result(context=context,
                                      real_question=real_question,
                                      posterior_question=posterior_question,
                                      prior_question=prior_question,
                                      real_distractor1=real_distractor1,
                                      posterior_distractor1=posterior_distractor1,
                                      prior_distractor1=prior_distractor1,
                                      real_distractor2=real_distractor2,
                                      posterior_distractor2=posterior_distractor2,
                                      prior_distractor2=prior_distractor2,
                                      real_distractor3=real_distractor3,
                                      posterior_distractor3=posterior_distractor3,
                                      prior_distractor3=prior_distractor3,
                                      real_answer=real_answer,
                                      posterior_answer=posterior_answer,
                                      prior_answer=prior_answer,
                                      posterior_zq_prob=posterior_zq_prob[i],
                                      prior_zq_prob=prior_zq_prob[i],
                                      posterior_zd1_prob=posterior_zd1_prob[i],
                                      prior_zd1_prob=prior_zd1_prob[i],
                                      posterior_zd2_prob=posterior_zd2_prob[i],
                                      prior_zd2_prob=prior_zd2_prob[i],
                                      posterior_zd3_prob=posterior_zd3_prob[i],
                                      prior_zd3_prob=prior_zd3_prob[i])
                                      )

            qg_results[unique_id] = posterior_question
            res_dict[unique_id] = real_question
            qd1_results[unique_id] = posterior_distractor1
            res_d1_dict[unique_id] = real_distractor1
            qd2_results[unique_id] = posterior_distractor2
            res_d2_dict[unique_id] = real_distractor2
            qd3_results[unique_id] = posterior_distractor3
            res_d3_dict[unique_id] = real_distractor3
            qa_results.append(RawResult(unique_id=unique_id,
                                        start_logits=start_logits,
                                        end_logits=end_logits))

    output_prediction_file = os.path.join(args.model_dir, "pred.json")
    write_predictions(eval_examples, eval_features, qa_results, n_best_size=20,
                      max_answer_length=30, do_lower_case=True,
                      output_prediction_file=output_prediction_file,
                      verbose_logging=False,
                      version_2_with_negative=False,
                      null_score_diff_threshold=0,
                      noq_position=True)

    with open(args.dev_dir) as f:
        dataset_json = json.load(f)
        dataset = dataset_json["data"]
    with open(os.path.join(args.model_dir, "pred.json")) as prediction_file:
        predictions = json.load(prediction_file)
    ret = evaluate(dataset, predictions)
    bleu = eval_qg(res_dict, qg_results)
    bleu_d1 = eval_qg(res_d1_dict, qd1_results)
    bleu_d2 = eval_qg(res_d2_dict, qd2_results)
    bleu_d3 = eval_qg(res_d3_dict, qd3_results)
    bleu_d = (bleu_d1+bleu_d2+bleu_d3)/3
    return ret, bleu, all_results, bleu_d
