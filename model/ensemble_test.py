import os
import sys
import csv
import json
from collections import UserDict
from typing import Optional, Tuple

from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F

from transformers import HfArgumentParser
from transformers.generation import (
    BeamScorer,
    BeamHypotheses,
    validate_stopping_criteria,
    LogitsProcessorList,
    StoppingCriteriaList,
    BeamSearchEncoderDecoderOutput,
    BeamSearchScorer,
)

from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
from utils.data_args import DataTrainingArguments
from utils.model_args import ModelArguments
from utils.training_args import TrainingArguments

from data_loader.data_loader import Text2TraceDataModule
from model.pl_model import Text2TraceForTransformerModel
from model.evaluation import EvalForMimicProgram
from utils.eval_utils import clean_text_for_spacing, recover_condition_value

def find_best_ckpt_by_metric(output_dir, target_metric='val_ex_acc', best_method='max'):
    files = os.listdir(output_dir)  # List files
    if target_metric == 'val_ex_acc':
        scores = [float(f.split(f'{target_metric}=')[-1].replace('.ckpt', '')) for i, f in enumerate(files)]
        best_method = 'max'
    elif target_metric == 'val_loss':
        scores = [float(f.split(f'-')[1].split(f'{target_metric}=')[-1]) for i, f in enumerate(files)]
        best_method = 'min'

    # Get the index of ckpt having best metric
    if best_method == 'max':
        best_idx = scores.index(max(scores))
    elif best_method == 'min':
        best_idx = scores.index(min(scores))
    else:
        raise ValueError()
    return os.path.join(output_dir, files[best_idx])

class BeamSearchScorer(BeamScorer):
    def __init__(
        self,
        batch_size: int,
        max_length: int,
        num_beams: int,
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[bool] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
    ):
        self.max_length = max_length
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        self._is_init = False
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                max_length=self.max_length,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
            )
            for _ in range(batch_size)
        ]
        self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1, one should make use of `greedy_search` instead."
            )

        if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
            raise ValueError(
                f"`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` "
                f"has to be divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}."
            )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
        self,
        input_ids: torch.LongTensor,
        output_entropy:Optional[torch.FloatTensor],
        output_mean_entropy:Optional[torch.FloatTensor],
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor]:
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        assert batch_size == (input_ids.shape[0] // self.group_size)

        device = input_ids.device
        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)
        if output_entropy is None:
            output_entropy = torch.zeros(input_ids.shape[0], device=device)
        if output_mean_entropy is None:
            output_mean_entropy = torch.zeros(input_ids.shape[0], device=device)
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                assert (
                    len(beam_hyp) >= self.num_beams
                ), f"Batch can only be done if at least {self.num_beams} beams have been generated"
                assert (
                    eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    try:
                        beam_hyp.add(
                            input_ids[batch_beam_idx].clone(),
                            output_entropy[batch_beam_idx].clone(),
                            output_mean_entropy[batch_beam_idx].clone(),
                            next_score.item(),
                        )
                    except Exception as e:
                        print(e)
                        breakpoint()
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id: {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )

    def finalize(
        self,
        input_ids: torch.LongTensor,
        output_entropy: torch.FloatTensor,
        output_mean_entropy: torch.FloatTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        vocab_size: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.LongTensor]:
        batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                final_entropy = output_entropy[batch_beam_idx]
                final_mean_entropy = output_mean_entropy[batch_beam_idx]
                beam_hyp.add(final_tokens, final_entropy, final_mean_entropy, final_score)

        # select the best hypotheses
        best = []
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)
        best_entropy = torch.zeros((batch_size * self.num_beam_hyps_to_keep, self.max_length), device=self.device, dtype=torch.float32)
        best_mean_entropy = torch.zeros((batch_size * self.num_beam_hyps_to_keep, self.max_length), device=self.device, dtype=torch.float32)

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            beam_scores = np.array([x[0] for x in beam_hyp.beams])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                best_hyp_entropy = best_hyp_tuple[2].unsqueeze(0)
                best_hyp_mean_entropy = best_hyp_tuple[3].unsqueeze(0)
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # append to lists
                best.append(best_hyp)
                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score
                
                seq_length = best_hyp_mean_entropy.size(1)
                best_entropy[i * self.num_beam_hyps_to_keep + j, :seq_length] = best_hyp_entropy
                best_mean_entropy[i * self.num_beam_hyps_to_keep + j, :seq_length] = best_hyp_mean_entropy

        # prepare for adding eos
        sent_max_len = min(sent_lengths.max().item() + 1, self.max_length)
        best_mean_entropy = best_mean_entropy[:, :sent_max_len]
        decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < self.max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
                "sequence_entropy": best_entropy,
                "sequence_mean_entropy": best_mean_entropy,
            }
        )

class BeamHypotheses:
    def __init__(self, num_beams: int, max_length: int, length_penalty: float, early_stopping: bool):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp: torch.LongTensor, hyp_entropy, hyp_mean_entropy, sum_logprobs: float):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, hyp_entropy, hyp_mean_entropy))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _, _, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


def ensemble_beam_search(
    input_ids: torch.LongTensor,
    decoder_start_token_id: int,
    attention_mask: torch.LongTensor,
    num_beams: int,
    num_return_sequences: int,
    length_penalty: Optional[float] = None,
    early_stopping: Optional[bool] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: Optional[bool] = None,
    config = None,
    model_list = None,
    seed_gpu_dict=None,
    **model_kwargs,
):
    # init values
    batch_size = input_ids.shape[0]

    length_penalty = length_penalty if length_penalty is not None else 1.0
    early_stopping = early_stopping if early_stopping is not None else False

    if num_return_sequences > num_beams:
        raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

    # Ensure all inputs are on the same GPU
    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda() if attention_mask is not None else None

    # Expand input to match beam size
    expanded_return_idx = (
        torch.arange(batch_size).view(-1, 1).repeat(1, num_beams).view(-1).to(input_ids.device)
    )
    input_ids = input_ids.index_select(0, expanded_return_idx)
    attention_mask = attention_mask.index_select(0, expanded_return_idx) if attention_mask is not None else None

    # Initialize beam search scorer
    beam_scorer = BeamSearchScorer(
        batch_size=batch_size,
        max_length=max_length,
        num_beams=num_beams,
        device=input_ids.device,
        length_penalty=length_penalty,
        do_early_stopping=early_stopping,
        num_beam_hyps_to_keep=num_return_sequences,
    )

    # Initialize decoder input
    decoder_input_ids = torch.ones((batch_size * num_beams, 1), dtype=torch.long, device=input_ids.device)
    decoder_input_ids = decoder_input_ids * decoder_start_token_id

    # Initialize generation
    cur_len = 1
    batch_beam_size = batch_size * num_beams
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    # Prepare model input
    model_kwargs = {
        "encoder_input_ids": input_ids,
        "encoder_attention_mask": attention_mask,
    }

    with torch.no_grad():
        # First run encoder
        encoder = model_list[0].model.model.encoder
        with torch.cuda.amp.autocast():
            encoder_outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        model_kwargs["encoder_outputs"] = encoder_outputs

        while cur_len < max_length:
            # Prepare input for current step
            model_inputs = {
                "decoder_input_ids": decoder_input_ids,
                "encoder_outputs": encoder_outputs,
                "attention_mask": attention_mask,
            }

            for model_id in range(len(model_list)):
                with torch.cuda.amp.autocast():
                    outputs = model_list[model_id].model.model(
                        **model_inputs,
                        return_dict=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                    )
                single_model_logits = outputs.logits[:, -1, :].cuda()
                single_model_prob = torch.softmax(single_model_logits, dim=1)
                single_model_log_prob = F.log_softmax(single_model_logits, dim=1)
                single_model_entropy = torch.sum(single_model_prob * single_model_log_prob, dim=1) * (-1)

                if model_id == 0:
                    next_token_probs = single_model_prob.unsqueeze(0)
                    next_token_lprobs = single_model_log_prob.unsqueeze(0)
                    sum_entropy = single_model_entropy
                else:
                    next_token_probs = torch.cat([next_token_probs, single_model_prob.unsqueeze(0)])
                    next_token_lprobs = torch.cat([next_token_lprobs, single_model_log_prob.unsqueeze(0)])
                    sum_entropy = sum_entropy + single_model_entropy

            mean_entropy = sum_entropy / len(model_list)
            next_token_prob = torch.mean(next_token_probs, dim=0)
            next_token_scores = torch.logsumexp(next_token_lprobs, dim=0) - torch.log(torch.tensor(len(model_list), dtype=torch.float32, device=next_token_lprobs.device))

            # Update decoder input
            next_tokens = torch.argmax(next_token_scores, dim=-1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Update length
            cur_len = decoder_input_ids.shape[1]

            # Check if EOS token is generated
            if eos_token_id is not None:
                if (decoder_input_ids == eos_token_id).any():
                    break

        return {
            "sequences": decoder_input_ids,
            "sequence_scores": beam_scores,
            "mean_entropy": mean_entropy,
            "sequences_entropy": sum_entropy,
        }

def evaluate_step(model_list, batch, batch_idx, eval_module, seed_gpu_dict):
    results = {
        "ex_acc": [], "pred": [], "pred_tokens": [], "question": [],
        "recover_ex_acc": [], "recover_pred": [], "answer": [],
        "data_uncertainty": [], "model_uncertainty": [], "total_uncertainty": []
    }

    tokenizer = eval_module.tokenizer
    beam_size = 5
    num_samples = 5   # Number of samples per input

    questions, answers = eval_module.test_trace_gt
    
    # Safety check for batch size
    bsz, seq_len = batch['decoder_input_ids'].shape
    total_questions = len(questions)
    
    # Calculate actual batch size for the current batch
    actual_bsz = min(bsz, total_questions - batch_idx * bsz)
    if actual_bsz <= 0:
        return results
        
    seq_len = 200  # Set max sequence length
    
    sos_token_id = tokenizer.convert_tokens_to_ids('<s>')
    if tokenizer.decode(sos_token_id) == '<unk>':
        raise ValueError()

    generation_output = ensemble_beam_search(
        input_ids=batch['input_ids'][:actual_bsz].cuda(),
        decoder_start_token_id=sos_token_id,
        attention_mask=batch['attention_mask'][:actual_bsz].cuda(),
        max_length=seq_len,
        num_beams=beam_size,
        num_return_sequences=num_samples,
        output_scores=True,
        return_dict_in_generate=True,
        config=model_list[0].config,
        model_list=model_list,
        seed_gpu_dict=seed_gpu_dict,
    )

    outputs = generation_output['sequences']
    sequence_mean_entropy = generation_output.get('mean_entropy', None)
    sequence_entropy = generation_output.get('sequences_entropy', None)

    for b_idx in range(actual_bsz):
        current_idx = batch_idx * bsz + b_idx
        if current_idx >= total_questions:
            break
            
        question = questions[current_idx]
        answer = answers[current_idx]

        for sample_id in range(num_samples):
            end_token = tokenizer.eos_token_id
            pred_tensor = outputs[b_idx * num_samples + sample_id][1:]
            
            # Initialize uncertainty values as lists
            data_uncertainty = []
            total_uncertainty = []
            model_uncertainty = []

            # Process predicted sequence
            if len(torch.nonzero(pred_tensor==end_token)) > 0:
                pred_eos_idx = torch.nonzero(pred_tensor==end_token)[0].item()
                pred_tensor = pred_tensor[:pred_eos_idx+1]
            
            # Process uncertainty values
            if sequence_mean_entropy is not None:
                if isinstance(sequence_mean_entropy, torch.Tensor):
                    data_uncertainty = sequence_mean_entropy[b_idx * num_samples + sample_id].cpu().tolist()
                    data_uncertainty = [data_uncertainty] if isinstance(data_uncertainty, (float, int)) else data_uncertainty
                else:
                    data_uncertainty = [float(sequence_mean_entropy)]

            if sequence_entropy is not None:
                if isinstance(sequence_entropy, torch.Tensor):
                    total_uncertainty = sequence_entropy[b_idx * num_samples + sample_id].cpu().tolist()
                    total_uncertainty = [total_uncertainty] if isinstance(total_uncertainty, (float, int)) else total_uncertainty
                else:
                    total_uncertainty = [float(sequence_entropy)]

            # Ensure uncertainty values have matching lengths
            if data_uncertainty and total_uncertainty:
                min_len = min(len(data_uncertainty), len(total_uncertainty))
                data_uncertainty = data_uncertainty[:min_len]
                total_uncertainty = total_uncertainty[:min_len]
                model_uncertainty = [tu - du for tu, du in zip(total_uncertainty, data_uncertainty)]
            
            pred = tokenizer.decode(pred_tensor, skip_special_tokens=True)
            pred_tokens = tokenizer.convert_ids_to_tokens(pred_tensor, skip_special_tokens=True)
            pred = clean_text_for_spacing(pred)

            results["question"].append(question)
            results["answer"].append(answer)
            results["pred_tokens"].append(pred_tokens)
            results["pred"].append(pred)
            results["data_uncertainty"].append(data_uncertainty)
            results["model_uncertainty"].append(model_uncertainty)
            results["total_uncertainty"].append(total_uncertainty)

            ex_flag = eval_module._get_flag_for_execution_accuracy(
                gt='unknown', 
                pred=pred, 
                interpreter=eval_module.interpreter, 
                answer=answer
            )
            results["ex_acc"].append(ex_flag)

            pred_recover = recover_condition_value(pred, eval_module.look_up)
            results["recover_pred"].append(pred_recover)
            
            recover_ex_flag = eval_module._get_flag_for_execution_accuracy(
                gt='unknown', 
                pred=pred_recover, 
                interpreter=eval_module.interpreter, 
                answer=answer
            )
            results['recover_ex_acc'].append(recover_ex_flag)

    return results

def gather_evaluation_outputs(epoch_outputs):
    ex_flag, recover_ex_flag = [], []
    data_uncertainty, model_uncertainty, total_uncertainty = [], [], [] 
    pred, pred_tokens, recover_pred = [], [], []
    questions, answers = [], []

    for step_outputs in epoch_outputs:
        data_uncertainty += step_outputs['data_uncertainty']
        model_uncertainty += step_outputs['model_uncertainty']
        total_uncertainty += step_outputs['total_uncertainty']
        
        ex_flag += step_outputs['ex_acc']
        recover_ex_flag += step_outputs['recover_ex_acc']
            
        pred += step_outputs['pred']
        pred_tokens += step_outputs['pred_tokens']
        recover_pred += step_outputs['recover_pred']
        questions += step_outputs['question']
        answers += step_outputs['answer']

    return_dict = {
        "ex_flag": ex_flag, "pred": pred, "pred_tokens": pred_tokens, "question":questions,
        "recover_ex_flag": recover_ex_flag, "recover_pred": recover_pred, "answer":answers,
        "data_uncertainty":data_uncertainty, "model_uncertainty": model_uncertainty, "total_uncertainty": total_uncertainty
        }
    
    return return_dict

def write_decode_output_file(save_file_path, save_file):
    fieldnames = ["idx", "pred", "pred_tokens", "data_uncertainty", "ex_flag", "recover_pred", "recover_ex_flag", "question", "answer", "model_uncertainty", "total_uncertainty"]
    with open(save_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        writer.writerow({k:k for k in fieldnames}) # write the first column
        for idx in range(len(save_file['pred'])):
            row = {k:(save_file[k][idx] if k != "idx" else idx) for k in fieldnames}
            writer.writerow(row)
        csvfile.close()

def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU setup.")
    
    n_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {n_gpus}")
    for i in range(n_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Set default GPU
    torch.cuda.set_device(0)
    print(f"Using GPU 0: {torch.cuda.get_device_name(0)}")

    # Gather the arguments
    triple_args = {"data_args": data_args, "model_args": model_args, "training_args": training_args}
    
    # Define pl.DataModule and pl.LightningModule
    SEEDS = model_args.ensemble_seed.split(',')
    
    # Simplify GPU allocation
    seed_gpu_dict = {SEED: 0 for SEED in SEEDS}
    
    model_list = []
    data_module = Text2TraceDataModule(data_args, model_args, training_args)
    data_module.setup("test")
    
    for SEED in tqdm(SEEDS):
        try:
            output_dir = training_args.output_dir
            output_dir = output_dir.split('_')
            output_dir[-1] = SEED
            model_dir = '_'.join(output_dir)
            ckpt_path = find_best_ckpt_by_metric(output_dir=model_dir, target_metric='val_ex_acc')
            
            print(f"\nLoading model with seed {SEED}")
            if model_args.encoder_decoder_type == 'unilm':
                raise NotImplementedError('Ensemble beam search for UNIQA is not supported.')
            elif model_args.encoder_decoder_type == 't5':
                # Load model
                model = Text2TraceForTransformerModel.load_from_checkpoint(
                    checkpoint_path=ckpt_path, 
                    data_module=data_module,
                    **triple_args
                )
                model = model.cpu()
                model = model.cuda()
                model_list.append(model)
                print(f"Successfully loaded model with seed {SEED}")
        
        except Exception as e:
            print(f"Error loading model with seed {SEED}: {str(e)}")
            raise e

    eval_module = EvalForMimicProgram(
        data_args=data_args,
        training_args=training_args,
        model_args=model_args,
        tokenizer=data_module.tokenizer,
    )

    save_dir = os.path.join(os.getcwd(), f"saved/ensemble/decode_outputs")
    results_fname = 'ensemble_test_result_beam_5.csv'
    os.makedirs(save_dir, exist_ok=True)
    results_fpath = os.path.join(save_dir, results_fname)

    data_loader = data_module.test_dataloader()
    results = []
    
    # Show progress and error handling
    total_batches = len(data_loader)
    print(f"\nStarting evaluation with {len(model_list)} models")
    print(f"Total number of batches: {total_batches}")
    print(f"Total number of test samples: {len(eval_module.test_trace_gt[0])}")
    
    for batch_idx, batch in enumerate(tqdm(data_loader)):
        try:
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            result = evaluate_step(
                model_list=model_list, 
                batch=batch, 
                batch_idx=batch_idx, 
                eval_module=eval_module, 
                seed_gpu_dict=seed_gpu_dict
            )
            if result["question"]:
                results.append(result)
                
            # Periodically clear GPU cache
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"\nError in batch {batch_idx}: {str(e)}")
            continue
    
    print("\nGathering evaluation outputs...")
    if results:
        results = gather_evaluation_outputs(results)
        write_decode_output_file(results_fpath, results)
        print(f"\nResults saved to: {results_fpath}")

        # === Added: Compute and print overall execution accuracy and recovered execution accuracy ===
        ex_acc = results['ex_flag']
        recover_ex_acc = results['recover_ex_flag']
        ex_acc_mean = np.mean(ex_acc)
        recover_ex_acc_mean = np.mean(recover_ex_acc)
        print(f"Overall Execution Accuracy (ex acc): {ex_acc_mean:.4f}")
        print(f"Overall Recovered Execution Accuracy: {recover_ex_acc_mean:.4f}")
    else:
        print("\nNo results were generated!")
    
    mild_and_high, high = [], []
    with open(data_args.test_data_file) as f:
        for line in f:
            data = json.loads(line)
            if data['ambiguity'] == 'high':
                high.append(1)
                mild_and_high.append(1)
            elif data['ambiguity'] == "mild":
                high.append(0)
                mild_and_high.append(1)
            else:
                high.append(0)
                mild_and_high.append(0)
    
    data_uncertainty = results['data_uncertainty']
    model_uncertainty = results['model_uncertainty']
    total_uncertainty = results['total_uncertainty']
    max_data_uncertainty = [max(du) for du in data_uncertainty]
    max_model_uncertainty = [max(mu) for mu in model_uncertainty]
    max_total_uncertainty = [max(tu) for tu in total_uncertainty]

    for metric, uncertainty in zip(["DATA", "MODEL", "TOTAL"], [data_uncertainty, model_uncertainty, total_uncertainty]):
        print(metric)
        uncertainty = np.array(uncertainty, dtype=object)
        n_labels = len(mild_and_high)
        print(f"DEBUG: uncertainty shape: {uncertainty.shape}, label len: {n_labels}")

        # Empty data processing
        if uncertainty.size == 0:
            print(f"Warning: Empty uncertainty array")
            uncertainty_mean = np.zeros(n_labels)
            uncertainty_max = np.zeros(n_labels)
        # list of list status
        elif isinstance(uncertainty[0], (list, np.ndarray)) and not all(isinstance(u, (int, float, np.floating, np.integer)) for u in uncertainty):
            sample_count = min(len(uncertainty), n_labels)
            uncertainty_max = np.zeros(n_labels)
            uncertainty_mean = np.zeros(n_labels)
            for i in range(sample_count):
                if len(uncertainty[i]) > 0:
                    uncertainty_max[i] = np.max(uncertainty[i])
                    uncertainty_mean[i] = np.mean(uncertainty[i])
        else:
            # flatten
            uncertainty = np.array(uncertainty, dtype=float).flatten()
            # 1D and length is n_labels*num_samples
            if uncertainty.ndim == 1 and len(uncertainty) % n_labels == 0:
                num_samples = len(uncertainty) // n_labels
                uncertainty = uncertainty.reshape((n_labels, num_samples))
                uncertainty_mean = uncertainty.mean(axis=1)
                uncertainty_max = uncertainty.max(axis=1)
            # 2D and shape[0] is n_labels
            elif uncertainty.ndim == 2 and uncertainty.shape[0] == n_labels:
                uncertainty_mean = uncertainty.mean(axis=1)
                uncertainty_max = uncertainty.max(axis=1)
            # 1D and length is n_labels
            elif uncertainty.ndim == 1 and len(uncertainty) == n_labels:
                uncertainty_mean = uncertainty
                uncertainty_max = uncertainty
            # 2D and shape[0] is n_labels*num_samples, shape[1]==1
            elif uncertainty.ndim == 2 and uncertainty.shape[0] % n_labels == 0 and uncertainty.shape[1] == 1:
                num_samples = uncertainty.shape[0] // n_labels
                uncertainty = uncertainty.reshape((n_labels, num_samples))
                uncertainty_mean = uncertainty.mean(axis=1)
                uncertainty_max = uncertainty.max(axis=1)
            else:
                raise ValueError(f"Unexpected uncertainty shape: {uncertainty.shape}, label len: {n_labels}")

        # Final assertion
        assert len(uncertainty_mean) == n_labels, f"Mean uncertainty length {len(uncertainty_mean)} doesn't match label length {n_labels}"
        assert len(uncertainty_max) == n_labels, f"Max uncertainty length {len(uncertainty_max)} doesn't match label length {n_labels}"

        print(f"[MILD & HIGH] PR AUC (mean): {average_precision_score(mild_and_high, uncertainty_mean)}")
        print(f"[MILD & HIGH] ROC AUC (mean): {roc_auc_score(mild_and_high, uncertainty_mean)}")
        print(f"[HIGH] PR AUC (mean): {average_precision_score(high, uncertainty_mean)}")
        print(f"[HIGH] ROC AUC (mean): {roc_auc_score(high, uncertainty_mean)}")

        print(f"[MILD & HIGH] PR AUC (max): {average_precision_score(mild_and_high, uncertainty_max)}")
        print(f"[MILD & HIGH] ROC AUC (max): {roc_auc_score(mild_and_high, uncertainty_max)}")
        print(f"[HIGH] PR AUC (max): {average_precision_score(high, uncertainty_max)}")
        print(f"[HIGH] ROC AUC (max): {roc_auc_score(high, uncertainty_max)}")
        print()


if __name__ == "__main__":
    main()