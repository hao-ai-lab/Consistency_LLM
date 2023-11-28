import torch
from transformers import Trainer, Seq2SeqTrainer, TrainerCallback
from transformers.trainer_pt_utils import LabelSmoother
import wandb
from common import pad_to_2d

from torch.utils.data import DataLoader

from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

from common import sample_from_distribution, sample_method

from enum import Enum
import random
import copy 
import numpy as np

import typing

from distill_trainer import SampleSource, SAMPLE_SOURCE_MAP, KLMethod, KL_METHOD_MAP

eval_cnt = 0

class Seq2SeqDistillTrainer(Seq2SeqTrainer):
    def __init__(self,
                 teacher_model,
                 propose_num,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        args = kwargs["args"]
        self.teacher_model = teacher_model

        self.sample_source = SAMPLE_SOURCE_MAP[args.sample_source]

        print(self.tokenizer.name_or_path, self.model.name_or_path)
        self.train_step_cnt = 0

        # online related params
        self.mode = args.mode
        self.online_eval_interval = args.online_eval_interval
        self.online_update_interval = args.online_update_interval
        self.buffer = []
        self.alphas = []
        self.sample_steps = []

        self.max_new_tokens = 128

        self.kl_method = KL_METHOD_MAP[args.kl_method]
    
    def training_step(self, model, inputs):
        self.train_step_cnt += 1
        return self.consistency_training_step(model, inputs)
        #TODO: support online/offline consistency training
        #if self.mode == "offline":
        #    return self.offline_training_step(model, inputs)
        #elif self.mode == "online":
        #    return self.online_training_step(model, inputs)
        #else:
        #    raise ValueError()
    
    def offline_training_step(self, model, inputs):
        max_new_tokens = self.max_new_tokens
        student_temperature = 1.0
        teacher_temperature = 1.0

        if self.sample_source == SampleSource.MixRequest:
            student_request_ratio = 0.5
        
        if self.sample_source == SampleSource.MixToken:
            student_token_ratio = 0.5

        if self.kl_method == KLMethod.JSD:
            fwd_loss_ratio = 0.9

        sample_mix_token = False
        # sample token ids
        if self.sample_source == SampleSource.Teacher:
            sample_student = False
        elif self.sample_source == SampleSource.Student:
            sample_student = True
        elif self.sample_source == SampleSource.MixRequest:
            sample_student = True if random.random() < student_request_ratio else False
        elif self.sample_source == SampleSource.MixToken:
            sample_mix_token = True

        # sample tokens
        if sample_mix_token:
            generated_ids = self.get_mix_generated_ids(
                model,
                self.teacher_model,
                self.tokenizer,
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs['decoder_input_ids'],
                max_new_tokens,
                student_token_ratio
            )
        elif sample_student:
            generated_ids, _ = self.get_generated_ids(
                model,
                self.tokenizer,
                inputs["input_ids"],
                inputs["attention_mask"],
                max_new_tokens,
                False,
            )
        else:
            generated_ids, _ = self.get_generated_ids(
                self.teacher_model,
                self.tokenizer,
                inputs["input_ids"],
                inputs["attention_mask"],
                max_new_tokens,
                False,
            )
        generated_ids = generated_ids.clone().detach()
        
        # prepare attention_mask and output_mask
        bsz, total_seq_len = generated_ids.shape
        prompt_len = inputs["input_ids"].shape[-1]

        attention_mask = inputs["attention_mask"]
        output_mask = generated_ids[..., 1:] == self.tokenizer.pad_token_id
        
        input_ids = inputs["input_ids"]
        # get student/teacher logits
        student_logits = self.get_logits(model, input_ids, attention_mask, generated_ids)
        with torch.no_grad():
                teacher_logits = self.get_logits(self.teacher_model, input_ids, attention_mask, generated_ids)
        student_logits = student_logits[..., :-1, :].float()
        teacher_logits = teacher_logits[..., :-1, :].float()
        
        # calculate loss
        if self.kl_method == KLMethod.Forward:
            loss = self.soft_cross_entropy(
                student_logits / student_temperature,
                teacher_logits / teacher_temperature,
                output_mask
            )
        elif self.kl_method == KLMethod.Reverse:
            loss = self.get_kl(
                teacher_logits / teacher_temperature,
                student_logits / student_temperature,
                output_mask
            )
        elif self.kl_method == KLMethod.JSD:
            reverse_loss = self.get_kl(
                teacher_logits / teacher_temperature,
                student_logits / student_temperature,
                output_mask
            )
            fwd_loss = self.get_kl(
                student_logits / student_temperature,
                teacher_logits / teacher_temperature,
                output_mask
            )
            loss = fwd_loss_ratio * fwd_loss + \
                (1 - fwd_loss_ratio) * reverse_loss

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()
        return loss.detach()

    def online_training_step(self, model, inputs):
        max_new_tokens = self.max_new_tokens
        bsz = inputs["input_ids"].shape[0]
        assert (
            bsz == 1
        ), f"Does not support batch size > 1 in online setting, input batch size: {bsz}"
        assert (
            self.args.gradient_accumulation_steps == 1
        ), f"Does not support grad_acc > 1 in online setting, grad_acc: {self.args.gradient_accumulation_steps}"

        student_temperature = 1.0
        teacher_temperature = 1.0

        if self.sample_source == SampleSource.MixRequest:
            student_request_ratio = 0.5
        
        if self.sample_source == SampleSource.MixToken:
            student_token_ratio = 0.5

        if self.kl_method == KLMethod.JSD:
            fwd_loss_ratio = 0.9

        sample_mix_token = False
        # sample token ids
        if self.sample_source == SampleSource.Teacher:
            sample_student = False
        elif self.sample_source == SampleSource.Student:
            sample_student = True
        else:
            raise NotImplementedError('online distillation only support teacher or student sampling.')
        
        # remove any masking
        input_ids =  inputs["input_ids"]
        # use speculative decoding to generate tokens
        attention_mask = inputs["attention_mask"]

        # sample tokens
        if sample_mix_token:
            generated_ids = self.get_mix_generated_ids(
                model,
                self.tokenizer,
                input_ids,
                attention_mask,
                inputs['decoder_input_ids'],
                max_new_tokens,
                student_token_ratio
            )
        elif sample_student:
            generated_ids, _ = self.get_generated_ids(
                model,
                self.tokenizer,
                input_ids,
                attention_mask,
                max_new_tokens,
                False,
            )
        else:
            generated_ids, _ = self.get_generated_ids(
                self.teacher_model,
                self.tokenizer,
                input_ids,
                attention_mask,
                max_new_tokens,
                False,
            )
        generated_ids = generated_ids.clone().detach()
        
        token_ids = generated_ids.clone().detach()
        token_ids = torch.cat([torch.zeros(1,1).long().cuda(), token_ids], dim=-1)

        self.buffer.append((token_ids, input_ids))

        if self.train_step_cnt % self.online_eval_interval == 0:
            window_size = 1

        if len(self.buffer) >= self.online_update_interval:
            self.model.train()  # switch back to training mode

            input_ids = pad_to_2d([x[1] for x in self.buffer], 0)
            # mix-token not yet supported 
            decoder_input_ids = pad_to_2d([x[0] for x in self.buffer], 0, 512)

            student_logits = self.get_logits(
                model, input_ids, attention_mask, decoder_input_ids
            )
            # generate teacher logits as the label
            # TODO: we can avoid this forward by getting logits during speculative decoding
            with torch.no_grad():
                teacher_logits = self.get_logits(
                    self.teacher_model, input_ids, attention_mask, decoder_input_ids
                )


            # only compute loss at wrong predictions
            if self.args.all_token_mask:
                mask = decoder_input_ids == 0
            else:
                raise NotImplementedError("only support loss from all tokens, except for the padded ones")
            
            loss = self.soft_cross_entropy(student_logits, teacher_logits, mask)
            loss.backward()
            self.buffer = []
            return loss.detach()
        else:
            return torch.tensor(-1)
    
    def log(self, logs):
        # Remove the 'loss' entry with value 0 before calling the superclass method
        if 'loss' in logs and logs['loss'] == -1:
            del logs['loss']
        
        # Call the original `log` method of the `Trainer` class
        super().log(logs)
    
    def consistency_training_step(self, model, inputs):
        debug=False
        max_new_tokens = 32
        #max_seq_len = 128
        student_temperature = 1.0
        teacher_temperature = 1.0

        all_losses = []    

        bsz = inputs["input_ids"].shape[0]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        initial_decode_inputs_ids = torch.full((bsz, 1), self.model.config.decoder_start_token_id).to(input_ids.device)
        decode_input_ids = initial_decode_inputs_ids
        
        #eos_reached = torch.tensor([False] * bsz, device="cuda")

        with torch.no_grad():
            jacobian_trajectory, teacher_logits = self.get_jacobian_trajectory(self.teacher_model, self.tokenizer, input_ids, attention_mask, max_new_tokens, initial_decode_inputs_ids, teacher_temperature)
        # print(f'jacobian_trajectory: {jacobian_trajectory}')
        if debug:
            print(self.tokenizer.decode(input_ids[0], skip_special_tokens=True))
            print(self.tokenizer.decode(jacobian_trajectory[-1][0], skip_special_tokens=True))
        
        for i in range(len(jacobian_trajectory)-1, -1, -2):
            logits_i = self.get_logits(model, input_ids, attention_mask, jacobian_trajectory[i].clone())
            # with torch.no_grad():
            #     logits_ii = self.get_logits(model, input_ids, attention_mask, jacobian_trajectory[i])  
            output_mask = jacobian_trajectory[i][..., 1:] == self.tokenizer.pad_token_id #it is used to mask pad_token because we do not intend to calculate the cross entrophy loss w.r.t pad
            loss = self.soft_cross_entropy(
                logits_i[..., :-1, :].float() / student_temperature,
                teacher_logits[..., :-1, :].float() / teacher_temperature,
                output_mask[..., 1:, :]
            )
            loss.backward()
            all_losses.append(loss)

        print(f'max loss: {max(all_losses)}')
        print(f'min loss: {min(all_losses)}')
        print(f'avg loss: {sum(all_losses)/len(all_losses)}')
        return max(all_losses).detach()
    
    @torch.inference_mode()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        generated_ids, _ = self.get_generated_ids(
                self.teacher_model,
                self.tokenizer,
                inputs["input_ids"],
                inputs["attention_mask"],
                self.max_new_tokens,
                False,
            )
        find = False
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, Seq2SeqDistillTrainerCallback):
                #print(f"Answer: {tokenizer.decode(generated_ids, skip_special_tokens=True)}")
                find = True
        assert find

        return None, None, None
    
    ###################### Helper Functions #############################
    @torch.inference_mode()
    def get_jacobian_trajectory(
        self,
        model,
        tokenizer,
        input_ids,
        attention_mask,
        max_new_tokens,
        decoder_input_ids,
        temperature
    ):
        bsz = decoder_input_ids.shape[0]
        decoder_input_len = [len(t) for t in decoder_input_ids]
        
        # there is one pad token at the beginning. the number of pad tokens has to be the same
        # multiple generated tokens in the preceeding generation
        assert all(x == decoder_input_len[0] for x in decoder_input_len)

        total_len = max(decoder_input_len) + max_new_tokens

        # initialize the first point of jacobian trajectory
        dummy_tokens = torch.full((bsz, max_new_tokens), tokenizer.pad_token_id, dtype=torch.long, device="cuda")
        for i in range(bsz):
            dummy_tokens[i, :] = torch.tensor(random.choices(input_ids[i][attention_mask[i]==1], k=max_new_tokens), dtype=torch.long, device="cuda")
        
        starting_tokens = torch.cat((decoder_input_ids, dummy_tokens), dim=-1)

        # begin jacobian iteration
        condition = False
        trajectory = []
        logits_trajectory = []
        itr = 0
        next_generation = starting_tokens
        trajectory.append(starting_tokens)
        while condition == False:
            itr+=1
            current_generation = next_generation
            logits = self.get_logits(model, input_ids, attention_mask, current_generation)
            logits_trajectory.append(logits)
            # TODO: optimize sampling performance
            tau = 0.001  # argmax
            distribution = torch.softmax(logits / tau, dim=-1)
            next_generation = torch.argmax(distribution, dim=-1) # starting after the pad token

            # hold prompt unchanged and update generated tokens
            for i in range(bsz):
                next_generation[i, :] = torch.cat((starting_tokens[i, :decoder_input_len[i]], next_generation[i, decoder_input_len[i]-1:-1]), dim=0)
            trajectory.append(next_generation)
            if torch.all(torch.eq(next_generation, current_generation)).item():
                condition = True
                print(f"Iteration steps: {itr}")

        return trajectory, logits_trajectory[-1] # one right-shift offset for logits trajectory to match the corresponding trajectory entry

    def soft_cross_entropy(self, predicts, targets, padding_mask):
        predict_log_prob = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        entropy = -targets_prob * predict_log_prob
        expand_mask = padding_mask.unsqueeze(-1).expand_as(entropy)
        entropy.masked_fill_(expand_mask, 0)
        mean_entropy = entropy.sum() / (~padding_mask).sum()
        return mean_entropy

    def get_kl(self, predicts, targets, padding_mask):
        kl_loss = torch.nn.KLDivLoss(reduction="none", log_target=True)
        predict_prob = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.log_softmax(targets, dim=-1)
        output = kl_loss(predict_prob, targets_prob)
        expand_mask = padding_mask.unsqueeze(-1).expand_as(output)
        output.masked_fill_(expand_mask, 0)
        mean_output = output.sum() / (~padding_mask).sum()
        return mean_output

    @torch.inference_mode()
    def get_generated_ids(
        self,
        model,
        tokenizer,
        input_ids,
        attention_mask,
        max_new_tokens,
        require_logits,
    ):
        with torch.no_grad():
            if isinstance(model, torch.nn.parallel.DistributedDataParallel) or isinstance(model, torch.nn.DataParallel):
                outputs = model.module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    output_scores=require_logits,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
            else:
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    output_scores=require_logits,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
            if require_logits:
                logits = torch.cat(
                    [score.unsqueeze(1) for score in outputs["scores"]], dim=1
                )
            else:
                logits = None
            return outputs["sequences"], logits

    @torch.inference_mode()
    def get_mix_generated_ids(
        self,
        student_model,
        teacher_model,
        tokenizer,
        input_ids,
        attention_mask,
        decoder_input_ids,
        max_new_tokens,
        mix_ratio
    ):
        bsz = input_ids.shape[0]
        for i in range(max_new_tokens):
            sample_model = student_model if random.random() < mix_ratio else teacher_model
            if isinstance(sample_model, torch.nn.parallel.DistributedDataParallel) or isinstance(sample_model, torch.nn.DataParallel):
                outputs = sample_model.module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
            )
            else:
                outputs = sample_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            decoder_input_ids = outputs["sequences"]
        return decoder_input_ids
    
    def get_logits(self, model, input_ids, attention_mask, decoder_input_ids):
        return model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
        ).logits


class Seq2SeqDistillTrainerCallback(TrainerCallback):
    def __init__(self) -> None:
        super().__init__()
        self.eval_step = 0
        self.correct_cnt = 0
        self.propose_cnt = 0

        self.alpha = 0
        self.sample_steps = 0
        self.predict_step = 0

    def on_evaluate(self, args, state, control, **kwargs):
        pass
    
    def on_predict(self, args, state, control, **kwargs):
        pass
