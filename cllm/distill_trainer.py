import transformers
import torch
from transformers import Trainer, TrainerCallback
from transformers.trainer_pt_utils import LabelSmoother
import wandb
from common import pad_to_2d, sychronize_time
from enum import Enum
import random
from torch.utils.data import DataLoader

import numpy as np

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

import copy

class SampleSource(Enum):
    Student = 1
    Teacher = 2
    MixRequest = 3
    MixToken = 4


SAMPLE_SOURCE_MAP = {
    "student": SampleSource.Student,
    "teacher": SampleSource.Teacher,
    "mix_request": SampleSource.MixRequest,
    "mix_token": SampleSource.MixToken,
}


class KLMethod(Enum):
    Forward = 1
    Reverse = 2
    JSD = 3


KL_METHOD_MAP = {
    "forward": KLMethod.Forward,
    "reverse": KLMethod.Reverse,
    "jsd": KLMethod.JSD
}

eval_cnt = 0
copy_model = transformers.AutoModelForCausalLM.from_pretrained(
    "JackFram/llama-160m")
copy_model.cuda()


class DistillTrainer(Trainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args = kwargs["args"]
        self.teacher_model = teacher_model
        self.train_step_cnt = 0

        # online related params
        self.mode = args.mode
        self.online_eval_interval = args.online_eval_interval
        self.online_update_interval = args.online_update_interval
        self.buffer = []
        self.sample_steps = []

        self.sample_source = SAMPLE_SOURCE_MAP[args.sample_source]
        self.kl_method = KL_METHOD_MAP[args.kl_method]

        self.max_new_tokens = 128

    def training_step(self, model, inputs):
        self.train_step_cnt += 1
        return self.consistency_training_step(model, inputs)
        #if self.mode == "offline":
        #    return self.offline_training_step(model, inputs)
        #elif self.mode == "online":
        #    return self.online_training_step(model, inputs)
        #else:
        #    raise ValueError()

    def online_training_step(self, model, inputs):
        max_new_tokens = self.max_new_tokens
        bsz = inputs["input_ids"].shape[0]
        assert (
            bsz == 1
        ), f"Does not support batch size > 1 in online setting, input batch size: {bsz}"
        assert (
            self.args.gradient_accumulation_steps == 1
        ), f"Does not support grad_acc > 1 in online setting, grad_acc: {self.args.gradient_accumulation_steps}"

        # remove any masking
        input_ids = inputs["input_ids"][inputs["attention_mask"]].unsqueeze(0)
        # use speculative decoding to generate tokens
        if sample_mix_token:
            copy_model.load_state_dict(model.state_dict())
            generated_ids = self.get_mix_generated_ids(
                copy_model,
                self.teacher_model,
                self.tokenizer,
                inputs["prompt_ids"],
                inputs["prompt_attention_mask"],
                max_new_tokens,
                student_token_ratio
            )
        elif sample_student:
            copy_model.load_state_dict(model.state_dict())
            generated_ids, _ = self.get_generated_ids(
                copy_model,
                self.tokenizer,
                inputs["prompt_ids"],
                inputs["prompt_attention_mask"],
                max_new_tokens,
                False,
            )
        else:
            generated_ids, _ = self.get_generated_ids(
                self.teacher_model,
                self.tokenizer,
                inputs["prompt_ids"],
                inputs["prompt_attention_mask"],
                max_new_tokens,
                False,
            )
        generated_ids = generated_ids.clone().detach()
        

        token_ids = torch.cat([input_ids, generated_ids], dim=-1)

        if "dataset" in inputs:
            dataset = inputs["dataset"][0]
            if dataset not in self.alphas_by_dataset:
                self.alphas_by_dataset[dataset] = []
            if self.train_step_cnt <= 2000:
                if dataset == "gsm8k":
                    self.buffer.append((token_ids))
            else:
                if dataset == "finance":
                    self.buffer.append((token_ids))
        else:
            self.buffer.append((token_ids))
            
        if self.train_step_cnt % self.online_eval_interval == 0:
            window_size = 1

        if len(self.buffer) >= self.online_update_interval:
            self.model.train()  # switch back to training mode

            input_ids = pad_to_2d(
                [x[0] for x in self.buffer], self.tokenizer.pad_token_id)
            student_logits = self.get_logits(
                model, input_ids, torch.ones_like(input_ids)
            ).float()
            # generate teacher logits as the label
            # TODO: we can avoid this forward by getting logits during speculative decoding
            with torch.no_grad():
                teacher_logits = self.get_logits(
                    self.teacher_model, input_ids, torch.ones_like(input_ids)
                ).float()

            # compute loss for all generated content
            output_mask = generated_ids[..., 1:] == self.tokenizer.pad_token_id

            loss = self.soft_cross_entropy(
                student_logits, teacher_logits, output_mask)
            loss.backward()
            self.buffer = []
            return loss.detach()
        else:
            return torch.tensor(-1)

    def offline_training_step(self, model, inputs):
        max_new_tokens = self.max_new_tokens
        student_temperature = 1.0
        teacher_temperature = 1.0
        debug = False
        if debug:
            step_start_time = sychronize_time()

        if self.sample_source == SampleSource.MixRequest:
            student_request_ratio = 0.5

        if self.sample_source == SampleSource.MixToken:
            student_token_ratio = 0.5

        if self.kl_method == KLMethod.JSD:
            fwd_loss_ratio = 0.9

        sample_mix_token = False

        ############ sample tokens #############
        if self.sample_source == SampleSource.Teacher:
            sample_student = False
        elif self.sample_source == SampleSource.Student:
            sample_student = True
        elif self.sample_source == SampleSource.MixRequest:
            sample_student = True if random.random() < student_request_ratio else False
        elif self.sample_source == SampleSource.MixToken:
            sample_mix_token = True

        if debug:
            sample_time_start = sychronize_time()

        if sample_mix_token:
            copy_model.load_state_dict(model.state_dict())
            generated_ids = self.get_mix_generated_ids(
                copy_model,
                self.teacher_model,
                self.tokenizer,
                inputs["prompt_ids"],
                inputs["prompt_attention_mask"],
                max_new_tokens,
                student_token_ratio
            )
        elif sample_student:
            copy_model.load_state_dict(model.state_dict())
            generated_ids, _ = self.get_generated_ids(
                copy_model,
                self.tokenizer,
                inputs["prompt_ids"],
                inputs["prompt_attention_mask"],
                max_new_tokens,
                False,
            )
        else:
            generated_ids, _ = self.get_generated_ids(
                self.teacher_model,
                self.tokenizer,
                inputs["prompt_ids"],
                inputs["prompt_attention_mask"],
                max_new_tokens,
                False,
            )
        generated_ids = generated_ids.clone().detach()
        if debug:
            print(f"Sample time: {sychronize_time() - sample_time_start}")

        if debug:
            prepare_time_start = sychronize_time()

        ############ preparet attention_mask and output_mask ##############
        prompt_len = inputs["prompt_ids"].shape[-1]
        attention_mask = generated_ids != self.tokenizer.pad_token_id
        output_mask = generated_ids[..., 1:] == self.tokenizer.pad_token_id
        # Ignore prompt when calculating loss
        output_mask[..., :prompt_len-1] = True
        if False:
            print("\n")
            print(generated_ids[:, prompt_len:])
            print("[prompt]", self.tokenizer.batch_decode(
                inputs["prompt_ids"], skip_special_tokens=True))
            print("[student] ", self.tokenizer.batch_decode(
                generated_ids[:, prompt_len:]))
            labels = torch.where(
                inputs["labels"] == IGNORE_TOKEN_ID, self.tokenizer.unk_token_id, inputs["labels"])
            print("[teacher]", self.tokenizer.batch_decode(
                labels, skip_special_tokens=True))
            print(f"bsz: {bsz}, total_len: {total_seq_len}, gen_len: {gen_len}, ",
                      f"output_sum:{(~output_mask).sum()}, atten_mask: {attention_mask.sum()}")

        
        if debug:
            print(f"Prepare time: {sychronize_time() - prepare_time_start}")

        ############ get student/teacher logits ######################
        cal_logits_start = sychronize_time()
        student_logits = self.get_logits(model, generated_ids, attention_mask)
        with torch.no_grad():
            teacher_logits = self.get_logits(
                self.teacher_model, generated_ids, attention_mask)
        student_logits = student_logits[..., :-1, :].float()
        teacher_logits = teacher_logits[..., :-1, :].float()
        if debug:
            print(
                f"Calculate logits time: {sychronize_time() - cal_logits_start}")

        if False:
            with torch.no_grad():
                fwd_student_logits = self.get_logits(
                    model, inputs["input_ids"], inputs["attention_mask"])[..., :-1, :].float()
                fwd_teacher_logits = self.get_logits(
                    self.teacher_model, inputs["input_ids"], inputs["attention_mask"])[..., :-1, :].float()
                fwd_loss = self.soft_cross_entropy(
                    fwd_student_logits / student_temperature,
                    fwd_teacher_logits / teacher_temperature,
                    inputs["labels"][..., 1:] == IGNORE_TOKEN_ID
                )
                print("teacher-sample-loss", fwd_loss)

        ################### calculate loss ##############
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

        if debug:
            compute_loss_time_start = sychronize_time()
        loss.backward()
        if debug:
            print(
                f"Backward time: {sychronize_time() - compute_loss_time_start}")
            print(f"Total step time: {sychronize_time() - step_start_time}")
        return loss.detach()

    def consistency_training_step(self, model, inputs):

        max_new_tokens = 15
        max_new_seq_len = 128
        student_temperature = 1.0
        teacher_temperature = 1.0        

        itr = 0
        ### the following is use jacobian generate per max_new_tokens, max_seq_len is initialized
        while True:
            all_losses = []
            print(itr)
            if itr == 0:
                input_ids = inputs["input_ids"]
                input_masks = inputs["attention_mask"]
            else:
                input_masks = torch.ones_like(input_ids).to(input_ids.device)
            
            bsz = input_ids.shape[0]
            eos_reached = torch.tensor([False] * bsz, device="cuda")
            jacobian_trajectory, teacher_logits = self.get_jacobian_trajectory(self.teacher_model, self.tokenizer, input_ids, input_masks, max_new_tokens)

            ### tokens generated after <eos> are set to <pad>
            for i in range(len(jacobian_trajectory)):
                for j in range(bsz):
                    prompt_len = torch.sum(input_masks)
                    eos_positions = torch.where(jacobian_trajectory[i][j, :prompt_len+i]==self.tokenizer.eos_token_id)[0]
                    if len(eos_positions)==0:
                        # no EOS, continue to the next item in the batch
                        continue
                    # otherwise, set tokens coming after EOS as pad 
                    eos_reached[j] = True
                    trajectory_copy = jacobian_trajectory[i].clone().detach()
                    eos_pos = eos_positions[0]
                    trajectory_copy[j, int(eos_pos)+1:] = self.tokenizer.pad_token_id
                    jacobian_trajectory[i] = trajectory_copy

            ########## use cosistency loss to train ##########
            attention_mask = torch.full_like(jacobian_trajectory[0], 1).to(input_ids.device)
            for i in range(len(jacobian_trajectory)-2, -1, -1):
                
                # get attention mask and get logits
                attention_mask = jacobian_trajectory[i] != self.tokenizer.pad_token_id
                logits_i = self.get_logits(model, jacobian_trajectory[i].clone(), attention_mask)
                # print(f'logits_{i}: {logits_i}')

                # ignore pad_token and prompt_token because we do not intend to calculate the cross entrophy loss w.r.t pad & prompt
                output_mask = jacobian_trajectory[i][..., 1:] == self.tokenizer.pad_token_id 
                output_mask[torch.where(input_masks) == 1] = True       

                loss = self.soft_cross_entropy(
                    logits_i[..., :-1, :].float() / student_temperature, # logits generated by the last token is dropped
                    teacher_logits[..., :-1, :].float() / teacher_temperature,
                    output_mask
                )
                loss.backward()
                all_losses.append(loss)
            
            ### see if next max_new_tokens should be generated & if True, update weights and prepare new input_ids
            print(f'max loss: {max(all_losses).detach()}')
            print(f'min loss: {min(all_losses).detach()}')
            print(f'avg loss: {sum(all_losses)/len(all_losses)}')

            itr+=1      
            if all(eos_reached) or itr*max_new_tokens >= max_new_seq_len:
                break
            self.optimizer.step()
            input_ids = jacobian_trajectory[-1][torch.where(eos_reached==False)[0].tolist(), ...] # delete samples with <eos> generated

        return max(all_losses).detach()
    
    def log(self, logs):
        # Remove the 'loss' entry with value 0 before calling the superclass method
        if 'loss' in logs and logs['loss'] == -1:
            del logs['loss']

        # Call the original `log` method of the `Trainer` class
        super().log(logs)

    def get_train_dataloader(self):
        # Create custom DataLoader with shuffle set to False
        shuffle = False if self.mode == "online" else True
        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "shuffle": shuffle,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        return self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

    @torch.inference_mode()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if self.args.local_rank == 0:
            generated_ids, _ = self.get_generated_ids(
                self.teacher_model,
                self.tokenizer,
                inputs["prompt_ids"],
                inputs["prompt_attention_mask"],
                self.max_new_tokens,
                False,
            )
            find = False
            for callback in self.callback_handler.callbacks:
                if isinstance(callback, DistillTrainerCallback):
                    #print(f"Answer: {tokenizer.decode(generated_ids, skip_special_tokens=True)}")
                    find = True
            assert find

        return None, None, None

    # def train(self, resume_from_checkpoint=None):
    #     if self.mode == "offline":
    #         # Evaluate the model before training
    #         self.evaluate()

    #     # Now start the actual training
    #     super().train(resume_from_checkpoint)

    ###################### Helper Functions #############################
    @torch.inference_mode()
    def get_jacobian_trajectory(
        self,
        model,
        tokenizer,
        input_ids,
        attention_mask,
        max_new_tokens,
    ):
        bsz = input_ids.shape[0]
        prompt_lens = [torch.sum(t) for t in attention_mask]

        max_total_len = max(prompt_lens) + max_new_tokens

        # initialize the first point of jacobian trajectory
        starting_tokens = torch.full((bsz, max_total_len), tokenizer.pad_token_id, dtype=torch.long, device="cuda")
        for i in range(bsz):
            starting_tokens[i, :] = torch.tensor(random.choices(input_ids[i][attention_mask[i]==1], k=max_total_len), dtype=torch.long, device="cuda")
        for k, t in enumerate(input_ids):
            prompt = t[attention_mask[k]]
            starting_tokens[i, :prompt_lens[k]] = prompt.detach().cuda()
        
        # begin jacobian iteration
        condition = False
        trajectory = []
        logits_trajectory = []
        itr = 0
        next_generation = starting_tokens
        jacobi_attention_mask = torch.full_like(next_generation, 1).to(starting_tokens.device)
        trajectory.append(starting_tokens)
        while condition == False:
            current_generation = next_generation
            logits = self.get_logits(model, current_generation, jacobi_attention_mask)
            logits_trajectory.append(logits)
            # sampling
            tau = 0.001  # argmax
            distribution = torch.softmax(logits / tau, dim=-1)
            next_generation = torch.argmax(distribution, dim=-1) # starting after the pad token

            # hold prompt unchanged and update generated tokens
            for i in range(bsz):
                next_generation[i, :] = torch.cat((starting_tokens[i, :prompt_lens[i]], next_generation[i, prompt_lens[i]-1:-1]), dim=0)
            trajectory.append(next_generation)
            if torch.all(torch.eq(next_generation, current_generation)).item():
                condition = True
                print(f"Iteration steps: {itr}")
            
            itr+=1
            if itr % 10 == 0:
                print(f'iteration: {itr}')

        return trajectory[:-1], logits_trajectory[-2] # one right-shift offset for logits trajectory to match the corresponding trajectory entry

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
        max_new_tokens,
        mix_ratio
    ):
        org_input_ids = input_ids.clone()

        def sample_token_from_logits(logits):
            tau = 0.001  # argmax
            distribution = torch.softmax(logits / tau, dim=-1)
            next_token_id = torch.multinomial(distribution, num_samples=1)
            return next_token_id

        def generate_one(model, input_ids, attention_mask, past_key_values):
            if past_key_values is None:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
            past_key_values = outputs.past_key_values
            next_token = sample_token_from_logits(outputs.logits[:, -1, :])
            return next_token, past_key_values

        bsz, prompt_len = input_ids.shape
        # always generate the first token for teacher/student to get the kv cache
        student_first_token, student_key_values = generate_one(
            student_model, input_ids, attention_mask, None)
        teacher_first_token, teacher_key_values = generate_one(
            teacher_model, input_ids, attention_mask, None)

        torch.manual_seed(1)
        input_ids = student_first_token if random.random(
        ) < mix_ratio else teacher_first_token
        attention_mask = torch.cat([attention_mask, torch.ones(
            bsz, 1, dtype=torch.long, device='cuda')], dim=1)

        for i in range(max_new_tokens - 1):
            sample_model, past_key_values = (student_model, student_key_values) if random.random(
            ) < mix_ratio else (teacher_model, teacher_key_values)
            next_token, _ = generate_one(sample_model, input_ids,
                                         attention_mask, past_key_values)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones(
                bsz, 1, dtype=torch.long, device='cuda')], dim=1)

        # mask eos
        eos_positions = (input_ids == tokenizer.eos_token_id).nonzero(
            as_tuple=True)
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for row, col in zip(*eos_positions):
            mask[row, col+1:] = True
        input_ids[mask] = tokenizer.pad_token_id
        return torch.cat((org_input_ids, input_ids), dim=-1).cuda()

    def get_logits(self, model, input_ids, attention_mask):
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits


class DistillTrainerCallback(TrainerCallback):
    def __init__(self) -> None:
        super().__init__()
        self.correct_cnt = 0
        self.propose_cnt = 0

        self.alpha = 0
        self.sample_steps = 0

    def on_evaluate(self, args, state, control, **kwargs):
        pass
