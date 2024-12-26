from itertools import chain

from transformers import PreTrainedTokenizerBase

from t2t.datasets import InstructedTrajectoryDataset
from t2t.args import InstructedTrajectoryDataArguments
from t2t.utils import current_state_token, next_state_token, action_token


def preprocess_it_dataset(
    dataset: InstructedTrajectoryDataset,
    tokenizer: PreTrainedTokenizerBase,
    args: InstructedTrajectoryDataArguments,
):
    if args.preprocess_inst_token:
        instructions = dataset.instructions.tolist()

        # tokenization
        if hasattr(tokenizer, "add_eos_token"):
            add_eos_token_flag = getattr(tokenizer, "add_eos_token")
            setattr(tokenizer, "add_eos_token", True)
        # only single instruction for each trajectory yet
        if not isinstance(instructions[0], str):
            instructions = list(chain(*instructions))

        # lky: change tokenization
        instructions_lky = []
        if dataset.env_nums == None:
            for id_goal in range(len(dataset.all_goals_list)):
                for id_data, goal_num in enumerate(dataset.goal_nums):
                    if id_goal == goal_num:
                        instructions_lky.extend(dataset.instructions[id_data, :])
                        break
        else:  # lky: for meta data
            for id_env in range(len(dataset.all_env_names_list)):
                for id_data, env_num in enumerate(dataset.env_nums):
                    if id_env == env_num:
                        instructions_lky.extend(dataset.instructions[id_data, :])
                        break

        # print(f'\n-----preprocess, pattern_num: {dataset.pattern_num}-----\n')
        print(f'\n-----preprocess, available patterns: {dataset.available_pattern_list}-----\n')
        
        obss_len, action_len = dataset.observations.shape[1], dataset.actions.shape[1]
        assert obss_len == action_len, "obss and action should have same length"
        instructions_pattern = []

        '''
        if dataset.pattern_num >= 1:
            # lky: add prompt to origin instruction
            prompt = 'Translate the textual instruction to state/action trajectory.\nInstruction: {instruction}.\nTrajectory:' + (current_state_token + action_token) * obss_len
            # instructions = [prompt.format(instruction=instruction) for instruction in instructions]
            instructions_pattern.extend([prompt.format(instruction=instruction) for instruction in instructions_lky])
        if dataset.pattern_num >= 2:
            format = f'You are an expert in identifying dynamics change in the environment. Current state is{current_state_token}, after executing action{action_token}, we get next state{next_state_token}.'
            instructions_pattern.append(format)
        if dataset.pattern_num >= 3:
            format = 'Initial state is{current_state_token}, after completing the instruction "{instruction}", we get terminal state{next_state_token}.'
            instructions_pattern.extend(
                [format.format(instruction=instruction, current_state_token=current_state_token, next_state_token=next_state_token) for instruction in instructions_lky]
                )
        if dataset.pattern_num >= 4:
            prompt = 'Translate the state/action trajectory to textual instruction.\nTrajectory:' + (current_state_token + action_token) * obss_len + '\nInstruction: {instruction}.'
            instructions_pattern.extend([prompt.format(instruction=instruction) for instruction in instructions_lky])
        if dataset.pattern_num >= 5 or dataset.pattern_num <= 0:
            raise NotImplementedError(f"Unsupported pattern_num: {dataset.pattern_num}")
        '''

        # lky: change from tokenize available patterns to tokenize all possible patterns
        # lky: add prompt to origin instruction
        prompt = 'Translate the textual instruction to state/action trajectory.\nInstruction: {instruction}.\nTrajectory:' + (current_state_token + action_token) * obss_len
        instructions_pattern.extend([prompt.format(instruction=instruction) for instruction in instructions_lky])
        format = f'You are an expert in identifying dynamics change in the environment. Current state is{current_state_token}, after executing action{action_token}, we get next state{next_state_token}.'
        instructions_pattern.append(format)
        format = 'Initial state is{current_state_token}, after completing the instruction "{instruction}", we get terminal state{next_state_token}.'
        instructions_pattern.extend(
            [format.format(instruction=instruction, current_state_token=current_state_token, next_state_token=next_state_token) for instruction in instructions_lky]
            )
        prompt = 'Translate the state/action trajectory to textual instruction.\nTrajectory:' + (current_state_token + action_token) * obss_len + '\nInstruction: {instruction}.'
        instructions_pattern.extend([prompt.format(instruction=instruction) for instruction in instructions_lky])
        
        # lky: waste to use new tokenization
        '''
        tokenized_inst = tokenizer(
            instructions,
            add_special_tokens=True,
            padding=True,
            truncation=args.inst_token_trunc,
            max_length=args.inst_token_max_len,
            return_tensors="np",
        )
        '''

        # lky: change tokenization
        tokenized_inst_lky = tokenizer(
            instructions_pattern,
            add_special_tokens=True,
            padding=True,
            truncation=args.inst_token_trunc,
            max_length=args.inst_token_max_len,
            return_tensors="np",
        )
        if hasattr(tokenizer, "add_eos_token"):
            setattr(tokenizer, "add_eos_token", add_eos_token_flag)

        # lky: waste to use new tokenization
        '''
        dataset.inst_tokens = tokenized_inst["input_ids"].reshape(
            len(dataset), len(instructions) // len(dataset), -1)
        dataset.inst_masks = tokenized_inst["attention_mask"].reshape(
            len(dataset), len(instructions) // len(dataset), -1)
        '''
            
        # lky: change tokenization
        inst_type_num = len(instructions_lky)

        '''
        if dataset.pattern_num >= 1:
            dataset.inst_tokens_lky[0] = tokenized_inst_lky["input_ids"][:inst_type_num].reshape(
                len(dataset.all_goals_list), inst_type_num // len(dataset.all_goals_list), -1)
            dataset.inst_masks_lky[0] = tokenized_inst_lky["attention_mask"][:inst_type_num].reshape(
                len(dataset.all_goals_list), inst_type_num // len(dataset.all_goals_list), -1)
        if dataset.pattern_num >= 2:
            dataset.inst_tokens_lky[1] = tokenized_inst_lky["input_ids"][inst_type_num]
            dataset.inst_masks_lky[1] = tokenized_inst_lky["attention_mask"][inst_type_num]
        if dataset.pattern_num >= 3:
            dataset.inst_tokens_lky[2] = tokenized_inst_lky["input_ids"][inst_type_num+1:inst_type_num*2+1].reshape(
                len(dataset.all_goals_list), inst_type_num // len(dataset.all_goals_list), -1)
            dataset.inst_masks_lky[2] = tokenized_inst_lky["attention_mask"][inst_type_num+1:inst_type_num*2+1].reshape(
                len(dataset.all_goals_list), inst_type_num // len(dataset.all_goals_list), -1)
        if dataset.pattern_num >= 4:
            dataset.inst_tokens_lky[3] = tokenized_inst_lky["input_ids"][inst_type_num*2+1:].reshape(
                len(dataset.all_goals_list), inst_type_num // len(dataset.all_goals_list), -1)
            dataset.inst_masks_lky[3] = tokenized_inst_lky["attention_mask"][inst_type_num*2+1:].reshape(
                len(dataset.all_goals_list), inst_type_num // len(dataset.all_goals_list), -1)
        '''

        # lky: change from tokenize available patterns to tokenize all possible patterns
        if dataset.env_nums == None:
            task_num = len(dataset.all_goals_list)
        else:
            task_num = len(dataset.all_env_names_list)
        dataset.inst_tokens_lky[0] = tokenized_inst_lky["input_ids"][:inst_type_num].reshape(
            task_num, inst_type_num // task_num, -1)
        dataset.inst_masks_lky[0] = tokenized_inst_lky["attention_mask"][:inst_type_num].reshape(
            task_num, inst_type_num // task_num, -1)
        dataset.inst_tokens_lky[1] = tokenized_inst_lky["input_ids"][inst_type_num]
        dataset.inst_masks_lky[1] = tokenized_inst_lky["attention_mask"][inst_type_num]
        dataset.inst_tokens_lky[2] = tokenized_inst_lky["input_ids"][inst_type_num+1:inst_type_num*2+1].reshape(
            task_num, inst_type_num // task_num, -1)
        dataset.inst_masks_lky[2] = tokenized_inst_lky["attention_mask"][inst_type_num+1:inst_type_num*2+1].reshape(
            task_num, inst_type_num // task_num, -1)
        dataset.inst_tokens_lky[3] = tokenized_inst_lky["input_ids"][inst_type_num*2+1:].reshape(
            task_num, inst_type_num // task_num, -1)
        dataset.inst_masks_lky[3] = tokenized_inst_lky["attention_mask"][inst_type_num*2+1:].reshape(
            task_num, inst_type_num // task_num, -1)

    if len(dataset.masks.shape) > 2:
        dataset.masks = dataset.masks.squeeze(-1)
    if hasattr(dataset, "observation_masks") and len(getattr(dataset, "observation_masks").shape) > 2:
        setattr(dataset, "observation_masks", getattr(dataset, "observation_masks").squeeze(-1))
    if hasattr(dataset, "action_masks") and len(getattr(dataset, "action_masks").shape) > 2:
        setattr(dataset, "action_masks", getattr(dataset, "action_masks").squeeze(-1))

    return dataset

