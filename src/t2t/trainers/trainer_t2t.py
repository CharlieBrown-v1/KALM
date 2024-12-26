from itertools import chain
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Tuple, Union, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from torch import Tensor
from torch.optim import Optimizer
from accelerate import Accelerator
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    DataCollator,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
)
from transformers.utils import is_sagemaker_mp_enabled

from t2t.args import Text2TrajectoryTrainingArguments


class Text2TrajectoryTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: Text2TrajectoryTrainingArguments = None,
        data_collator: Union[DataCollator, Callable[[List[Dict[str, Union[Tensor, Any]]]], Dict[str, Tensor]]] = None,
        train_dataset: torch.utils.data.Dataset = None,
        eval_dataset: torch.utils.data.Dataset = None,
        tokenizer: PreTrainedTokenizerBase = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Callable[[EvalPrediction], Dict[str, float]] = None,
        callbacks: List[TrainerCallback] = None,
        optimizers: Tuple[Optimizer, LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[Tensor, Tensor], Tensor] = None
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # TODO: further check the trainer initialization process and modify according to specific requirements

    # lky waste: override this to frozen parameters of llama
    # use new way to freeze parameters, change name to stop override
    def create_optimizer_1(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            target_opt_name_list = [
                'embed_observation.weight',
                'embed_observation.bias',
                'embed_action.weight',
                'observation_head.weight', 
                'observation_head.bias',
                'action_head.weight',
                'action_head.bias',
                ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and n in target_opt_name_list)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and n in target_opt_name_list)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")
            
        if is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
    
    # lky: override this to save only embedding layer rather than the whole model
    # stop override to save the whole model
    def save_model_1(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """
        '''
        # origin version
        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif self.is_fsdp_enabled:
            if ("FULL_STATE_DICT" in str(self.accelerator.state.fsdp_plugin.state_dict_type)) and (
                version.parse(accelerate_version) > version.parse("0.24.1")
            ):
                state_dict = self.accelerator.get_state_dict(self.model)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
        elif self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.deepspeed)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                    " zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model_wrapped.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")
        '''

        assert self.is_deepspeed_enabled

        self.accelerator.wait_for_everyone()

        state_dict = self.accelerator.get_state_dict(self.deepspeed)

        if self.accelerator.is_main_process:
            print('\n---save embedding layer---\n')
            
            # save_name_list = ['embed_observation', 'embed_action']
            save_name_list = ['embed_observation', 'embed_action', 'observation_head', 'action_head']
            for name in save_name_list:
                name_state_dict = {
                    key[len(f'{name}.'):]: value for key, value in state_dict.items() if name in key
                }
                if self.args.should_save:
                    torch.save(name_state_dict, self.args.output_dir + f'/{name}_epoch_{self.state.epoch}.pt')


# lky waste: old way to change model save method from trainer default to embedding only with callback
# incompatible with deepspeed
class SaveCallback(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control, model: PreTrainedModel, **kwargs):
        save_name_list = ['embed_observation', 'embed_action']
        state_dict = model.state_dict()
        for name in save_name_list:
            name_state_dict = {
                key[len(f'{name}.'):]: value for key, value in state_dict.items() if name in key
            }
            torch.save(name_state_dict, args.output_dir + f'/{name}_epoch_{state.epoch}.pt')
        