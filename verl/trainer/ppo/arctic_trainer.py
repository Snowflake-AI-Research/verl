import torch
from typing import Optional
from torch.utils.data import Dataset, Sampler
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.workers.arctic_workers import ActorRolloutRefWorker
from verl.trainer.ppo.utils import Role, WorkerType
from omegaconf import OmegaConf
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.arctic_rl_client import create_arctic_rl_client

def my_pdb():
    return 
    import pdb; pdb.set_trace()

class ArcticPPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):        
        super().__init__(config=config, 
        tokenizer=tokenizer, 
        processor=processor, 
        role_worker_mapping=role_worker_mapping, 
        resource_pool_manager=resource_pool_manager, 
        ray_worker_group_cls=ray_worker_group_cls, 
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        collate_fn=collate_fn, 
        train_sampler=train_sampler, 
        device_name=device_name)

        self.use_gpu = False
        self.rl_client = create_arctic_rl_client(config=config)
        self.rl_client.initialize.remote(model_name="Qwen/Qwen3-0.6B")
        self.wg_kwargs["arctic_rl_client"] = self.rl_client
        

    def destroy(self):
        # self.actor_rollout_wg.destroy() 
        self.rl_client.destroy.remote()