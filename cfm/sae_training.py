"""
This code is adapted from "https://github.com/saprmarks/dictionary_learning/blob/main/dictionary_learning/training.py"
"""
"""
Training dictionaries
"""

import json
import torch.multiprocessing as mp
import os
from queue import Empty
from typing import Optional
from contextlib import nullcontext

import torch as t
from tqdm import tqdm

import wandb

from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.evaluation import evaluate
from dictionary_learning.trainers.standard import StandardTrainer


def new_wandb_process(config, log_queue, entity, project):
    try: 
        if config is None:
            raise ValueError("Error: W&B config is None! Check sweep setup.")

        wandb.init(entity=entity, project=project, config=config, name=config["wandb_name"])
        while True:
            try:
                log = log_queue.get(timeout=1)
                if log == "DONE":
                    break
                
                # Create Table from data (if present)
                if "Feature Activation Scatter Plot" in log:
                    table = log["Feature Activation Scatter Plot"]
                    wandb.log({"Feature Activation Scatter Plot": wandb.plot.scatter(
                        table, x="Feature Index", y="Activation Frequency",
                        title="Feature Activation Frequencies"
                    )})
                    wandb.log({"Feature Activation Strength Scatter Plot": wandb.plot.scatter(
                        table, x="Feature Index", y="Activation Strength",
                        title="Feature Activation Strengths"
                    )})
                    # wandb.log(table)
                else:
                    wandb.log(log, step= log["step"])

            except Empty:
                continue
            except Exception as e:
                print(f"Error during WandB logging: {e}")
                break
    except Exception as e:
        print(f"Critical error in W&B process: {e}")
    wandb.finish()


def log_stats(
    trainers,
    step: int,
    act: t.Tensor,
    activations_split_by_head: bool,
    transcoder: bool,
    log_queues: list=[],
    verbose: bool=False,
    dataset_type = "train"  # train or val
):
    with t.no_grad():
        # quick hack to make sure all trainers get the same x
        z = act.clone()
        for i, trainer in enumerate(trainers):
            log = {}
            act = z.clone()
            if activations_split_by_head:  # x.shape: [batch, pos, n_heads, d_head]
                act = act[..., i, :]
            if not transcoder:
                act, act_hat, f, losslog = trainer.loss(act, step=step, logging=True)

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()
                # fraction of variance explained
                total_variance = t.var(act, dim=0).sum()
                residual_variance = t.var(act - act_hat, dim=0).sum()
                frac_variance_explained = 1 - residual_variance / total_variance
                log[f"{dataset_type}/frac_variance_explained"] = frac_variance_explained.item()
            else:  # transcoder
                x, x_hat, f, losslog = trainer.loss(act, step=step, logging=True)

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()

            if verbose:
                print(f"Step {step} ({dataset_type}): L0 = {l0}, frac_variance_explained = {frac_variance_explained}")

            # log parameters from training
            log.update({f"{dataset_type}/{k}": v.cpu().item() if isinstance(v, t.Tensor) else v for k, v in losslog.items()})
            log[f"{dataset_type}/l0"] = l0
            trainer_log = trainer.get_logging_parameters()
            for name, value in trainer_log.items():
                if isinstance(value, t.Tensor):
                    value = value.cpu().item()
                log[f"{dataset_type}/{name}"] = value
            log["step"]= step
            if log_queues:
                log_queues[i].put(log)

def get_norm_factor(data, steps: int) -> float:
    """Per Section 3.1, find a fixed scalar factor so activation vectors have unit mean squared norm.
    This is very helpful for hyperparameter transfer between different layers and models.
    Use more steps for more accurate results.
    https://arxiv.org/pdf/2408.05147
    
    If experiencing troubles with hyperparameter transfer between models, it may be worth instead normalizing to the square root of d_model.
    https://transformer-circuits.pub/2024/april-update/index.html#training-saes"""
    total_mean_squared_norm = 0
    count = 0

    for step, act_BD in enumerate(tqdm(data, total=steps, desc="Calculating norm factor")):
        if step > steps:
            break

        count += 1
        mean_squared_norm = t.mean(t.sum(act_BD ** 2, dim=1))
        total_mean_squared_norm += mean_squared_norm

    average_mean_squared_norm = total_mean_squared_norm / count
    norm_factor = t.sqrt(average_mean_squared_norm).item()

    print(f"Average mean squared norm: {average_mean_squared_norm}")
    print(f"Norm factor: {norm_factor}")
    
    return norm_factor



def trainSAE(
    data,
    trainer_configs: list[dict],
    steps: int,
    use_wandb:bool=False,
    wandb_entity:str="",
    wandb_project:str="",
    save_steps:Optional[list[int]]=None,
    save_dir:Optional[str]=None,
    log_steps:Optional[int]=None,
    activations_split_by_head:bool=False,
    transcoder:bool=False,
    run_cfg:dict={},
    normalize_activations:bool=False,
    verbose:bool=False,
    device:str="cuda",
    autocast_dtype: t.dtype = t.float32,
    val_data = None,
    epochs = None,
    dead_feature_threshold = 50_000_000, 
):
    """
    Train SAEs using the given trainers

    If normalize_activations is True, the activations will be normalized to have unit mean squared norm.
    The autoencoders weights will be scaled before saving, so the activations don't need to be scaled during inference.
    This is very helpful for hyperparameter transfer between different layers and models.

    Setting autocast_dtype to t.bfloat16 provides a significant speedup with minimal change in performance.
    """

    device_type = "cuda" if "cuda" in device else "cpu"
    autocast_context = nullcontext() if device_type == "cpu" else t.autocast(device_type=device_type, dtype=autocast_dtype)

    trainers = []
    for i, config in enumerate(trainer_configs):
        if "wandb_name" in config:
            # config["wandb_name"] = f"{config['wandb_name']}_trainer_{i}"
            config["wandb_name"] = f"{config['wandb_name']}"
        trainer_class = config["trainer"]
        del config["trainer"]
        trainers.append(trainer_class(**config))

    wandb_processes = []
    log_queues = []

    if use_wandb:
        # Note: If encountering wandb and CUDA related errors, try setting start method to spawn in the if __name__ == "__main__" block
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.set_start_method
        # Everything should work fine with the default fork method but it may not be as robust
        for i, trainer in enumerate(trainers):
            log_queue = mp.Queue()
            log_queues.append(log_queue)
            wandb_config = trainer.config | run_cfg
            # Make sure wandb config doesn't contain any CUDA tensors
            wandb_config = {k: v.cpu().item() if isinstance(v, t.Tensor) else v 
                          for k, v in wandb_config.items()}
            wandb_process = mp.Process(
                target=new_wandb_process,
                args=(wandb_config, log_queue, wandb_entity, wandb_project),
            )
            wandb_process.start()
            wandb_processes.append(wandb_process)
            print("group indices: ",trainer.ae.group_indices)

    # make save dirs, export config
    if save_dir is not None:
        save_dirs = [
            os.path.join(save_dir, f"trainer_{i}") for i in range(len(trainer_configs))
        ]
        for trainer, dir in zip(trainers, save_dirs):
            os.makedirs(dir, exist_ok=True)
            # save config
            config = {"trainer": trainer.config}
            try:
                config["buffer"] = data.config
            except:
                pass
            with open(os.path.join(dir, "config.json"), "w") as f:
                json.dump(config, f, indent=4)
    else:
        save_dirs = [None for _ in trainer_configs]

    if normalize_activations:
        norm_factor = get_norm_factor(data, steps=100)

        for trainer in trainers:
            trainer.config["norm_factor"] = norm_factor
            # Verify that all autoencoders have a scale_biases method
            trainer.ae.scale_biases(1.0)
            trainer.dead_feature_threshold = dead_feature_threshold # for cc12m

    val_iter = iter(val_data)
    feature_activation_counts = None
    activation_strengths_sum = None
    total_number_of_feature_activation_samples = 0
    step = -1
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}/{epochs}")
        for step_in_epoch, act in enumerate(tqdm(data, total=steps, desc=f"Training SAE epoch {epoch}")):
            step+=1
            act = act.to(device_type, dtype=autocast_dtype, non_blocking=True)
            if step < 5:
                print("act.shape: ", act.shape)
            if normalize_activations:
                act /= norm_factor

            # logging
            if (use_wandb or verbose) and (step % log_steps)-1 == 0:
                print("Logging at step ", step)
                log_stats(
                    trainers, step, act, activations_split_by_head, transcoder, log_queues=log_queues, verbose=verbose, dataset_type="train"
                )
                # validation logs
                with t.no_grad():
                    try:
                        val_acts = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_data) 
                        val_acts = next(val_iter)
                    # val_acts = next(val_iter)  
                    val_acts = val_acts.to(device_type, dtype=autocast_dtype)

                    if normalize_activations:
                        val_acts /= norm_factor

                log_stats(
                    trainers, step, val_acts, activations_split_by_head, transcoder, dataset_type="val",
                    log_queues=log_queues, verbose=verbose
                )
            # count the activations of the concepts in the last iteration
            if (use_wandb or verbose) and (epoch==epochs):
                act_clone = act.clone() 
                with t.no_grad():
                    _, _, f, _ = trainer.loss(act_clone, step=step, logging=True)
                    activated_features = (f != 0).float().sum(dim=0).cpu()
                    activation_strengths = f.sum(dim=0).cpu()
                    number_of_samples = len(f)
                    total_number_of_feature_activation_samples +=number_of_samples
                    if feature_activation_counts is None:
                        feature_activation_counts = t.zeros_like(activated_features)
                        activation_strengths_sum = t.zeros_like(activated_features)
                    feature_activation_counts += activated_features
                    activation_strengths_sum  += activation_strengths


            # saving
            if save_steps is not None and step in save_steps:
                for dir, trainer in zip(save_dirs, trainers):
                    if dir is not None:

                        if normalize_activations:
                            # Temporarily scale up biases for checkpoint saving
                            trainer.ae.scale_biases(norm_factor)

                        if not os.path.exists(os.path.join(dir, "checkpoints")):
                            os.mkdir(os.path.join(dir, "checkpoints"))

                        checkpoint = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
                        t.save(
                            checkpoint,
                            os.path.join(dir, "checkpoints", f"ae_{step}.pt"),
                        )

                        if normalize_activations:
                            trainer.ae.scale_biases(1 / norm_factor)

            # training
            for trainer in trainers:
                with autocast_context:
                    trainer.update(step, act)

    log_feature_activation_scatter_plot(feature_activation_counts, activation_strengths_sum, total_number_of_feature_activation_samples, log_queue= log_queues[0])
    # save final SAEs
    for save_dir, trainer in zip(save_dirs, trainers):
        if normalize_activations:
            trainer.ae.scale_biases(norm_factor)
        if save_dir is not None:
            final = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
            t.save(final, os.path.join(save_dir, "ae.pt"))

    # Signal wandb processes to finish
    if use_wandb:
        for queue in log_queues:
            queue.put("DONE")
        for process in wandb_processes:
            process.join()


# own plot

def log_feature_activation_scatter_plot(feature_activation_counts, activation_strengths, total_samples, log_queue):
    """Prepare and log the feature activation scatter plot using WandB."""
    if feature_activation_counts is None:
        print("No activation data collected!")
        return

    # Normalize activation counts
    activation_frequencies = feature_activation_counts / total_samples
    activation_strengths = activation_strengths / feature_activation_counts

    print("sum of activation frequencies: ", activation_frequencies.sum())
    print("mean activation frequency: ", activation_frequencies.mean())
    print("mean activation strengths: ", activation_strengths.mean())
    # Convert to numpy
    activation_frequencies_np = activation_frequencies.cpu().numpy()
    activation_strengths_np = activation_strengths.cpu().numpy()

    # Create a WandB Table
    table = wandb.Table(columns=["Feature Index", "Activation Frequency", "Activation Strength"])
    for i, (freq, strength) in enumerate(zip(activation_frequencies_np, activation_strengths_np)):
        table.add_data(i, freq, strength)  # Add data point (Feature Index, Activation Frequency, Activation Strength)


    # Send the table to the WandB log queue
    log_queue.put({"Feature Activation Scatter Plot": table})
    
