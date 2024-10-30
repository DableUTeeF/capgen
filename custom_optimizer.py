from transformers import TrainingArguments, Trainer
from torch.optim.adamw import AdamW, adamw
import torch
from torch.optim.optimizer import _use_grad_for_differentiable


@_use_grad_for_differentiable
def step(self, closure=None):
    """Performs a single optimization step.

    Args:
        closure (Callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    self._cuda_graph_capture_health_check()

    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    for group in self.param_groups:
        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        max_exp_avg_sqs = []
        state_steps = []
        amsgrad = group["amsgrad"]
        beta1, beta2 = group["betas"]

        self._init_group(
            group,
            params_with_grad,
            grads,
            amsgrad,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
        )

        adamw(
            params_with_grad,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            amsgrad=amsgrad,
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=group["maximize"],
            foreach=group["foreach"],
            capturable=group["capturable"],
            differentiable=group["differentiable"],
            fused=group["fused"],
            grad_scale=getattr(self, "grad_scale", None),
            found_inf=getattr(self, "found_inf", None),
        )

    return loss


if __name__ == '__main__':
    AdamW.step = step
    training_args = TrainingArguments(
        output_dir='/tmp/s',
        save_safetensors=False,
    )

    trainer = Trainer(
        model=base_model,
        optimizers=(optim, None)
    )

