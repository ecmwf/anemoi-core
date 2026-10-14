# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING
from typing import Any

import torch
import torch.distributed as dist
from pytorch_lightning.plugins.precision import Precision

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytorch_lightning as pl
    from torch import nn
    from torch.optim import Optimizer

LOGGER = logging.getLogger(__name__)


class BF16FP32OptPrecision(Precision):
    """BF16 precision plugin with FP32 reference weights.

    This plugin maintains two copies of parameters:
    1. Model parameters in bf16 (used for forward/backward)
    2. Reference weights in fp32 (used for optimizer updates)

    The reference weights approach ensures numerical stability by:
    - Avoiding precision loss from repeated bf16↔fp32 conversions
    - Allowing small optimizer updates to accumulate correctly in fp32
    - Maintaining optimizer state (momentum, variance) in fp32

    Trade-off: Uses approximately 40% more memory than in-place casting approach.

    The result is:
    - Model parameters: bf16
    - Reference weights: fp32
    - Optimizer state: fp32
    - Checkpoints: bf16 model params + fp32 reference weights + fp32 optimizer state

    Note that:
    - gradient accumulation happens in fp32 on the reference weights
    - gradient clipping happens at fp32 precision on the reference weights


    Attributes
    ----------
    precision : str
        Precision identifier for Lightning
    _model_converted : bool
        Whether the model has been converted to bf16
    _ref_by_model_param : dict[nn.Parameter, torch.Tensor] | None
        Mapping from BF16 model parameters to their FP32 reference weights
        Using dict ensures correct handling of weight tying (shared parameters)
    _model : nn.Module | None
        Reference to the model, needed for parameter name mapping in checkpoints
    """

    precision: str = "bf16-fp32-opt"

    def __init__(self) -> None:
        """Initialize the BF16FP32OptPrecision plugin."""
        super().__init__()
        self._model_converted = False
        self._ref_by_model_param: dict[nn.Parameter, torch.Tensor] | None = None
        self._model: nn.Module | None = None

    def convert_module(self, module: nn.Module) -> nn.Module:
        """Convert the model to bfloat16.

        This is called once by Lightning during setup, before the optimizer
        is created. The model will remain in bf16 for all forward/backward passes.

        Parameters
        ----------
        module : nn.Module
            The model to convert

        Returns
        -------
        nn.Module
            The model converted to bfloat16
        """
        LOGGER.info("BF16FP32OptPrecision: Converting model to bfloat16")
        module.to(torch.bfloat16)
        self._model_converted = True
        self._model = module  # Store reference for parameter name mapping
        LOGGER.info(
            "BF16FP32OptPrecision: Model converted, first param dtype: %s",
            next(module.parameters()).dtype,
        )
        return module

    def forward_context(self) -> nullcontext:
        """Return context manager for forward pass.

        No autocast is needed since the model is already in bf16.

        Returns
        -------
        nullcontext
            Empty context manager
        """
        return nullcontext()

    def backward(
        self,
        tensor: torch.Tensor,
        model: pl.LightningModule,
        optimizer: Optimizer | None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Run backward pass and copy gradients to fp32 reference weights.

        After the actual backward pass completes, we immediately copy bf16 gradients
        to fp32 reference weights. This enables gradient accumulation to happen in
        fp32 instead of bf16, improving numerical stability.

        Parameters
        ----------
        tensor : torch.Tensor
            The loss value to backpropagate
        model : pl.LightningModule
            The model being trained
        optimizer : Optimizer | None
            Current optimizer (None if manual optimization)
        *args : Any
            Additional args for backward
        **kwargs : Any
            Additional kwargs for backward
        """
        # Run actual backward pass (computes bf16 gradients on model params)
        super().backward(tensor, model, optimizer, *args, **kwargs)

        # Copy bf16 gradients to fp32 reference weights (accumulates)
        if self._ref_by_model_param is not None:
            self._copy_grads_to_reference_weights()

    def _create_reference_weights(self, optimizer: Optimizer) -> None:
        """Create FP32 reference weights and replace optimizer params.

        This creates a FP32 copy of all model parameters and reconfigures
        the optimizer to track the reference weights instead of model params.

        Handles weight tying correctly: if the same nn.Parameter appears multiple
        times in optimizer.param_groups, it will map to the same reference weight.

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer to reconfigure
        """
        if self._ref_by_model_param is not None:
            return  # Already created

        LOGGER.info("BF16FP32OptPrecision: Creating FP32 reference weights")

        self._ref_by_model_param = {}

        # First pass: create unique reference weights for each unique parameter
        # This handles weight tying - same param object gets same reference weight
        all_params_seen = set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                all_params_seen.add(p)

        for model_param in all_params_seen:
            reference_param = model_param.detach().clone().float()
            reference_param.requires_grad_(model_param.requires_grad)
            self._ref_by_model_param[model_param] = reference_param

        # Second pass: replace optimizer's params with reference weights
        for group in optimizer.param_groups:
            original_params = group["params"]
            reference_params = []
            for model_param in original_params:
                # Look up the reference weight for this model param
                reference_param = self._ref_by_model_param[model_param]
                reference_params.append(reference_param)

            # Replace optimizer's params with reference weights
            group["params"] = reference_params

        total_params = len(self._ref_by_model_param)
        total_size_gb = sum(p.numel() * 4 for p in self._ref_by_model_param.values()) / 1e9
        num_param_slots = sum(len(g["params"]) for g in optimizer.param_groups)

        assert (
            len(all_params_seen) == total_params
        ), f"BF16FP32OptPrecision: Expected {len(all_params_seen)} reference weights but created {total_params}"

        for group in optimizer.param_groups:
            for p in group["params"]:
                assert (
                    p.dtype == torch.float32
                ), f"BF16FP32OptPrecision: Optimizer param has dtype {p.dtype}, expected torch.float32"

        if num_param_slots != total_params:
            LOGGER.info(
                "BF16FP32OptPrecision: Detected weight tying - %s param slots map to %s unique parameters",
                num_param_slots,
                total_params,
            )

        LOGGER.info(
            "BF16FP32OptPrecision: Created %s reference weights (~%.2f GB in fp32)",
            total_params,
            total_size_gb,
        )

    def _copy_grads_to_reference_weights(self) -> None:
        """Copy and accumulate gradients from BF16 model params to FP32 reference weights.

        After each backward pass, gradients are computed on the bf16 model parameters.
        We copy and accumulate these gradients on the fp32 reference weights, enabling
        gradient accumulation to happen in fp32 for better numerical stability.

        IMPORTANT: Clears gradients on model params after copying to prevent bf16 accumulation.
        """
        assert self._ref_by_model_param is not None, "Reference weights not created"

        for model_param, reference_param in self._ref_by_model_param.items():
            if model_param.grad is not None:
                # Convert bf16 gradient to fp32
                grad_fp32 = model_param.grad.detach().to(torch.float32)

                # Accumulate gradient on fp32 reference param
                if reference_param.grad is None:
                    reference_param.grad = grad_fp32
                else:
                    reference_param.grad.add_(grad_fp32)

                # IMPORTANT: Clear model grads to prevent bf16 accumulation
                model_param.grad = None

    def _copy_reference_weights_to_model(self) -> None:
        """Copy updated FP32 reference weights back to BF16 model params.

        After the optimizer updates the fp32 reference weights, we copy them
        back to the bf16 model parameters so the next forward pass uses the
        updated values.
        """
        assert self._ref_by_model_param is not None, "Reference weights not created"

        for model_param, reference_param in self._ref_by_model_param.items():
            # Copy updated reference weight (fp32) to model param (bf16)
            model_param.data.copy_(reference_param.data.bfloat16())

    def optimizer_step(
        self,
        optimizer: Optimizer,
        model: pl.LightningModule,  # noqa: ARG002
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        """Run optimizer step with FP32 reference weights.

        Flow:
        1. Run closure (forward + backward in bf16)
           - backward() automatically copies bf16 grads → fp32 reference weights
           - Gradient accumulation happens in fp32
        2. Lightning clips gradients (on fp32 reference weights)
        3. Run optimizer.step() (updates fp32 reference weights)
        4. Copy updated fp32 reference weights back to bf16 model params

        This ensures that:
        - Forward/backward passes run efficiently in bf16
        - Gradient accumulation happens in fp32 (numerically stable)
        - Gradient clipping operates on fp32 (accurate norms)
        - Optimizer updates happen accurately in fp32
        - No precision loss from repeated conversions

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer to step
        model : pl.LightningModule
            The model being trained (unused, required by interface)
        closure : Callable
            Closure that runs forward and backward pass
        **kwargs : Any
            Additional kwargs for optimizer.step()

        Returns
        -------
        Any
            Result from closure (typically the loss value)
        """
        # Create reference weights on first call (fallback if prepare_optimizer() wasn't called)
        if self._ref_by_model_param is None:
            LOGGER.warning(
                "BF16FP32OptPrecision: Creating reference weights in optimizer_step(). "
                "This may cause issues with checkpoint resume if optimizer state was already loaded. "
                "Custom strategies should call prepare_optimizer() during setup().",
            )
            self._create_reference_weights(optimizer)

        # 1. Run closure (forward + backward in bf16, gradients copied to fp32 in backward())
        closure_result = closure()

        # 2. Optimizer step updates fp32 reference weights
        # (creates/updates fp32 optimizer state on first call)
        optimizer.step(**kwargs)

        # 3. Copy updated fp32 reference weights back to bf16 model params
        self._copy_reference_weights_to_model()

        return closure_result

    def state_dict(self) -> dict:
        """Return state dict for checkpointing.

        Saves reference weights by parameter name for robust checkpoint recovery.
        This allows correct restoration even if parameter order changes.

        Returns
        -------
        dict
            State dictionary containing precision metadata and reference weights by name
        """
        state = {
            "precision_mode": self.precision,
            "model_converted": self._model_converted,
            "model_dtype": str(torch.bfloat16),
        }

        # Save reference weights by parameter name if they exist
        if self._ref_by_model_param is not None and self._model is not None:
            # Build name->parameter mapping
            name_map = {}
            for n, p in self._model.named_parameters():
                name_map[id(p)] = n

            # Save reference weights keyed by parameter name
            refs_by_name = {}
            for model_param, ref_param in self._ref_by_model_param.items():
                param_name = name_map.get(id(model_param), f"param_{id(model_param)}")
                refs_by_name[param_name] = ref_param.detach().cpu()

            state["reference_params_by_name"] = refs_by_name
            state["has_reference_weights"] = True
        else:
            state["has_reference_weights"] = False

        return state

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state dict from checkpoint.

        Validates that the checkpoint was saved with compatible precision settings
        and stores reference weights for restoration in prepare_optimizer().

        Parameters
        ----------
        state_dict : dict
            State dictionary from checkpoint
        """
        saved_mode = state_dict.get("precision_mode")
        if saved_mode and saved_mode != self.precision:
            LOGGER.warning(
                "Checkpoint was saved with precision mode '%s', loading with '%s'. "
                "Model will be converted to bfloat16.",
                saved_mode,
                self.precision,
            )
        self._model_converted = state_dict.get("model_converted", False)

        # Store saved reference weights for restoration in prepare_optimizer()
        # We can't restore them here because the optimizer hasn't been created yet
        if state_dict.get("has_reference_weights", False):
            if "reference_params_by_name" in state_dict:
                saved_references = state_dict["reference_params_by_name"]
                LOGGER.info(
                    "BF16FP32OptPrecision: Loaded %s reference weights from checkpoint",
                    len(saved_references),
                )
                self._saved_reference_params_by_name = saved_references
            elif "reference_params" in state_dict:
                # Legacy format (list-based) - warn and skip
                LOGGER.warning(
                    "BF16FP32OptPrecision: Checkpoint uses legacy reference weights format. "
                    "Reference weights will be reinitialized from model parameters.",
                )

    def prepare_optimizer(self, optimizer: Optimizer, model: pl.LightningModule) -> None:  # noqa: ARG002
        """Prepare optimizer with FP32 reference weights.

        This method must be called AFTER optimizer creation but BEFORE optimizer state
        is loaded from checkpoint. It:
        1. Creates FP32 reference weights from BF16 model parameters
        2. Replaces optimizer's params with reference weights
        3. Restores saved reference weights from checkpoint (if available)
        4. Validates optimizer configuration for compatibility

        Called by the strategy's setup() method.

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer to prepare
        model : pl.LightningModule
            The Lightning module (unused, required by interface)
        """
        # Create reference weights (replaces optimizer params with fp32 refs)
        self._create_reference_weights(optimizer)

        # Restore saved reference weights from checkpoint if available
        if hasattr(self, "_saved_reference_params_by_name") and self._saved_reference_params_by_name is not None:
            self._restore_reference_weights_from_checkpoint()

        # Validate optimizer configuration
        self._validate_optimizer_config(optimizer)

    def _restore_reference_weights_from_checkpoint(self) -> None:
        """Restore FP32 reference weights from checkpoint.

        Matches saved reference weights to current model parameters by name
        and copies the saved fp32 values to the reference weights.
        """
        assert self._model is not None, "Model not available for parameter name mapping"
        assert self._ref_by_model_param is not None, "Reference weights not created"
        assert hasattr(self, "_saved_reference_params_by_name"), "No saved reference weights"

        saved_refs = self._saved_reference_params_by_name

        # Build name->parameter mapping for current model
        name_to_param = dict(self._model.named_parameters())

        # Restore each saved reference weight by matching parameter names
        restored_count = 0
        for param_name, saved_ref in saved_refs.items():
            if param_name in name_to_param:
                model_param = name_to_param[param_name]
                if model_param in self._ref_by_model_param:
                    ref_param = self._ref_by_model_param[model_param]
                    # Validate shape matches
                    if ref_param.shape == saved_ref.shape:
                        ref_param.data.copy_(saved_ref.to(dtype=torch.float32, device=ref_param.device))
                        restored_count += 1
                    else:
                        LOGGER.warning(
                            "BF16FP32OptPrecision: Shape mismatch for parameter '%s': "
                            "checkpoint has %s, model has %s. Using model initialization.",
                            param_name,
                            saved_ref.shape,
                            ref_param.shape,
                        )
            else:
                LOGGER.warning(
                    "BF16FP32OptPrecision: Parameter '%s' in checkpoint not found in model. Skipping.",
                    param_name,
                )

        LOGGER.info(
            "BF16FP32OptPrecision: Restored %s/%s reference weights from checkpoint",
            restored_count,
            len(saved_refs),
        )

        # Clear saved refs to free memory
        self._saved_reference_params_by_name = None

    def _validate_optimizer_config(self, optimizer: Optimizer) -> None:
        """Validate optimizer configuration for compatibility.

        Warns if optimizer uses features that may not work with reference weights.

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer to validate
        """
        # Check for foreach/fused optimizers which may have issues
        for group_idx, group in enumerate(optimizer.param_groups):
            if group.get("foreach", False):
                LOGGER.warning(
                    "BF16FP32OptPrecision: Optimizer param group %s has foreach=True. "
                    "This may cause issues with reference weights. Recommend using foreach=False.",
                    group_idx,
                )
            if group.get("fused", False):
                LOGGER.warning(
                    "BF16FP32OptPrecision: Optimizer param group %s has fused=True. "
                    "This may cause issues with reference weights. Recommend using fused=False.",
                    group_idx,
                )

    def get_ddp_communication_hook(self) -> Callable:
        """Get DDP communication hook for fp32 gradient reduction.

        Returns a communication hook that performs gradient all-reduce in fp32
        instead of bf16. This improves numerical stability when training with
        many GPUs, as averaging in fp32 reduces rounding errors.

        Returns
        -------
        Callable
            DDP communication hook function
        """

        def fp32_gradient_reduction_hook(state, bucket):
            """DDP communication hook to reduce gradients in fp32 (async).

            Converts bf16 gradients to fp32, performs async all-reduce, then converts
            back to bf16. This ensures gradient averaging across GPUs happens in
            higher precision while allowing overlap with computation.

            Parameters
            ----------
            state : object
                State object (process group)
            bucket : dist.GradBucket
                Bucket containing gradients to reduce

            Returns
            -------
            torch.futures.Future[torch.Tensor]
                Future that resolves to reduced gradients in bf16
            """
            # Get bf16 gradients from bucket
            bf16_grads = bucket.buffer()

            # Convert to fp32 for reduction
            fp32_grads = bf16_grads.to(torch.float32)

            # Start async all-reduce in fp32 (sum across all ranks)
            work = dist.all_reduce(fp32_grads, op=dist.ReduceOp.SUM, group=state, async_op=True)

            # Get future from the work handle
            fut = work.get_future()

            # Chain finalization: average and convert back to bf16
            def finalize(fut):
                # fut.value() returns a list with the result tensor
                result = fut.value()[0] if isinstance(fut.value(), list) else fut.value()

                # Average by world size
                world_size = dist.get_world_size(group=state)
                result.div_(world_size)

                # Convert back to bf16 and copy into original buffer
                bf16_grads.copy_(result.to(torch.bfloat16))

                return bf16_grads

            # Return chained future
            return fut.then(finalize)

        return fp32_gradient_reduction_hook
