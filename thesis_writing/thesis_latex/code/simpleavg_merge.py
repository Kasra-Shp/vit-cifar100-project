# Extracted from the production implementation used for the primary
# CIFAR-100 benchmark. Comments and docstrings were cleaned for thesis
# presentation; the executable integration logic is unchanged.


def extract_lora_state(model):
    """
    Extract the state of one trained SimpleAvg specialist (one step):
    - the dense LoRA update delta_W of every adapted module;
    - the specialist's classifier weights and biases.
    Called once at the end of every step; the per-step results are collected,
    in step order, into the list step_states used below.

    PEFT convention:
    A shape = [r, in_features]
    B shape = [out_features, r]
    delta_W = B @ A * scaling
    """
    state = {
        "deltas": {},
        "lora_A": {},
        "lora_B": {},
        "scaling": {},
        "classifier_weight": None,
        "classifier_bias": None,
    }

    for name, module in model.named_modules():
        has_lora = (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and "default" in module.lora_A
            and "default" in module.lora_B
        )

        if not has_lora:
            continue

        adapter_name = "default"
        A = module.lora_A[adapter_name].weight.detach().cpu().float().clone()
        B = module.lora_B[adapter_name].weight.detach().cpu().float().clone()

        scaling = (
            module.scaling[adapter_name]
            if isinstance(module.scaling, dict)
            else module.scaling
        )

        scaling = float(scaling)
        # Dense update of this module: [out_features, in_features].
        delta = scaling * (B @ A)

        plain_name = normalize_module_name(name)
        state["deltas"][plain_name] = delta.clone()
        # The raw A/B factors are stored alongside the dense update, but the
        # SimpleAvg merge below uses only state["deltas"].
        state["lora_A"][plain_name] = A
        state["lora_B"][plain_name] = B
        state["scaling"][plain_name] = scaling

    for name, tensor in model.state_dict().items():
        if "classifier.modules_to_save.default.weight" in name:
            state["classifier_weight"] = tensor.detach().cpu().clone()

        if "classifier.modules_to_save.default.bias" in name:
            state["classifier_bias"] = tensor.detach().cpu().clone()

    return state


def simple_average_deltas(step_states):
    # step_states holds one extract_lora_state() result per step.
    # Module-wise arithmetic mean of the stored dense updates of all steps.
    # The LoRA factors A and B are never averaged.
    keys = sorted(step_states[0]["deltas"].keys())
    merged = {}

    for key in keys:
        vals = []

        for state in step_states:
            if key in state["deltas"]:
                vals.append(state["deltas"][key].float())

        merged[key] = torch.stack(vals, dim=0).mean(dim=0)

    return merged


def apply_deltas_to_base(merged_deltas, step_states):
    """
    Apply the merged dense updates to a fresh pretrained CLIP-ViT model and
    stitch the classifier rows of every class from the specialist of the step
    that introduced it.
    """
    # A fresh pretrained backbone ensures that the deployed weights are
    # exactly W_0 + mean(delta_W): no specialist's weights are reused.
    model = fresh_pretrained_model()

    with torch.no_grad():
        # Add each averaged dense update directly to the corresponding frozen
        # weight matrix; no separate LoRA module remains after this step.
        for key, delta in merged_deltas.items():
            try:
                module = get_submodule_by_name(model, key)
            except Exception as e:
                print("Could not find module:", key, "|", e)
                continue

            if not hasattr(module, "weight"):
                print("Module has no weight:", key)
                continue

            module.weight.add_(
                delta.to(
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
            )

        # Classifier-row stitching: the rows (and biases) of the classes
        # introduced at a given step are copied from that step's specialist.
        for step_idx, state in enumerate(step_states):
            classes = classes_for_step(step_idx)

            if state["classifier_weight"] is None:
                print("Missing classifier for step", step_idx + 1)
                continue

            w = state["classifier_weight"].to(model.classifier.weight.device)
            b = state["classifier_bias"].to(model.classifier.bias.device)

            for c in classes:
                model.classifier.weight[c].copy_(w[c])
                model.classifier.bias[c].copy_(b[c])

    return model
