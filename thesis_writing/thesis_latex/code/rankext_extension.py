# Extracted from the production implementation used for the primary
# CIFAR-100 benchmark. Comments and docstrings were cleaned for thesis
# presentation, and some long statements were re-wrapped; the executable
# integration logic is unchanged. The selected definitions appear in source
# order; unrelated code between them is omitted.

# Cumulative rank after each of the five steps: a rank-16 block is appended
# at every step, reaching a final cumulative rank of 80.
RANKEXT_RANK_SCHEDULE = [16, 32, 48, 64, 80]
RANKEXT_ALPHA_PER_RANK = 2.0

# active_rankext_rank_schedule() returns RANKEXT_RANK_SCHEDULE in the primary
# configuration; the production script enforces this at start-up.
assert active_rankext_rank_schedule() == [16, 32, 48, 64, 80], (
    f"RankExt rank schedule must be [16,32,48,64,80], got {active_rankext_rank_schedule()}"
)


class GrowingRankLoRALinear(nn.Module):
    """
    Growing-rank LoRA wrapper around one frozen linear layer.
    The frozen slice (A_frozen, B_frozen) holds all rank blocks learned at
    earlier steps; the new slice (A_new, B_new) is the only trainable block.
    """

    def __init__(
        self,
        base_layer,
        total_rank,
        frozen_A=None,
        frozen_B=None,
        dropout=0.0,
        old_active_in_forward=True,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.total_rank = int(total_rank)
        self.old_active_in_forward = bool(old_active_in_forward)

        if frozen_A is None or frozen_B is None:
            self.frozen_rank = 0
        else:
            if frozen_A.shape[0] != frozen_B.shape[1]:
                raise ValueError(
                    f"A/B frozen rank mismatch: A={tuple(frozen_A.shape)}, B={tuple(frozen_B.shape)}"
                )
            self.frozen_rank = int(frozen_A.shape[0])

        # Rank of the block appended at this step (16 under the primary
        # schedule).
        self.new_rank = self.total_rank - self.frozen_rank
        if self.new_rank < 0:
            raise ValueError(
                f"new_rank < 0 | total_rank={self.total_rank} frozen_rank={self.frozen_rank}"
            )

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        # The pretrained backbone weight is never updated.
        for p in self.base_layer.parameters():
            p.requires_grad = False

        # alpha = 2 * total_rank, hence a constant scaling of 2 at every step.
        self.rankext_alpha = RANKEXT_ALPHA_PER_RANK * self.total_rank
        self.scaling = self.rankext_alpha / self.total_rank
        self.dropout = nn.Dropout(dropout)

        # Previously learned blocks: copied from the previous step's state and
        # hard-frozen (requires_grad=False).
        # A_frozen: [frozen_rank, in_features]
        # B_frozen: [out_features, frozen_rank]
        if self.frozen_rank > 0:
            self.A_frozen = nn.Parameter(frozen_A.detach().clone().float(),
                                         requires_grad=False)
            self.B_frozen = nn.Parameter(frozen_B.detach().clone().float(),
                                         requires_grad=False)
        else:
            self.A_frozen = None
            self.B_frozen = None

        # Newly appended block: the only trainable LoRA parameters.
        # Standard LoRA initialisation (B_new = 0), so the new block initially
        # contributes nothing to the output.
        if self.new_rank > 0:
            self.A_new = nn.Parameter(
                torch.zeros(self.new_rank, self.in_features))
            self.B_new = nn.Parameter(
                torch.zeros(self.out_features, self.new_rank))
            nn.init.kaiming_uniform_(self.A_new, a=np.sqrt(5))
            nn.init.zeros_(self.B_new)
        else:
            self.A_new = None
            self.B_new = None

    def full_A_B(self):
        # Concatenate the frozen and new blocks into the cumulative factors:
        # A: [total_rank, in_features], B: [out_features, total_rank].
        A_parts = []
        B_parts = []
        if self.frozen_rank > 0:
            A_parts.append(self.A_frozen.to(
                device=self.base_layer.weight.device,
                dtype=self.base_layer.weight.dtype))
            B_parts.append(self.B_frozen.to(
                device=self.base_layer.weight.device,
                dtype=self.base_layer.weight.dtype))
        if self.new_rank > 0:
            A_parts.append(self.A_new)
            B_parts.append(self.B_new)
        if len(A_parts) == 0:
            raise ValueError("No LoRA blocks available in full_A_B.")
        A = torch.cat(A_parts, dim=0)
        B = torch.cat(B_parts, dim=1)
        return A, B

    def current_new_delta(self):
        if self.new_rank <= 0:
            return None
        return (self.B_new @ self.A_new) * float(self.scaling)

    def cumulative_old_delta(self):
        if self.frozen_rank <= 0:
            return None
        return (self.B_frozen @ self.A_frozen) * float(self.scaling)

    def forward(self, x):
        # Output = W_0 x + s * B_frozen A_frozen x' + s * B_new A_new x',
        # where x' is the dropout-regularised input of the LoRA branches.
        base_out = self.base_layer(x)
        x_dropped = self.dropout(x)
        out = base_out

        # Previously learned blocks remain active in the forward pass
        # (old_active_in_forward is True for every reported RankExt variant).
        if self.old_active_in_forward and self.frozen_rank > 0:
            hidden_old = torch.matmul(x_dropped, self.A_frozen.T)
            lora_old = torch.matmul(hidden_old, self.B_frozen.T)
            out = out + self.scaling * lora_old

        if self.new_rank > 0:
            hidden_new = torch.matmul(x_dropped, self.A_new.T)
            lora_new = torch.matmul(hidden_new, self.B_new.T)
            # New-block output warmup: scales only the newly appended block's
            # contribution. The multiplier ramps linearly over the first epoch
            # of a step for non-KD RankExt variants and is exactly 1.0
            # otherwise (including for KD-bearing variants and at evaluation).
            new_block_multiplier = get_rankext_new_block_warmup_multiplier()
            out = out + self.scaling * lora_new * new_block_multiplier

        return out


def get_rank_extension_rank_schedule():
    schedule = [int(v) for v in active_rankext_rank_schedule()]
    if len(schedule) != NUM_STEPS:
        raise ValueError(f"active rank schedule must have NUM_STEPS={NUM_STEPS} entries, got {schedule}")
    for i in range(1, len(schedule)):
        if schedule[i] <= schedule[i - 1]:
            raise ValueError(f"RANKEXT_RANK_SCHEDULE must be strictly increasing, got {schedule}")
    return schedule


def get_rank_extension_rank_triplet(step_idx):
    # Returns (cumulative rank, rank inherited from earlier steps,
    # rank of the new block), e.g. step_idx = 2 -> (48, 32, 16).
    schedule = get_rank_extension_rank_schedule()
    total_rank = int(schedule[step_idx])
    frozen_rank = int(schedule[step_idx - 1]) if step_idx > 0 else 0
    new_rank = int(total_rank - frozen_rank)
    if new_rank <= 0:
        raise ValueError(
            f"Rank schedule must leave a positive new block at each step. step_idx={step_idx}, schedule={schedule}"
        )
    return total_rank, frozen_rank, new_rank


def build_rank_extension_model(previous_rank_state=None, step_idx=0,
                               old_active_in_forward=True):
    """
    Build the RankExt model for one step from the persistent state carried
    over from the previous step (previous_rank_state; None at the first step).
    """
    model = fresh_pretrained_model()

    # Frozen backbone; trainable classifier head.
    for _, p in model.vision_model.named_parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    total_rank, expected_frozen_rank, expected_new_rank = (
        get_rank_extension_rank_triplet(step_idx))
    rankext_target_modules = family_target_modules("rank_extension")
    target_names = find_clip_target_linear_modules(
        model, target_modules=rankext_target_modules)
    model._rank_extension_target_names = list(target_names)
    model._rank_extension_old_active_in_forward = bool(old_active_in_forward)

    print(f"[rank_extension] Step {step_idx + 1}")
    print(f"  total_rank: {total_rank}")
    print(f"  target linear modules: {len(target_names)}")
    print(f"  target module names: {rankext_target_modules}")
    print(f"  rank schedule: {get_rank_extension_rank_schedule()}")
    print(f"  expected_frozen_rank: {expected_frozen_rank}")
    print(f"  expected_new_rank: {expected_new_rank}")
    print(f"  old_active_in_forward: {bool(old_active_in_forward)}")

    # Replace every target projection (q_proj, v_proj) by a growing-rank
    # wrapper: all factors learned so far become its frozen slice, and a new
    # trainable block fills the remaining total_rank - frozen_rank.
    for module_name in target_names:
        parent, child_name = get_parent_module_and_child_name(
            model, module_name)
        base_layer = getattr(parent, child_name)

        frozen_A = None
        frozen_B = None
        if (previous_rank_state is not None
                and module_name in previous_rank_state["lora"]):
            frozen_A = previous_rank_state["lora"][module_name]["A"]
            frozen_B = previous_rank_state["lora"][module_name]["B"]

        setattr(
            parent,
            child_name,
            GrowingRankLoRALinear(
                base_layer=base_layer,
                total_rank=total_rank,
                frozen_A=frozen_A,
                frozen_B=frozen_B,
                dropout=LORA_DROPOUT,
                old_active_in_forward=old_active_in_forward,
            ),
        )

    # The classifier also carries over from the previous step.
    if (previous_rank_state is not None
            and previous_rank_state["classifier_weight"] is not None):
        with torch.no_grad():
            model.classifier.weight.copy_(
                previous_rank_state["classifier_weight"].to(
                    device=model.classifier.weight.device,
                    dtype=model.classifier.weight.dtype,
                )
            )
            model.classifier.bias.copy_(
                previous_rank_state["classifier_bias"].to(
                    device=model.classifier.bias.device,
                    dtype=model.classifier.bias.dtype,
                )
            )

    return model


def extract_rank_extension_state(model):
    """
    Persistent RankExt state at the end of a step: the cumulative
    (frozen + new) factors of every adapted module and the classifier.
    It is passed as previous_rank_state to build_rank_extension_model()
    at the next step, where all of these factors become frozen.
    """
    state = {"lora": {}, "classifier_weight": None, "classifier_bias": None}
    for name, module in model.named_modules():
        if isinstance(module, GrowingRankLoRALinear):
            A, B = module.full_A_B()
            state["lora"][name] = {
                "A": A.detach().cpu().clone(),
                "B": B.detach().cpu().clone(),
                "scaling": float(module.scaling),
                "total_rank": int(module.total_rank),
                "frozen_rank": int(module.frozen_rank),
                "new_rank": int(module.new_rank),
                "rankext_alpha": float(module.rankext_alpha),
            }
    state["classifier_weight"] = model.classifier.weight.detach().cpu().clone()
    state["classifier_bias"] = model.classifier.bias.detach().cpu().clone()
    return state
