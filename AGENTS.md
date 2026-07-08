\# Project Instructions for Codex



Main file:

vit\_lora\_cifar100\_full5step\_n5.ipynb



This notebook is the main experiment file. Do not work on older notebook/script variants unless explicitly asked.



Notebook structure:

\- Cell 2: main configuration and hyperparameters

\- Cell 8: LoRA extraction and merge helpers

\- Cell 14: orthogonal\_loss implementation

\- Cell 15: rank-extension implementation

\- Cell 18: final pivot table and summary metrics

\- Cell 19: plots

\- Cell 20: supervisor explanation



Project:

Continual learning on CIFAR-100 using CLIP-ViT + LoRA.



Backbone:

openai/clip-vit-base-patch16



Main setup:

\- CIFAR-100

\- 5 steps

\- 20 classes per step

\- LoRA target modules: q\_proj and v\_proj



Main goal:

The goal is to obtain the best possible and most defensible results across all relevant methods, while keeping the experimental design clean and comparable.



The current results for rank-extension and orthogonality are disappointing and need to be debugged/improved carefully. The goal is not just to make the code run, but to identify why these methods underperform and improve them through minimal, well-justified changes.



Important comparison logic:

\- The supervisor recommended not relying on replay as the main method.

\- However, I want to build a comparative experimental setup with both:

&#x20; - no replay

&#x20; - with replay

\- Replay should be treated as an additional comparison/diagnostic, not as the only final solution.

\- The notebook should allow clean comparison between replay and no-replay variants.



Main methods/comparisons:

\- seq\_ft\_no\_replay

\- simple\_avg\_no\_replay

\- simple\_avg\_replay

\- rank\_extension\_no\_replay

\- rank\_extension\_replay

\- joint\_upper\_bound if available



Orthogonality clarification:

Orthogonality is not a standalone final method by itself.

It should be treated as an optional regularizer that can be added to existing methods and then compared against the same method without orthogonality.



For example:

\- simple\_avg\_no\_replay

\- simple\_avg\_no\_replay\_orth

\- simple\_avg\_replay

\- simple\_avg\_replay\_orth

\- rank\_extension\_no\_replay

\- rank\_extension\_no\_replay\_orth

\- rank\_extension\_replay

\- rank\_extension\_replay\_orth



After comparison, if orthogonality does not improve results, it can be removed from the final method. Its role is to test whether it improves average-merging or rank-extension, not to be presented as an independent method.



Supervisor-intended rank-extension:

\- There should be one growing LoRA.

\- Step 1: rank 8

\- Step 2: rank 16, copy old rank-8 A/B into the new LoRA and freeze copied ranks.

\- Step 3: rank 24, copy old rank-16 A/B into the new LoRA and freeze copied ranks.

\- Continue similarly.

\- Only the newly added rank block should be trainable.



Rank-extension objective:

\- Check whether the current implementation truly follows the growing-rank design.

\- If it only sequentially absorbs or merges independent LoRAs, that is not sufficient.

\- The old copied rank blocks must be preserved and frozen.

\- The optimizer should update only the newly added rank block.

\- Add diagnostics to verify whether old copied A/B blocks changed after later training.



Orthogonality objective:

\- Previous LoRA should be absorbed into the model after each step when orthogonality is being tested.

\- Next LoRA should be trained with:

&#x20; CE loss + lambda\_orth \* mean over layers of trace(M\_previous @ delta\_W\_current.T)

\- delta\_W should correspond to the current LoRA update for the same q\_proj/v\_proj layer.

\- Implementation must be dimensionally correct and numerically stable.

\- Orthogonality should be tested as a switch/regularizer inside existing methods, especially average merging and rank-extension.



Strict constraints:

\- Modify only vit\_lora\_cifar100\_full5step\_n5.ipynb unless I explicitly ask otherwise.

\- Do not rename existing variables.

\- Do not rewrite the notebook.

\- Do not do large refactors.

\- Do not install packages.

\- Do not run heavy training.

\- Do not add cudatoolkit or any CUDA-related package.

\- Keep changes minimal and cell-level.

\- If changing code, explain exactly which cell changed and why.

\- Prefer configuration switches over duplicated large blocks of code.

\- Keep results comparable across methods.



Goal:

I will run the notebook manually on the university cluster. Codex should only prepare local code changes and analyze results I paste back later.



What I need:

1\. Check whether Cell 15 correctly implements growing-rank LoRA.

2\. Check whether old copied rank blocks are really frozen.

3\. Check whether Cell 14 orthogonality loss is correct.

4\. Check whether replay and no-replay variants are implemented cleanly.

5\. Check whether orthogonality can be added/removed as a regularizer for average-merging and rank-extension.

6\. Check whether final summary metrics correctly compare:

&#x20;  - no replay vs replay

&#x20;  - no orthogonality vs orthogonality

&#x20;  - simple average vs rank-extension

&#x20;  - rank-extension vs joint upper bound

7\. Add useful diagnostics if needed.

8\. After I paste cluster results, analyze them and suggest the next minimal change.



Important output metrics:

\- first\_step accuracy

\- later\_steps accuracy

\- all\_seen accuracy

\- forgetting from step 1

\- replay gain

\- orthogonality gain

\- gap to joint upper bound if available

\- whether rank-extension is better or worse than simple average



Expected Codex behavior:

\- First inspect and diagnose.

\- Do not apply changes immediately unless I approve.

\- Explain what is wrong and why.

\- Propose minimal patches.

\- Apply only approved minimal patches.

\- After results are available, analyze them honestly and suggest one next hypothesis at a time.









\# Relevant Papers / Method References



The supervisor suggested two papers as references:



1\. arXiv:2505.15875

&#x20;  "Decouple and Orthogonalize: A Data-Free Framework for LoRA Merging"



&#x20;  Use this paper as the main reference for LoRA merging improvements.

&#x20;  Important ideas to consider:

&#x20;  - Existing full-finetuning model-merging methods may perform poorly when directly applied to LoRA.

&#x20;  - LoRA modules can have large parameter magnitude variance.

&#x20;  - Magnitude variance can hurt merging.

&#x20;  - The paper proposes decoupling magnitude and direction before merging.

&#x20;  - It also uses orthogonal constraints to reduce interference during merging.

&#x20;  - For this project, this paper is most relevant to:

&#x20;    - simple\_avg\_no\_replay

&#x20;    - simple\_avg\_replay

&#x20;    - possible column-importance weighted merging

&#x20;    - possible orthogonalized / decoupled LoRA merging

&#x20;  - Do not blindly copy the whole paper method unless requested.

&#x20;  - First inspect whether our simple average merging can be upgraded toward a more paper-consistent merging method with minimal changes.



2\. arXiv:2504.13407

&#x20;  "LoRA-Based Continual Learning with Constraints on Critical Parameter Changes"



&#x20;  Use this paper as the main reference for the experimental setup and LoRA-based continual learning design.

&#x20;  Important ideas to consider:

&#x20;  - LoRA-based continual learning is evaluated on pretrained ViT-style models and class-incremental benchmarks such as Split CIFAR-100.

&#x20;  - Orthogonal LoRA tuning can reduce forgetting, but the paper argues it may still allow critical parameters for previous tasks to change.

&#x20;  - The paper proposes freezing critical ViT parameter matrices before learning later tasks.

&#x20;  - It also proposes orthogonal LoRA composition based on QR decomposition.

&#x20;  - For this project, this paper is most relevant to:

&#x20;    - pretrained ViT / CLIP-ViT continual learning setup

&#x20;    - Split CIFAR-100 evaluation

&#x20;    - orthogonality as a regularizer/composition mechanism

&#x20;    - protecting previous-task knowledge

&#x20;    - checking whether old classifier rows or critical parameters drift



Important:

These papers should guide the method and experimental design, but the first priority is still to implement the supervisor/Pietro rank-extension exactly:

one single growing LoRA whose old rank slices are copied, frozen, and preserved.



Do not replace the supervisor-intended rank-extension with a completely different paper method unless explicitly approved.



Preferred implementation strategy:

1\. First fix the current implementation to match the supervisor's exact rank-extension design.

2\. Then make simple average merging stronger, using ideas from arXiv:2505.15875 if feasible.

3\. Then test orthogonality as an add-on/regularizer/composition improvement, guided by arXiv:2504.13407 and arXiv:2505.15875.

4\. Keep every change separable so results remain interpretable.

