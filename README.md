# aisam

Work-in-progress code.

## Verbiage

- For solving any given problem, think: can "thinking patterns/templates" (transformations) be _recursively_ applied?
- "Learning recursive thinking programs"
- Spurious learning of recursable programs (SLeRP): theory and applications (for memory-contrained inferencing)
- Learn how many times to apply the same computation rather than learning different computations
- Decision making process where some recursive/able (repetitive) operation can be used to arrive at the desired decision
- Pretraining for latent COT from articulated COT enables SOTA sample-efficient RL learning of new (long-horizon-reasoning-dependent) environments
    - E.g. Crafter & Emerald BattleFactory

## TODO
- Residual blocks
- ACT - same weights and biases across intermediate steps?


## ACT
- output hidden state becomes input hidden state for next iteration
- output DOES NOT become input for next iteration
- original input is input for next iteration (augmented w/ flag for whether is 1st iter)
- once halt: final output, final hidden state = accumulations--weighted sums (by p_n)

1. network (without ponder cost) will want to compute more and more ad inf
2. ponder cost allow gradients to flow back through dense layer that computes p_n
- exploration in halting level is a push-pull b/w 1 and 2

> The halting level $ N $ is not fixed by the initial parameters $ \theta_0 $. It’s a function of the learned halting probabilities, which are updated via gradients from both the task loss and the ponder cost. These updates allow the model to explore different $ N $ values over time.

NOTE: criticism of ACT from PonderNet paper:

> Unfortunately ACT is notably unstable and sensitive to the choice of a hyper-parameter that trades-off accuracy and computation cost. Additionally, the gradient for the cost of computation can only back-propagate through the last computational step, leading to a biased estimation of the gradient.

Since ponder cost is a function of:
1. The remainder p (1-sum(p_n))
2. N (num recurrent iters)
"The contribution of the final step’s probability (or remainder) dominates the ponder cost’s gradient."

> The gradient is biased because it overemphasizes the last step’s contribution to the ponder cost, potentially underweighting the role of earlier steps in shaping the computation trajectory.

## PonderNet

- Better general purpose algo than ACT (for SLeRP)
- Loss requires L for all n, which is not possible in RL, unless:
    1. You branch the simulator ALOT in real time - not REAALLLY feasible (?), certainly not elagant/typical
    2. You learn a (Q function... I.e., larger q-learned model) or latent dynamics model?
    ...I would only do #1 to benchmark #2 I think...

- Idea(s):
    1. Phases: (1) Train on supervised language-expressed hierarchies from LRXC, (2) Behavior clone other expert trajectories, & (3) do limited RL w/ #2 (above), or a variant of PonderNet that updates only via L_n and does not optimize "halting level" further (max KL penalty from SL-learned distribution?)
    2. Do some version of the above, BUT interleave steps 2 and 3 training with step 1 to retain original (non-optimal, but interpretable) halting behavior/abstraction levels, so that intermediate y_n's can be decoded into coherent language-expressed concepts
    3. Focus on adapting PonderNet for RL by learning latent dynamics models (e.g., dreamer) from which to compute values/loss -- I.E., "PonderNetDreamer"

 ## Dreamer

 Questions:

 We learn the world (latent dynamics model). Then, we use it (exclusively?) to estimate advantages across imagined "rollout" trajectories, yes?

 If so, then just borrow that way of computing advantage, to get loss(es) for PonderNet algo.

2 Things can be done interleaved or in parallel:
1. Learn latent dynamics model
2. Learn action and value models

 Latent dynamics model:
 1. Representation model: p(s_t | s_{t-1}, a_{t-1}, o_t)
 2. Transition model: q(s_t | s_{t-1}, a_{t-1})
 3. Reward model: q(r_t | s_t)

 Other:
 1. Action model: q(a_t | s_t)
 2. Value model: v(s_t) (Bellman-consistency-framed expected (imagined) rewards from each state)

 Action and value models are trained cooperatively (as typicaly in AC policy iteration): action models aims to maximize an estimate of value, and value model aims to match an estimate of the value that changes as the action model changes.

 NOTE: value model learns from expected rewards under the imagined trajectories(?)

## Dreamer adaptations (for my idea):
- Action model = goal model -> action decoder
- Goal model = PonderNet-wrapped inner goal model
- Gets pretrained in phase 0:
    - env encoder ("representation model"): to learn shared control-oriented representation space for pixels/images
- Gets pretrained in phase 1:
    - inner goal model (the text-based goal hierarchies of LRXC)
    - action decoder (the immediate-goal-to-action's of LRXC)
    - (extrapolate trajectories/rewards and add to D)
- Gets pretrained in phase 2 (after we have an LRXC-thought-process-emulating policy):
- Phase 3:
    - Do some PonderNet-algo-based "behavior cloning" on LRXC-gathered and other (never-COT-annotated) trajectories to get closer to the optimal "halting" (computation amount) for making the decisions
    - Pretrain the latent dynamics model on these trajectories/rewards
- Phase 4: Do ponder-augmented DreamerV3 RL

- Phase 5+: Train everything except env encoders and action decoders for generalist control/pondering/representation (i.e., adapt things to be highest- or next- goal-conditioned and train on all sort of envs, goals, and COT data). Then, see how sample-efficiently this "starting point"/"foundation model" generalizes to new domains (Genesis! Genie + LPIPS?)
    - Start with a generalist latent-dynamics "world" model from which we borrow the "world encoder", then goal condition:
        - transition model (next-goal) (by adapting already existing input space)
        - reward model (highest-goal)
        - value model (highest-goal) (can learn to do LPIPS b/w highest-goal and state when applicable)
        - action model (highest-goal)

___

## Altogether:

### Models

1. $E$ - Env/Goal/Action Encoder: $š/ğ/ǎ = f(s/g/a_{img} \text{ and/or } s/g/a_{txt})$

<!-- 1. Representation Model: $p(š_t | š_{t-1}, ǎ_{t-1}, ǒ_t)$ (why does dreamer need this again?) -->

2. $T$ - Transition Model: $q(š_t | š_{t-1}, ǎ_{t-1})$

3. $R$ - Reward Model: $q(r_t | š_t, ğ)$

4. $N$ Next-Most Intermediate Env/Goal/Action Model: $ǎ^{int/nxt} = f(š, ğ/ǎ^{int})$

5. $I$ - Immediate-Env/Goal/Action Model: $ǎ_t^{nxt} = f(š_t, ğ)$

6. $A$ - Action Decoder: $a_t = f(ǎ_t)$

7. $V$ - Value Model: $v_{š/ǎ} = f(š/ǎ, ğ)$

8. $S_{txt}$ - Text Decoder(?): $o/g/a_{txt} = f(š,ğ,ǎ)$
    - For interpretability of intermediate reasoning concepts later

8. $S_{img}$ - Img Decoder(?): $o/g/a_{img} = f(š,ğ,ǎ)$

### Training signals:

(likely used in some ordered/interleaved-update strategy)

#### Pre-pre-train:

1. $V$: init to assign high-ish value to $ǎ_{nxt}$ from expert trajectories

#### Pre-train:
- Can order training epochs from high-p(bad)- -> high-p(good)- quality datasets
- Can interleave later RL to keep policy close to outputting chosen supervised data

<br>

1. $N$: IMITATION - $p(\text{halt})$ & $y$ from LRXC and other hierarchical plan/COT-annotated data
2. $E$, $T$, $O$: All observed transitions with no specified action ($q(š_t | š_{t-1}, 0)$)
3. $E$, $T$, $O$: All observed transitions where the action can be described in text and encoded ($q(š_t | š_{t-1}, ǎ_{t-1})$)
4. $N$: IMITATION - All expert-trajectory data w/ only $ǎ_{nxt}$ and $g$ (encoded from language)
5. $I$, $A_{env}$: IMITATION - All expert trajectory data w/ only $g$ as well as $a$ _or_ $a$ & $ǎ_{nxt}$ (better)
6. $I$, $A_{env}$, $R$, $V$: IMITATION+ - All expert trajectory data w/ known rewards

#### "Cherry-on-top" RL optimization/generalization to new envs:

1. $\text{ALL}\%\text{frzn}$: DreamerV3 RL w/ imagined rollouts to get loss for all $I$ halt-levels

___

Verbiage(?):
- Ponder&Dream: Recipe For Creating Foundation Goal/Action Policies (Model) For Sample-Efficient Learning Of New Environments

___

Subprojects/research questions:

1. Extracting hierarchical strategic thinking data from youtube videos (ARR workshop?)
2. Pretraining PonderNet-style Goal/Action w/ hierarchical strategic thinking data gets us farther in Dreamer-Style RL (Pokemon first, then generalize to Crafter w/ text-heuristic-$ǎ_{int}$)
3. Recipe for creating generalist/foundation goal/action policies w/ implicit/latent COT and learned latent space for online (inference-time), gradient-based plan optimization (à la [navigation world models](https://arxiv.org/abs/2412.03572))
4. OURMODELV1: generalist/foundation latent goal/action policy

___

Pitch:

Hypothesis 1:
- Extracting (clean) hierarchical strategic thinking data from youtube videos will be useful (specifically, for learning "hierarchical planning")

Hypothesis 2:
- Pretraining PonderNet-style Goal/Action w/ hierarchical strategic thinking data can get us farther in same-env Dreamer-Style RL

Hypothesis 3:
- Pretraining PonderNet-style Goal/Action w/ hierarchical strategic thinking data can get us farther in different-env Dreamer-Style RL

Hypothesis 4:
- Because 3, leveraging this will be an advantageous solution to training a generalist/foundation latent goal/action policy

___

NOTE: 
- Notice how nothing appeals to learning tasks in a more memory-efficient way. That is already/given proposed by by ACT/PonderNet. The crux is, (remember the addage, smallest model possible is usually the most generalizing), the fewer the parameters (by learning abstract(?) recursible sub-programs for getting desired output) the more generalizing (I.e, BETTER/USEFUL) the (generalist) policy will be.

___

... Leverage JEPA learning algo somehow for predicting in latent space?