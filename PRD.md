# PRD：基于 MiniMind 的结构感知 Token Superposition 预训练实验

## 0. 文档元信息

**项目名称：** 基于 MiniMind 的结构感知 Token Superposition 预训练实验  
**英文简称：** HST-MiniMind  
**用途：** 用于 CV / 申请材料的小型、可复现、可解释 LLM 预训练研究项目  
**运行方式：** 本机仅做代码验证；远程机器负责真实训练  
**基线仓库：** `jingyaogong/minimind` 或用户本地已有 MiniMind 克隆  
**主基线：** MiniMind 预训练流程，重点参考 `trainer/train_pretrain.py`  
**实验类型：** 预训练目标与 embedding 组合方式实验，不涉及 SFT、RLHF、DPO、PPO、GRPO

---

## 1. 一句话目标

在 MiniMind 预训练框架中实现一种 **结构感知的 superposed embedding**：把相邻 token 压缩成较少训练位置，同时尽量保留 token 局部顺序、边界类型和层级块结构，并验证它是否相比普通 NTP 和 vanilla Token Superposition Training 具有更快 loss 收敛、更小 recovery gap，或更短的 NTP recovery 需求。

---

## 2. 研究动机

Vanilla Token Superposition Training 会把相邻 token 的 embedding 压缩成一个 super-token。最简单的做法是对多个 token embedding 求平均。这种做法虽然降低了序列长度，但会损失局部顺序和结构信息。

本项目的核心假设是：

> Token superposition 训练后的 recovery phase 之所以需要较长时间，部分原因是简单平均丢失了 token-level 局部结构。若在 superposed embedding 中加入顺序、边界和层级结构信息，模型可能能更快恢复到标准 next-token prediction 表现，并减少 recovery 阶段的训练成本。

该项目目标不是做 SOTA，而是完成一个小型、可复现、实验逻辑清楚的预训练研究项目。

---

## 3. 最终产物

Coding agent 需要产出一个干净、隔离、可运行的实验实现，支持两种模式。

### 3.1 本机验证模式

本机只用于验证代码正确性，不做正式训练。

要求：

- 可以在 CPU 或单张小 GPU 上运行。
- 使用 tiny synthetic dataset 或从数据集中切出来的小样本。
- 只运行极少步数。
- 验证 tensor shape、loss、checkpoint 路径、日志输出、recovery phase 切换是否正常。
- 不允许在本机启动真实训练。

### 3.2 远程训练模式

远程机器用于正式训练。

要求：

- 使用 MiniMind 预训练数据。
- 所有日志、checkpoint、metric 和实验结果只保存到项目目录下。
- 支持安全断点续训。
- 不修改公共机器上的共享目录、系统目录、全局环境或其他用户文件。

---

## 4. 公用机器安全约束

由于本机和远程机器都是公用机器，以下要求是硬约束。

### 4.1 文件系统边界

Coding agent 只能在以下项目自有目录下创建、修改或删除文件：

```bash
$PROJECT_ROOT
$PROJECT_ROOT/hst_experiments
$PROJECT_ROOT/hst_logs
$PROJECT_ROOT/hst_runs
$PROJECT_ROOT/hst_checkpoints
$PROJECT_ROOT/hst_outputs
$PROJECT_ROOT/hst_tmp
$PROJECT_ROOT/dataset        # 仅当数据被明确放在这里时允许使用
```

如果环境变量未设置，默认使用：

```bash
PROJECT_ROOT=$(pwd)
RUN_ROOT=$PROJECT_ROOT/hst_runs
LOG_ROOT=$PROJECT_ROOT/hst_logs
CKPT_ROOT=$PROJECT_ROOT/hst_checkpoints
OUT_ROOT=$PROJECT_ROOT/hst_outputs
TMP_ROOT=$PROJECT_ROOT/hst_tmp
```

禁止写入或修改：

```bash
/
/home
/home/*，除非是当前项目目录
/root
/etc
/usr
/usr/local
/opt
/tmp，除非明确改用 $PROJECT_ROOT/hst_tmp
~/.ssh
~/.cache，除非无法避免且必须在日志中明确说明
~/.bashrc
~/.zshrc
~/.profile
~/.condarc
任何 $PROJECT_ROOT 之外的目录，除非它是只读数据路径
```

### 4.2 禁止修改全局环境

禁止执行：

```bash
sudo ...
apt install ...
yum install ...
brew install ...
在非本地虚拟环境中 pip install ...
向 conda base 环境安装包
rm -rf /...
rm -rf ~...
对共享目录执行 chmod / chown
修改 shell 启动文件
```

允许的依赖安装方式：

```bash
conda create xxx
pip install -r requirements.txt
```
(本地仅debug，如果需要环境可以用tiny-jepa-debug)
如果使用已有 conda 环境，必须先打印其路径，并确认不是 `base` 环境。

### 4.3 数据集安全

Coding agent 不得修改共享数据目录。

如果数据集已存在于共享路径，只能通过只读参数引用：

```bash
--data_path /path/to/shared/pretrain_t2t_mini.jsonl
```

如果需要下载数据，只能下载到：

```bash
$PROJECT_ROOT/dataset
```

禁止删除、改写、格式化或清洗原始数据文件。任何派生小样本都必须保存到：

```bash
$PROJECT_ROOT/hst_tmp
```

### 4.4 Git 安全

允许新建分支和新增项目文件：

```bash
git checkout -b hst-structure-aware-superposition
```

禁止：

```bash
git reset --hard
git clean -fdx
git push --force
```

### 4.5 训练安全

本机只允许运行验证任务：

```bash
--max_steps <= 20
--dry_run 1 或 --debug 1
```

远程训练脚本必须强制要求输出路径位于：

```bash
$PROJECT_ROOT/hst_runs/...
```

---

## 5. 非目标

本项目不做以下内容：

1. 不实现新 tokenizer。
2. 不重写一个与 MiniMind 无关的新 Transformer 架构。
3. 不做 SFT、DPO、PPO、GRPO、RLHF。
4. 不做大规模超参数搜索。
5. 不做新的数据清洗 pipeline。
6. 不写任何默认独占机器资源的代码。
7. 不自动删除当前实验目录之外的 checkpoint 或日志。

---

## 6. 必须支持的实验变体

所有方法必须能通过 CLI 或 config 选择。

### 6.1 `ntp_baseline`

标准 MiniMind next-token prediction。

用途：

- 验证与现有 MiniMind pipeline 兼容。
- 提供 validation loss 基线。

行为：

- 输入输出与普通自回归 LM 相同。
- 不做 token grouping。
- 不需要 recovery phase。

### 6.2 `vanilla_tst`

普通 Token Superposition Training。

用途：

- 作为 superposition baseline。

行为：

- 把相邻 token 按固定大小 `superpose_size = s` 分组。
- 每组 token embedding 用 mean pooling 压成一个 superposed embedding。
- 用当前 chunk 预测下一个 chunk。
- 支持后续切回普通 NTP recovery。

### 6.3 `order_aware_tst`

顺序感知 superposition。

用途：

- 验证保留 chunk 内 slot 顺序是否能减少信息损失。

组合方式：

```python
z_j = sum_r gate[r] * token_embedding[x_{j,r}]
```

实现优先级：

1. 最小实现：给每个 slot 加 learned slot embedding，然后 pooling。
2. 更好实现：每个 slot 使用 learned diagonal gate。
3. 除非显存允许，不优先使用每个 slot 一个 full dense projection。

必须支持参数：

```bash
--superpose_mode order_aware
--superpose_size 2|4
--slot_gate_type embedding|diagonal
```

### 6.4 `boundary_aware_tst`

边界 / 类型感知 superposition。

用途：

- 保留标点、换行、数字、英文、中文、括号、代码符号等轻量结构信息。

每个 token 需要被分类到以下类型之一：

```text
normal
punctuation
newline
digit
latin
cjk
bracket
code_symbol
whitespace_like
special_token
unknown
```

组合方式：

```python
x_embed = token_embed[token_id] + type_embed[type_id]
z_j = pool(x_embed within chunk)
```

要求：

- token type id 计算后必须缓存。
- 不得修改 tokenizer 文件。
- decode 失败时必须有 fallback 分类逻辑。

### 6.5 `hierarchical_tst`

层级结构感知 superposition。

用途：

- 验证 block-level summary 是否能改善收敛和 recovery。

组合方式：

```python
local_z_j = order_or_boundary_aware_superpose(tokens_j)
block_h_k = mean(local_z_j in block k) + block_type_embedding[k]
input_z_j = local_z_j + alpha * block_h_k
```

block 划分方式：

```bash
--block_mode fixed        # 每 K 个 chunk 一个 block
--block_mode newline      # 尽量按换行切分
--block_mode punctuation  # 尽量按标点切分
```

MVP 默认：

```bash
--block_mode fixed
--chunks_per_block 8
--hier_alpha 0.1
```

---

## 7. Recovery Phase 设计

Recovery 长度是实验变量，而不是固定假设。

### 7.1 Full Recovery

```text
superposition phase: 30% total steps
NTP recovery phase: 70% total steps
```

### 7.2 Short Recovery

```text
superposition phase: 80% total steps
NTP recovery phase: 20% total steps
```

### 7.3 No Recovery

```text
superposition phase: 100% total steps
NTP recovery phase: 0% total steps
```

No recovery 只是探索性实验。除非使用 ordered slot prediction 或 micro-decoder，否则它不能被解释成等价于普通自回归 LLM。

---

## 8. 预测头与 Loss

至少支持以下 loss。

### 8.1 标准 NTP Loss

用于 `ntp_baseline` 和 recovery phase。

```python
loss = cross_entropy(logits[:, :-1], input_ids[:, 1:])
```

### 8.2 Repeated-token Chunk CE Loss

用于 `vanilla_tst` MVP。

对每个 chunk hidden state `h_j`，目标是下一个 chunk 中的 token。

为简单稳定，优先实现：

```python
loss = mean_r CE(head(h_j), target_token_{j+1,r})
```

这不是严格的 multi-hot BCE，而是 repeated-token CE。它更容易在 MiniMind 规模上稳定运行，也更容易 debug。

### 8.3 Ordered Slot Loss

用于 `order_aware_tst` 和 no-recovery 实验。

```python
logits_r = head_r(h_j)
loss = mean_r CE(logits_r, target_token_{j+1,r})
```

实现建议：

- 不要为每个 slot 建一个巨大 vocab head。
- 使用共享 LM head，加 output slot embedding。

```python
slot_hidden_r = h_j + out_slot_embed[r]
logits_r = lm_head(slot_hidden_r)
```

---

## 9. CLI 要求

新增训练入口，不要破坏 MiniMind 原有预训练脚本。

推荐新增文件：

```text
trainer/train_hst_pretrain.py
```

示例命令：

```bash
python trainer/train_hst_pretrain.py \
  --method hierarchical_tst \
  --data_path ./dataset/pretrain_t2t_mini.jsonl \
  --run_name hst_s4_short_recovery \
  --output_dir ./hst_runs/hst_s4_short_recovery \
  --superpose_size 4 \
  --superpose_mode hierarchical \
  --loss_mode ordered_slot \
  --recovery_ratio 0.2 \
  --max_steps 10000 \
  --eval_interval 200 \
  --save_interval 1000
```

必须支持参数：

```text
--method: ntp_baseline | vanilla_tst | order_aware_tst | boundary_aware_tst | hierarchical_tst
--data_path
--run_name
--output_dir
--max_steps
--eval_interval
--save_interval
--seed
--superpose_size
--superpose_mode
--loss_mode
--recovery_ratio
--learning_rate
--batch_size
--max_seq_len
--device
--dry_run
--debug
--from_resume
```

可选参数：

```text
--block_mode
--chunks_per_block
--hier_alpha
--slot_gate_type
--type_vocab_size
--log_jsonl
--use_wandb
--use_swanlab
```

必须进行路径安全检查：

```python
assert output_dir.resolve().is_relative_to(project_root.resolve())
```

如果 Python 版本不支持 `Path.is_relative_to`，必须手写安全 fallback。

---

## 10. 推荐文件结构

优先做 additive changes，不要破坏原仓库。

推荐新增文件：

```text
trainer/train_hst_pretrain.py
model/hst_superposition.py
model/hst_losses.py
model/hst_token_types.py
scripts/hst_make_tiny_dataset.py
scripts/hst_local_verify.sh
scripts/hst_remote_train.sh
scripts/hst_collect_metrics.py
scripts/hst_eval_probes.py
configs/hst/ntp_baseline_debug.yaml
configs/hst/vanilla_tst_s2_debug.yaml
configs/hst/order_aware_s2_debug.yaml
configs/hst/boundary_aware_s2_debug.yaml
configs/hst/hierarchical_s4_debug.yaml
configs/hst/remote_hst_s4_short_recovery.yaml
tests/test_hst_shapes.py
tests/test_hst_losses.py
tests/test_hst_path_safety.py
README_HST.md
```

允许少量修改现有文件：

```text
只允许为兼容 import 或 inputs_embeds 添加可选路径
不得重写原有 MiniMind 训练逻辑
```

除非绝对必要，不要修改 `train_pretrain.py`。如必须修改，必须保证默认行为完全不变。

---

## 11. 本机验证要求

本机验证必须快速、安全、可重复。

### 11.1 Tiny Dataset 生成

新增脚本：

```bash
python scripts/hst_make_tiny_dataset.py \
  --output ./hst_tmp/tiny_pretrain.jsonl \
  --num_examples 128
```

格式：

```json
{"text": "..."}
```

数据内容应包含：

- 中文句子。
- 英文句子。
- 数字和类数学表达式。
- 代码片段。
- 标点密集样本。
- 换行样本。

### 11.2 Smoke Test 脚本

新增脚本：

```bash
bash scripts/hst_local_verify.sh
```

该脚本必须运行：

```bash
python trainer/train_hst_pretrain.py --method ntp_baseline --dry_run 1 --max_steps 3 ...
python trainer/train_hst_pretrain.py --method vanilla_tst --dry_run 1 --max_steps 3 --superpose_size 2 ...
python trainer/train_hst_pretrain.py --method order_aware_tst --dry_run 1 --max_steps 3 --superpose_size 2 ...
python trainer/train_hst_pretrain.py --method boundary_aware_tst --dry_run 1 --max_steps 3 --superpose_size 2 ...
python trainer/train_hst_pretrain.py --method hierarchical_tst --dry_run 1 --max_steps 3 --superpose_size 4 ...
```

验收标准：

- 所有方法至少完成 3 个 train step。
- loss 是有限值。
- 输出 shape 符合预期。
- 日志只写入 `./hst_runs/...`。
- 不在 `$PROJECT_ROOT` 外创建文件。
- tiny run 中能触发 recovery phase 切换。

---

## 12. 远程训练要求

正式训练只在远程机器上进行。

### 12.1 远程 run 目录

每个 run 必须使用：

```text
$PROJECT_ROOT/hst_runs/{timestamp}_{run_name}/
```

目录内部结构：

```text
config.yaml
metrics.jsonl
stdout.log
stderr.log
checkpoints/
outputs/
plots/
artifacts/
```

### 12.2 远程训练脚本

新增脚本：

```bash
bash scripts/hst_remote_train.sh configs/hst/remote_hst_s4_short_recovery.yaml
```

脚本必须：

- 打印当前 host、GPU 信息、Python 路径、项目根目录。
- 如果 `PROJECT_ROOT` 为空或等于 `/`，必须拒绝运行。
- 如果输出目录不在 `$PROJECT_ROOT/hst_runs` 下，必须拒绝运行。
- 只创建当前 run 目录及其子目录。
- 支持从 checkpoint resume。
- 只有设置 `NPROC_PER_NODE` 时才使用 `torchrun`。

伪逻辑：

```bash
set -euo pipefail
cd "$PROJECT_ROOT"
mkdir -p "$RUN_DIR"
python - <<'PY'
# 执行路径安全检查
PY
python trainer/train_hst_pretrain.py --config "$CONFIG" --output_dir "$RUN_DIR"
```

### 12.3 推荐首批远程实验

先跑小中规模实验，确认稳定后再延长训练。

```text
R0: ntp_baseline, max_steps=2000
R1: vanilla_tst, s=2, recovery_ratio=0.7, max_steps=2000
R2: order_aware_tst, s=2, recovery_ratio=0.7, max_steps=2000
R3: boundary_aware_tst, s=2, recovery_ratio=0.7, max_steps=2000
R4: hierarchical_tst, s=4, recovery_ratio=0.2, max_steps=2000
R5: hierarchical_tst, s=4, recovery_ratio=0.0, max_steps=2000
```

---

## 13. Metrics 与日志

每个训练 run 必须向以下文件写入 JSONL 日志：

```text
metrics.jsonl
```

每条记录至少包含：

```json
{
  "time": "ISO-8601 timestamp",
  "run_name": "...",
  "method": "hierarchical_tst",
  "step": 100,
  "phase": "superposition|recovery|ntp",
  "loss_train": 3.14,
  "loss_eval": 3.20,
  "lr": 0.0003,
  "tokens_seen": 123456,
  "effective_tokens_seen": 123456,
  "superpose_size": 4,
  "recovery_ratio": 0.2,
  "wall_time_sec": 120.5,
  "gpu_mem_gb": 10.2
}
```

`scripts/hst_collect_metrics.py` 必须统计：

1. final eval loss。
2. best eval loss。
3. time-to-loss threshold。
4. 相同 wall-clock 下的 loss。
5. 相同 token budget 下的 loss。
6. recovery gap：

```text
recovery_gap = first_ntp_eval_loss_after_switch - last_superposition_eval_loss_before_switch
```

7. 从 superposition 切到 NTP 后的 loss jump。
8. tokens/sec。
9. examples/sec。
10. 可用时记录 GPU memory usage。

---

## 14. Probe 评估

训练后实现轻量 probe。

### 14.1 顺序敏感性 Probe

构造样本：

```text
AB vs BA
北京 到 上海 vs 上海 到 北京
x = y + 1 vs y = x + 1
if a: b() vs b(): if a
```

目标：

- 比较模型对顺序扰动的 loss 或 perplexity。
- 验证 order-aware 版本是否比 vanilla mean-pooling TST 更能区分顺序。

### 14.2 边界敏感性 Probe

构造样本：

```text
function call(args)
function call args
中文，标点影响语义。
中文 标点 影响 语义
line1\nline2
line1 line2
```

目标：

- 验证 boundary-aware embedding 是否对标点、换行、代码结构更敏感。

### 14.3 Recovery Probe

在以下时刻评估同一模型：

- superposition phase 结束。
- 切换到 NTP 后的第一个 eval。
- short recovery 后。
- 训练结束。

输出：

```text
probe_results.jsonl
```

---

## 15. 实现细节

### 15.1 Superposition Composer 接口

新增模块，概念接口如下：

```python
class SuperpositionComposer(nn.Module):
    def __init__(self, token_embedding, hidden_size, vocab_size, config):
        ...

    def compose(self, input_ids: torch.Tensor) -> dict:
        """
        Args:
            input_ids: [batch, seq_len]
        Returns:
            {
              "inputs_embeds": Tensor[batch, chunk_len, hidden],
              "chunk_targets": Tensor[batch, chunk_len, superpose_size],
              "attention_mask": Optional[Tensor],
              "metadata": dict
            }
        """
```

必须支持：

```text
mean
order_aware
boundary_aware
hierarchical
```

### 15.2 Sequence Length 处理

如果 `seq_len` 不能被 `superpose_size` 整除，则在 superposition phase 中裁掉尾部多余 token。不要通过 padding 生成虚假 target，除非 MiniMind 原有 collator 已经安全处理 padding。

示例：

```python
usable_len = (seq_len // s) * s
input_ids = input_ids[:, :usable_len]
```

对于 next-chunk prediction：

```text
num_train_chunks = num_chunks - 1
```

### 15.3 Phase Switching

训练循环按 global step 决定 phase：

```python
recovery_start_step = int(max_steps * (1.0 - recovery_ratio))
if method == "ntp_baseline":
    phase = "ntp"
elif step < recovery_start_step:
    phase = "superposition"
else:
    phase = "recovery"
```

Recovery phase 中：

- 使用标准 token embedding。
- 使用标准 NTP loss。
- 保持 superposition phase 训练后的模型权重。
- 默认不重新初始化 optimizer，除非 config 显式要求。

### 15.4 MiniMind 模型集成

优先策略：

- 尽量复用 MiniMind 原模型。
- 如果模型 forward 支持 `inputs_embeds`，直接使用。
- 如果不支持，添加最小可选路径，并保持原 `input_ids` 行为不变。

必须满足：

```python
model(input_ids=...)       # 原始路径行为不变
model(inputs_embeds=...)   # 新增可选路径
```

如果 MiniMind 内部不适合直接改，优先写 wrapper model，不要大改 backbone。

### 15.5 Token Type Classifier

新增确定性 token 类型分类器：

```python
def classify_token_text(text: str) -> int:
    ...
```

类别：

```python
TOKEN_TYPE_NORMAL = 0
TOKEN_TYPE_PUNCT = 1
TOKEN_TYPE_NEWLINE = 2
TOKEN_TYPE_DIGIT = 3
TOKEN_TYPE_LATIN = 4
TOKEN_TYPE_CJK = 5
TOKEN_TYPE_BRACKET = 6
TOKEN_TYPE_CODE_SYMBOL = 7
TOKEN_TYPE_WHITESPACE = 8
TOKEN_TYPE_SPECIAL = 9
TOKEN_TYPE_UNKNOWN = 10
```

分类器必须确定性可复现，并有测试覆盖。

---

## 16. 配置示例

### 16.1 本机 Debug：Hierarchical TST

```yaml
method: hierarchical_tst
run_name: debug_hierarchical_s4
output_dir: ./hst_runs/debug_hierarchical_s4
data_path: ./hst_tmp/tiny_pretrain.jsonl
max_steps: 3
eval_interval: 1
save_interval: 100
batch_size: 2
max_seq_len: 128
learning_rate: 0.0003
seed: 42
superpose_size: 4
superpose_mode: hierarchical
loss_mode: ordered_slot
recovery_ratio: 0.2
block_mode: fixed
chunks_per_block: 4
hier_alpha: 0.1
dry_run: 1
debug: 1
```

### 16.2 远程主实验：Short Recovery

```yaml
method: hierarchical_tst
run_name: hst_s4_short_recovery
output_dir: ./hst_runs/hst_s4_short_recovery
data_path: ./dataset/pretrain_t2t_mini.jsonl
max_steps: 10000
eval_interval: 200
save_interval: 1000
batch_size: 16
max_seq_len: 768
learning_rate: 0.0003
seed: 42
superpose_size: 4
superpose_mode: hierarchical
loss_mode: ordered_slot
recovery_ratio: 0.2
block_mode: fixed
chunks_per_block: 8
hier_alpha: 0.1
dry_run: 0
debug: 0
from_resume: 1
```

---

## 17. 验收标准

### 17.1 代码验收

实现被接受的条件：

- 新功能隔离在新文件或可选代码路径中。
- MiniMind 原始预训练默认行为不变。
- path safety 测试通过。
- 所有 superposition mode 的 shape 测试通过。
- loss 测试能得到有限 scalar loss。
- 本机 smoke test 通过。
- 没有任何输出写到项目目录之外。

### 17.2 实验验收

实验包必须能产出：

- 每个 run 的 `metrics.jsonl`。
- 至少一个 NTP baseline run。
- 至少一个 vanilla TST run。
- 至少一个 structure-aware TST run。
- recovery gap 测量。
- 汇总表，包含 final eval loss、best eval loss、time-to-loss、recovery gap。

### 17.3 研究验收

项目至少应回答以下问题之一：

1. Vanilla superposition 是否减少 wall-clock 成本，但增加 recovery gap？
2. Order-aware superposition 是否比 vanilla mean pooling 有更小 recovery gap？
3. Boundary-aware superposition 是否在中文、代码、标点密集 probe 上更好？
4. Hierarchical TST + short recovery 是否能接近或超过 vanilla TST + full recovery？
5. No recovery 失败的方式是否支持“token-level recovery 仍然必要”的解释？

负结果可以接受，但实验必须干净、可复现，并能解释为什么假设没有成立。

---

## 18. Coding Agent 里程碑

### Milestone 1：仓库检查

任务：

- 找到 MiniMind 模型类。
- 找到 tokenizer 加载代码。
- 找到 dataset 加载代码。
- 找到现有 pretraining loop。
- 确认当前 loss 计算方式。
- 暂时不修改代码。

交付物：

```text
hst_experiments/repo_inspection.md
```

### Milestone 2：路径安全工具

任务：

- 实现项目路径边界检查。
- 添加测试。

交付物：

```text
utils/hst_path_safety.py 或等价文件
tests/test_hst_path_safety.py
```

### Milestone 3：Tiny Dataset 与 Smoke Script

任务：

- 实现 tiny dataset 生成脚本。
- 实现本机验证脚本。

交付物：

```text
scripts/hst_make_tiny_dataset.py
scripts/hst_local_verify.sh
```

### Milestone 4：Superposition Composer

任务：

- 实现 mean、order-aware、boundary-aware、hierarchical composition。
- 添加 shape tests。

交付物：

```text
model/hst_superposition.py
tests/test_hst_shapes.py
```

### Milestone 5：Losses

任务：

- 实现 NTP、repeated-token CE、ordered slot loss。
- 添加 finite-loss tests。

交付物：

```text
model/hst_losses.py
tests/test_hst_losses.py
```

### Milestone 6：训练入口

任务：

- 实现 `trainer/train_hst_pretrain.py`。
- 支持 CLI / config。
- 支持 phase switching。
- 支持 JSONL logging。

交付物：

```text
trainer/train_hst_pretrain.py
configs/hst/*.yaml
```

### Milestone 7：远程训练脚本

任务：

- 实现带安全检查的远程训练启动脚本。

交付物：

```text
scripts/hst_remote_train.sh
```

### Milestone 8：Metric 汇总

任务：

- 实现 metric 汇总脚本。
- 生成 CSV / Markdown summary。

交付物：

```text
scripts/hst_collect_metrics.py
hst_outputs/summary.md
hst_outputs/summary.csv
```

### Milestone 9：Probe 评估

任务：

- 实现顺序、边界、recovery probes。

交付物：

```text
scripts/hst_eval_probes.py
hst_outputs/probe_results.jsonl
```

---

## 19. 实验报告模板

远程实验完成后生成：

```text
hst_outputs/report.md
```

报告结构：

```markdown
# 基于 MiniMind 的结构感知 Token Superposition 预训练实验

## 摘要

## 研究假设

## 方法
- NTP baseline
- Vanilla TST
- Order-aware TST
- Boundary-aware TST
- Hierarchical TST

## 实验设置
- 模型
- 数据集
- 硬件
- 训练步数
- Recovery ratio
- 评估协议

## 结果
| Method | s | Recovery Ratio | Best Eval Loss | Final Eval Loss | Recovery Gap | Time-to-Loss |

## Probe Results

## Discussion

## Limitations

## Reproducibility Checklist
```

---

## 20. 最小可行实现

如果时间有限，优先只实现：

1. `vanilla_tst`，使用 mean pooling。
2. `order_aware_tst`，使用 learned slot embedding。
3. `hierarchical_tst`，使用 fixed block mean。
4. repeated-token CE loss。
5. short recovery schedule。
6. 本机 smoke test。
7. 远程训练脚本。
8. metrics JSONL。

暂时跳过：

- 真正的 BCE multi-hot loss。
- newline / punctuation dynamic block。
- micro-decoder。
- WandB / SwanLab 集成。
- 大型 probe suite。

---

## 21. 代码风格要求

- 尽量使用明确 type hints。
- 优先小函数，不写巨型脚本。
- 尽量保持 MiniMind 原有代码风格。
- 只有在逻辑不明显时添加注释。
- 所有随机行为都必须可设 seed。
- run 开始时打印 config。
- 每个 run 目录中保存一份 config copy。
- invalid path 或 invalid config 必须 fail fast。
- 如果 output path 验证失败，不允许静默 fallback 到当前目录。

---

## 22. 给 Coding Agent 的最终指令

请增量实现该项目。任何写操作之前，必须确认目标路径位于 `$PROJECT_ROOT` 内。不要修改系统配置、shell 启动文件、共享数据目录或项目目录之外的任何路径。本机只用于 tiny dataset 与少量 step 的代码验证；正式训练只能通过远程训练脚本启动，并且必须显式写入 `$PROJECT_ROOT/hst_runs` 下的 run 目录。

