# ReasoningVQA

ReasoningVQA 的公开代码框架：一个 **配置驱动、可复现** 的多跳推理 VQA 训练 / 评测流水线。

- 数据源：官方 [ReasonVQA](https://duong-tr.github.io/ReasonVQA/)（ICCV 2025）数据集，按来源拆分为 iNaturalist / GLDv2 / Visual Genome 三个子集。
- 数据协议：所有源数据先被 source adapters 归一化到统一的 `reasoningvqa.canonical.v1` manifest，再进入共享的训练 / 评测代码。
- 训练：Qwen-VL 系列（Qwen2.5-VL / Qwen3-VL，兼容 MiniCPM-V）的 BF16 MM-LoRA SFT，response-only 交叉熵，支持 language-only LoRA 与多卡 `device_map="balanced"`。
- 评测：生成 + 确定性评分（canonical open/json 两种协议，或 structured hybrid-100 协议），并带独立的 re-score 校验。
- 额外提供纯评分 harness（`scoring/`），只消费已生成的预测，不进行推理。

本仓库只包含代码、配置、评分脚本、工具和文档；**不包含**模型权重、原始数据集、conda 环境和实验产物（详见 `.gitignore`）。

---

## 1. 文件树

```text
ReasoningVQA/
├── README.md
├── .gitignore
└── rvqa_bundle/
    ├── MANIFEST.txt                    # bundle 构建来源与 sha256 溯源
    ├── pipeline/                       # 核心：配置驱动的 prepare/train/evaluate 流水线
    │   ├── README.md
    │   ├── run_pipeline.py             # 入口：加载配置、解析动作、渲染并执行命令
    │   ├── config_schema.py            # 配置结构校验
    │   ├── configs/
    │   │   ├── pipeline.template.yaml  # 便携三源起点模板（无硬编码路径）
    │   │   └── example_pipeline.yaml   # 结构化 GLDv2 冻结示例
    │   ├── model_backends/             # 模型族操作注册表
    │   │   ├── __init__.py
    │   │   ├── registry.py             # backend 注册
    │   │   ├── adapter.py              # BackendAdapter 契约
    │   │   ├── common.py               # require / resolve_script
    │   │   └── qwen_vl.py              # qwen_vl backend：train/evaluate 操作映射
    │   ├── pipelines/                  # 动作命令构建器 + 阶段脚本
    │   │   ├── __init__.py             # BUILDERS = {prepare, train, evaluate}
    │   │   ├── common.py               # Command / build_action_commands / execute
    │   │   ├── data.py                 # prepare 的 canonical 命令构建
    │   │   ├── training.py             # train 命令构建
    │   │   ├── evaluation.py           # evaluate 命令构建
    │   │   ├── prepare/                # 数据准备阶段
    │   │   │   ├── __init__.py
    │   │   │   ├── adapt_source.py     # 单源 JSONL → canonical manifest
    │   │   │   ├── validate_canonical_manifest.py   # 独立结构校验
    │   │   │   ├── compose_canonical_manifests.py   # 多源确定性合并
    │   │   │   ├── canonical_schema.py              # canonical 行契约
    │   │   │   ├── build_structured_reasoning.py    # 结构化 20-root 课程构建
    │   │   │   ├── validate_structured_reasoning.py # 独立校验结构化课程
    │   │   │   └── source_adapters/     # 源归一化边界
    │   │   │       ├── __init__.py
    │   │   │       ├── registry.py      # adapter 注册
    │   │   │       ├── common.py        # canonical_row 共享实现
    │   │   │       ├── inaturalist.py
    │   │   │       ├── gldv2.py
    │   │   │       └── visual_genome.py
    │   │   ├── train/
    │   │   │   ├── __init__.py
    │   │   │   └── train_qwen_vl_sft.py # BF16 Qwen-VL MM-LoRA SFT
    │   │   └── evaluate/
    │   │       ├── __init__.py
    │   │       ├── evaluate_canonical_reasoning.py    # canonical 生成 + 评分
    │   │       ├── evaluate_structured_reasoning.py   # structured 生成 + hybrid-100 评分
    │   │       ├── validate_canonical_predictions.py  # canonical 独立 re-score
    │   │       └── validate_predictions.py            # structured 独立 re-score
    │   ├── tools/
    │   │   ├── build_model_inventory.py  # 从 safetensors header 构建 LoRA target inventory
    │   │   ├── verify_model_inventory.py # 校验 inventory 与模型契约
    │   │   └── verify_lora_scope_smoke.py# CPU-only 冒烟测试 lora-scope
    │   └── tests/
    │       ├── test_pipeline_contracts.py
    │       ├── test_multi_gpu_device_plan.py
    │       ├── test_source_adapters.py
    │       └── test_validation_helpers.py
    ├── configs/                         # 实验 runbook / 运行配置
    │   ├── README.md
    │   ├── vg-qwen25vl7b.yaml
    │   ├── vg-qwen25vl32b.yaml          # 占位：>=3 卡才可运行
    │   ├── vg-qwen3vl32b-awq.yaml       # 占位：AWQ 32B 需 >=2 卡
    │   ├── vg-minicpmv26.yaml           # 占位：chat-template 阻塞
    │   └── inat-qwen25vl7b.yaml
    ├── scoring/                         # 纯评分 harness（只读预测，不做推理）
    │   ├── README.md
    │   ├── score_predictions.py         # 主评分脚本
    │   ├── validate_semantic_against_history.py
    │   ├── semantic_history_validation.json
    │   ├── synthetic_predictions.jsonl  # 自测样例
    │   ├── synthetic_predictions.notes
    │   ├── oracle_predictions.inaturalist.jsonl
    │   └── oracle_predictions.visual_genome.jsonl
    ├── scoring_extra/
    │   └── evaluate_reasonvqa_mc_qwen.py # canonical MC 评分脚本（运行时 import 源）
    ├── tools/
    │   └── filter_english_rows.py       # 过滤非英文 / 退化行
    ├── requirements/                    # 各环境依赖
    │   ├── requirements.txt             # CUDA/x86_64 主环境
    │   ├── requirements-qwen3vl32b-awq.txt
    │   ├── requirements-minicpmv26.txt
    │   ├── requirements-ascend*.txt
    │   └── frozen-*.txt                 # 各环境冻结版本
    ├── gldv2/
    │   ├── README.md                    # GLDv2 子集构建说明
    │   └── summary.json                 # 采样 / join 统计
    └── *.sh                             # Ascend / 环境准备脚本（含硬编码源机路径）
```

---

## 2. 调用逻辑

### 2.1 总入口

```bash
python pipeline/run_pipeline.py \
  --config <config.yaml> \
  {prepare|train|evaluate|all} \
  [--dry-run] [--print-command]
```

`run_pipeline.py` 的职责：

1. `load_config(path)`：`yaml.safe_load`。
2. `validate_config(config)`：
   - `config_schema.validate_pipeline_config()` 做结构校验；
   - 通过 `model.backend` 从 `model_backends` 取得 backend；
   - `backend.context(config)` 生成占位符上下文。
3. `freeze_config()`：非 dry-run 时，把配置原样复制到
   `<output.experiment_root>/pipeline_config.yaml`；若目标已存在且内容不同则拒绝覆盖。
4. 对每个 action，调用 `pipelines.build_commands(config, backend, action)` 生成
   `Command` 列表，再逐条 `print` 并 `execute()`（`dry-run` 只打印不执行，
   `--print-command` 只渲染不执行）。

### 2.2 命令构建

`pipelines/__init__.py` 里的 `BUILDERS`：

```text
prepare   -> pipelines.data.build_commands
train     -> pipelines.training.build_commands
evaluate  -> pipelines.evaluation.build_commands
```

#### prepare 两条路径

- **canonical 路径**（配置含 `datasets` 段，推荐）：
  由 `pipelines/data.py` 自动生成命令，无需手写脚本：

  ```text
  对每个 datasets.sources：
      adapt_source.py --adapter <inaturalist|gldv2|visual_genome> ...
      validate_canonical_manifest.py ...
  对每个 datasets.manifests：
      compose_canonical_manifests.py --input <sources> ...
      validate_canonical_manifest.py ...
  ```

- **explicit 路径**（配置无 `datasets`，`example_pipeline.yaml` 的结构化示例）：
  直接执行 `actions.prepare` 中声明的脚本，例如
  `prepare/build_structured_reasoning.py` + `prepare/validate_structured_reasoning.py`。

#### train / evaluate

由 `pipelines/common.build_action_commands()` 展开 `actions.train` /
`actions.evaluate`：

- 每个 step 要么声明 `operation`（由 backend 解析到脚本），要么直接声明 `script`；
- 占位符 `{python} {model} {inventory} {experiment_root} {pipeline_root}` 会被渲染；
- `GRPO` 相关入口会被显式拒绝；
- `script` 只能落在 `source.pipeline_root` 内（防路径逃逸）。

`model_backends/qwen_vl.py` 当前操作映射：

```text
train:
    canonical_sft    -> train/train_qwen_vl_sft.py
    structured_sft   -> train/train_qwen_vl_sft.py
evaluate:
    canonical_generation  -> evaluate/evaluate_canonical_reasoning.py
    structured_generation -> evaluate/evaluate_structured_reasoning.py
```

### 2.3 数据边界

```text
iNaturalist source -> inaturalist adapter ─┐
GLDv2 source      -> gldv2 adapter ────────┼─> reasoningvqa.canonical.v1
Visual Genome     -> visual_genome adapter ─┘            │
                                                          v
                                validate -> compose -> SFT -> generate -> score/validate
```

各 adapter 保留来源特有的 taxonomy / landmark / scene-graph 出处；下游训练与评测
只读取 canonical 字段（`sample_id`、`image_path`、`sft_prompt`、`sft_target`、
`answer` / `answers` / 推理路径等），因此更换数据源或数据集族不会改变训练 / 评测代码。

---

## 3. 核心阶段说明

### 3.1 prepare：source → canonical manifest

| 脚本 | 作用 |
| --- | --- |
| `adapt_source.py` | 读取一个源 JSONL，用指定 adapter 归一化为 canonical 行，create-only 写入输出；生成输入/输出 sha256、行数、split 分布等 summary |
| `validate_canonical_manifest.py` | 独立校验 manifest：schema、非空字段、`answers`/`answer_options`、去重 `sample_id`、source/split/task 计数、可选图片存在性 |
| `compose_canonical_manifests.py` | 合并多个已校验 manifest，跨源去重 `sample_id`，输出 train/dev manifest + composition summary |

`canonical_schema.py` 定义唯一行契约 `reasoningvqa.canonical.v1`，并支持两种 answer mode：

- `open`（默认）：`sft_target` 就是裸答案，可直接 `open_exact` 评分；
- `json`（向后兼容）：`sft_target` 为 `{"answer", "reasoning_path"}`，prompt 带候选答案与 JSON 指令。

### 3.2 train：Qwen-VL MM-LoRA SFT

`train_qwen_vl_sft.py`：

- BF16 + `AutoModelForImageTextToText`（失败回退 `AutoModel` + `trust_remote_code`）；
- PEFT LoRA，`target_modules` 来自 model inventory（`language` / `visual_blocks` / `merger` 三族）；
- `--lora-scope language` 只训语言部分；`all` 训全部三族；
- 响应边界：prompt 部分 label 置 `-100`，只对 response 计算交叉熵；
- 支持 `--gpus 0,1` 多卡 `device_map="balanced"` 与 `--max-memory-per-gpu`；
- 训练结束做参数完整性检查，保存 adapter 并 reload 校验。

### 3.3 evaluate：生成 + 确定性评分 + 独立复评

**canonical 协议**（`evaluate_canonical_reasoning.py`）：

- 三种 mode：`correct-image` / `no-image` / `cyclic-shuffled-image`；
- `open` 模式直接对 raw 输出做归一化 exact-match；`json` 模式解析
  `{answer, reasoning_path}`，同时评 schema / answer / path / full_chain；
- 输出 `predictions.jsonl`、`summary.json`、`contract.json`；
- `validate_canonical_predictions.py` 重新计算分数并逐行比对，发现不一致则失败。

**structured 协议**（`evaluate_structured_reasoning.py`）：

- 面向 20-root 结构化多选题（root / evidence / hops / answer），
  使用 hybrid-100 分项 rubric + rotation 一致性；
- `validate_predictions.py` 独立 re-score。

### 3.4 scoring：纯评分 harness

`scoring/score_predictions.py` 只读预测 + `eval.jsonl`，不做推理，复现历史指标：

`open_exact`、`open_substring`、`fuzzy_proxy`、`mc_accuracy`、
`mc_relaxed_accuracy`、`semantic_accuracy`、`mean_cosine`、
`threshold_sensitivity`，以及按 hop 聚合。语义后端默认 raw transformers + torch
+ `all-MiniLM-L6-v2` 的 attention-mask mean pooling + L2 normalize。

```bash
python scoring/score_predictions.py \
  --eval-jsonl <eval.jsonl> \
  --predictions-jsonl <predictions.jsonl> \
  --output-dir <out-dir> \
  [--hops 1,2,3] [--limit N] [--device cpu] [--batch-size 64]
```

### 3.5 tools

- `pipeline/tools/build_model_inventory.py`：仅读 safetensors header，生成 LoRA
  target inventory（不加载权重，支持 AWQ/GPTQ 量化）。
- `pipeline/tools/verify_model_inventory.py`：校验 inventory 与模型契约。
- `pipeline/tools/verify_lora_scope_smoke.py`：CPU-only 冒烟测试 `--lora-scope`。
- `tools/filter_english_rows.py`：过滤非英文 / 退化行（用于构造英文子集）。

---

## 4. 快速开始

### 4.1 dry-run（无 GPU、无数据也能渲染命令）

```bash
cd rvqa_bundle/pipeline
python run_pipeline.py --config configs/example_pipeline.yaml prepare --dry-run
python run_pipeline.py --config configs/example_pipeline.yaml train --dry-run
python run_pipeline.py --config configs/example_pipeline.yaml evaluate --dry-run
python run_pipeline.py --config configs/example_pipeline.yaml all --dry-run
```

用便携模板渲染三源 canonical 流程：

```bash
python run_pipeline.py --config configs/pipeline.template.yaml all --print-command
```

### 4.2 真实运行（需配置源机路径、模型、inventory、GPU）

以 `rvqa_bundle/configs/vg-qwen25vl7b.yaml` 为例（路径按实际机器修改）：

```bash
cd /path/to/reasoningvqa_bench/pipeline
PY=/path/to/conda_envs/videoespresso-common/bin/python

$PY run_pipeline.py --config /path/to/configs/vg-qwen25vl7b.yaml prepare
CUDA_VISIBLE_DEVICES=8 TRANSFORMERS_OFFLINE=1 \
  $PY run_pipeline.py --config /path/to/configs/vg-qwen25vl7b.yaml train
$PY run_pipeline.py --config /path/to/configs/vg-qwen25vl7b.yaml evaluate
```

### 4.3 运行测试

```bash
cd rvqa_bundle/pipeline
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

---

## 5. 环境与依赖

- 主训练环境（CUDA / x86_64）：`rvqa_bundle/requirements/requirements.txt`
  （torch 2.5.1+cu121、transformers 4.49.0、peft 0.15.2 等；transformers 版本与
  LoRA inventory 的 module 前缀强相关）。
- Qwen3-VL-AWQ：`requirements-qwen3vl32b-awq.txt`；MiniCPM-V：`requirements-minicpmv26.txt`。
- Ascend 目标机：`requirements-ascend*.txt`。

model inventory 需先用
`pipeline/tools/build_model_inventory.py --model <model_dir> --output <inventory.json>`
生成（本仓库不含 inventories，它们与具体 checkpoint 绑定）。

---

## 6. 未提交内容

以下内容体积过大或属于本地环境 / 实验产物，不进入 Git：

- `conda_envs/`、`llm_models/`、`models/`（模型权重与环境）
- `reasoningvqa_official/`（官方 ReasonVQA 原始 JSONL）
- `reasoningvqa_subsets/`（子集图片与数据）
- `reasoningvqa_bench/`（源机实验工作区：runs / data / 早期 pipeline 副本）
- `rvqa_bundle/*.tgz`、`*.log`、`*.pid`、`scoring_extra/all-MiniLM-L6-v2/`
  （模型权重）、`scoring/*_run/`、`inat_manifests_open/`、`inventories/`

官方数据集请访问：<https://duong-tr.github.io/ReasonVQA/>
