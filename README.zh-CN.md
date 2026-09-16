# ReasoningVQA

> [English](README.md) | 简体中文

面向**多跳知识推理 VQA** 的配置驱动训练与评测流水线，支持视觉语言模型。
源数据集（iNaturalist、GLDv2、Visual Genome）先归一化为统一的 canonical
manifest，再进入共享的 LoRA SFT 训练器与确定性评测/评分组件。

## 特性

- **所有数据源共用一套 schema** —— source adapters 统一输出
  `reasoningvqa.canonical.v1`，训练与评测不再按数据集分支。
- **配置驱动 CLI** —— 一个入口从 YAML 渲染并执行 `prepare`、`train`、
  `evaluate`（或 `all`）。
- **Qwen-VL LoRA SFT** —— Qwen2.5-VL / Qwen3-VL（兼容 MiniCPM-V）BF16
  MM-LoRA，仅对 response 计算 loss，支持 language-only 或全家族 LoRA，可选
  多卡 `device_map="balanced"`。
- **确定性评测** —— open-ended 或 structured 生成，并附带独立 re-score，
  分数不一致即失败。
- **纯评分 harness** —— 无需加载模型即可对已生成的预测打分。

## 目录结构

```text
rvqa_bundle/
├── pipeline/                 # 核心流水线代码
│   ├── run_pipeline.py       # CLI 入口
│   ├── config_schema.py      # 配置校验
│   ├── configs/              # 流水线模板
│   ├── model_backends/       # 模型族 operation 注册表
│   ├── pipelines/            # prepare / train / evaluate 阶段
│   │   ├── prepare/          # adapters、canonical schema、合并
│   │   ├── train/            # LoRA SFT 训练器
│   │   └── evaluate/         # 生成、评分、复评
│   ├── tools/                # LoRA inventory 构建/校验
│   └── tests/                # 契约测试
├── configs/                  # 各模型的 runbook 配置
├── scoring/                  # 纯评分 harness
├── scoring_extra/            # canonical MC 指标源
├── tools/                    # 数据过滤工具
└── requirements/             # 环境依赖（CUDA / Ascend）
```

## 环境安装

```bash
conda create -y -n rvqa python=3.10 && conda activate rvqa
pip install -r rvqa_bundle/requirements/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

LoRA target inventory 与 `transformers` 版本强相关，请保留 pin 的版本；
Ascend 环境使用 `requirements/requirements-ascend*.txt`。

为你的 checkpoint 构建 LoRA inventory（不加载权重）：

```bash
python rvqa_bundle/pipeline/tools/build_model_inventory.py \
  --model  /path/to/Qwen2.5-VL-7B-Instruct \
  --output /path/to/inventory.json
```

## 数据准备

每条源数据会被归一化到 canonical schema，校验后合并为 train/dev manifest：

```bash
cd rvqa_bundle/pipeline
python run_pipeline.py --config /path/to/config.yaml prepare --dry-run
python run_pipeline.py --config /path/to/config.yaml prepare
```

| Adapter | 数据源 |
| --- | --- |
| `inaturalist` | iNaturalist taxonomy + Wikidata |
| `gldv2` | GLDv2 landmarks |
| `visual_genome` | Visual Genome scene graphs |

canonical manifest 是下游唯一读取的契约（`sample_id`、`image_path`、
`sft_prompt`、`sft_target`、答案字段、推理路径、provenance）。

## 训练

```bash
python run_pipeline.py --config /path/to/config.yaml train
```

`pipelines/train/train_qwen_vl_sft.py` 的主要参数：

- `--lora-scope {language,all}` —— 仅语言部分或全部 LoRA 家族
- `--gpus 0,1` —— 大模型用 `device_map="balanced"` 分片
- `--init-adapter`、`--epochs`、`--max-rows` —— 续训 / 冒烟

训练器会写出 `adapter/`、`run_config.json`、`step_metrics.jsonl` 和
`result.json`，并在保存前校验参数完整性。

## 评测

```bash
python run_pipeline.py --config /path/to/config.yaml evaluate
```

- **Canonical**（`evaluate_canonical_reasoning.py`）：`open` / `json` 两种
  答案模式，支持 `correct-image`、`no-image`、`cyclic-shuffled-image`。
- **Structured**（`evaluate_structured_reasoning.py`）：root / evidence /
  hop / answer 分项 rubric，含 rotation 一致性。
- 两者之后都会跑**独立校验器**，逐行重算所有分数。

## 纯评分 Harness

对已生成的预测打分（不做推理）：

```bash
python rvqa_bundle/scoring/score_predictions.py \
  --eval-jsonl /path/to/eval.jsonl \
  --predictions-jsonl /path/to/predictions.jsonl \
  --output-dir /path/to/out \
  --model-path /path/to/all-MiniLM-L6-v2
```

输出 `open_exact`、`open_substring`、`fuzzy_proxy`、`mc_accuracy`、
`semantic_accuracy`、`mean_cosine` 以及按 hop 聚合的指标。若找不到 canonical
指标模块，会退回内置的 verbatim port（可用 `RVQA_CANONICAL_OPEN` /
`RVQA_CANONICAL_MC` 指定）。

## 配置说明

`pipeline/configs/` 与 `configs/` 下的配置故意把机器相关路径留为空字符串，
使用前请填写：

| 字段 | 含义 |
| --- | --- |
| `source.pipeline_root` | 包含 `prepare/`、`train/`、`evaluate/` 的目录 |
| `output.experiment_root` | 运行输出目录 |
| `runtime.python` | 训练环境解释器 |
| `model.checkpoint` | 本地模型目录 |
| `model.inventory` | LoRA inventory JSON |
| 数据源 `input` / `image_root` | 源 JSONL 与图片目录 |

`--dry-run` 只打印渲染后的命令；`--print-command` 只渲染不执行。

## 测试

```bash
cd rvqa_bundle/pipeline
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

## 引用

本仓库基于 [ReasonVQA](https://duong-tr.github.io/ReasonVQA/) 基准
（ICCV 2025）构建。使用数据时请引用原论文。

## 许可

遵循上游 ReasonVQA 数据集的许可（AGPL-3.0）；详见官方数据集页面。
