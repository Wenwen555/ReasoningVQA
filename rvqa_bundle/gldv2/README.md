# ReasonVQA subset — GLDv2

子集由官方 ReasonVQA 数据集构建。

- Source: `GLDv2`
- Train rows: 5000 / target 5000
- Eval rows: 1000 / target 1000
- Unique images train/val: 4958 / 995
- Image overlap train∩eval: 0
- Seed: 20260713

## 构成 / Layout
- `train.jsonl` — 5000 行, 已按 `question_id` JOIN `train_ann.jsonl`
- `eval.jsonl` — 1000 行, 已按 `question_id` JOIN `val_ann.jsonl`
- `summary.json` — 采样/join 统计, hop 与 property_label 分布
- `images/` — 保持原样, 本次未改动 (image_path 保持官方原值 `""`, 图像下载为后续步骤)

## 采样规则 / Sampling
uniform random sample of question rows (random.Random(20260713).sample) from the official split filtered to the source; train and val use an independent Random(20260713) so both samples are reproducible; rows are joined 1:1 with the matching *_ann.jsonl by question_id; after sampling any train image_id also present in the val sample is dropped from train.

## 字段 / Fields
原始 main 字段 (`image_id, image_name, source, image_path, question_id,
question, answers, correct, categories, image_url`) + JOIN 补入
`hop, property_id, property_label, entity_id, entity_name, route,
has_scene_graph` from `*_ann.jsonl`。

## 说明 / Notes
- 官方 train/val 本身按图像划分, 采样后 train 与 eval 的 image_id 交集为
  0.
- 图像未下载, `images/` 目录未修改; `image_path` 保持原值.
