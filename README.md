# CorefQA: Coreference Resolution as Query-based Span Prediction
本仓库包含论文[CorefQA: Coreference Resolution as Query-based Span Prediction](https://arxiv.org/abs/1911.01746)的代码以及数据和预训练模型的获取方式。

## 实验准备
* 安装python依赖：`pip install -r requirements.txt`
* 准备训练数据：`python prepare_training_data.py`
* 在`experiments.conf`调节实验所用的超参数。

## 模型训练
1. 下载[Ontonotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19)数据集。
2. 下载[SpanBERT](https://github.com/facebookresearch/SpanBERT)预训练模型。
3. 运行`./setup_training.sh <ontonotes/path/ontonotes-release-5.0> $data_dir`进行数据预处理。
4. 训练模型`GPU=0 python train.py <experiment>`，训练结果保存在`log_root`目录，可以用TensorBoard查看训练细节。

## 使用预训练好的模型
使用如下命令下载预训练好的CorefQA模型。如果你想自己训练CorefQA模型，可以跳过这个步骤。
`./download_pretrained.sh <model_name>` (e.g,: spanbert_base, spanbert_large) 将会下载在Ontonotes英文数据集上fine-tune过的CorefQA模型。 您可以将其用于评估 `evaluate.py` 和预测 `predict.py`。

## 模型评估
运行 `GPU=0 python evaluate.py <experiment>`评估模型，可以通过在`experiments.conf`设置`eval_path`和`conll_eval_path`来选择在开发集还是在测试集上做评估。模型的评估效果如下：

| Model          | F1 (%) |
| -------------- |:------:|
| CorefQA + SpanBERT-base  | 79.9  |
| CorefQA + SpanBERT-large | 83.1   |

## 模型预测

* 将待预测的文本存为txt文件，每行是一段待预测的文本。如果有`speaker`信息，把它用(`<speaker></speaker>`)符号包起来放在他所说的话的前面。例如：
```text
<speaker> Host </speaker> A traveling reporter now on leave and joins us to tell her story. Thank you for coming in to share this with us.
```
* 运行 `GPU=0 python predict.py <experiment> <input_file> <output_file>`会把预测结果以jsonline的形式存入`<output_file>`中，每个instance的输出结果为list of clusters，每个cluster为list of mentions，每个mention为(text, (span_start, span_end))，例如：
```python
[[('A traveling reporter', (26, 46)), ('her', (81, 84)), ('you', (98, 101))]]
```

## 引用
如果您觉得我们的论文很有意思，请引用我们的论文 [Coreference Resolution as Query-based Span Prediction](https://arxiv.org/abs/1911.01746).
```
@article{wu2019coreference,
  title={Coreference Resolution as Query-based Span Prediction},
  author={Wu, Wei and Wang, Fei and Yuan, Arianna and Wu, Fei and Li, Jiwei},
  journal={arXiv preprint arXiv:1911.01746},
  year={2019}
}
```

## 致谢
我们在实现时参考了`https://github.com/mandarjoshi90/coref`，非常感谢！
