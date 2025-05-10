#import "../uestc-thesis-template/lib.typ":*

= UTD24_QA数据集

利用MiniRAG-QAG框架，本研究将选自UTD24中的17542篇文章转换成了*170226*组高质量的QA对，将其命名为UTD24_QA数据集，该数据符合Alpaca#footnote(link("Alpaca：" + "https://github.com/tatsu-lab/stanford_alpaca"))格式，可以直接用于大模型的微调。

== 论文收集

#picture-figure("UTD24_QA数据集中各个期刊的论文数量分布，总论文数量为17542。", 
            image("/src/pic/utd24_c.jpg",width: 100%)
            )<utd24>

*UTD24：*全称为The University of Texas at Dallas 24期刊#footnote(link("UTD24" + "https://jsom.utdallas.edu/the-utd-top-100-business-school-research-rankings/"))，是美国德克萨斯大学达拉斯分校(The University of Texas at Dallas)的纳文金达尔管理学院(Naveen Jindal School of Management)创建的一个期刊数据库，总共24本期刊，用以对全世界前100名商学院进行排名。

本研究从UTD24系列期刊中选择了20000篇文章进行预处理，最终有17542篇文章符合后续要求。各期刊选择论文数量比例如 @utd24 所示，这些文章包含基本的三元式内容（abstract、instruction、conclusion），对于每个期刊，下载最新的且引用量靠前的文章（PDF），这些文章旨在为后续QA生成提供权威、可信的数据源，详细的论文数量数据见附录B。

== 使用MiniRAG-QAG

本研究融合并改造了MinerU框架作为数据预处理的结构，从预处理后的文章中选取了2500篇，利用mini-rag架构生成22370个高质量的种子QA对，利用这些种子QA构造了符合Alpaca格式的数据集，得到该数据集后，微调了开源大模型Qwen_v2.5_7b_Instruct作为$"LLM"_"QAG"$，利用$"LLM"_"QAG"$处理剩余的文章，总共生成*170226*个QA对（包含种子QA）。从中随机选取了5000条QA用于构建benchmark，其余QA作为UTD-24_QA数据集的源数据。

MiniRAG-QAG使用到的模型以及关键参数如下：



- *LLM：*$"LLM"_"extract" | "LLM"_"QG" | "LLM"_"AG"$
    
    - 模型：Qwen-Max
    - 调用方式：API
    - temperature#footnote(link("temperature：" + "采样温度，控制模型生成文本的多样性，越高，生成的文本更多样，反之，生成的文本更确定。")) ：0.7 | 0.7 | 0.3
    - 其他参数：max_tokens#footnote(link("max_tokens：" + "一次请求返回的最大 Token 数。"))：131,072 tokens；top_p#footnote(link("top_p" + "核采样的概率阈值，控制模型生成文本的多样性。"))：0.8；presence_penalty#footnote(link("presence_penalty" + "控制模型生成文本时的内容重复度。"))：1
- *Embedder：*
    - 模型：BAAI/bge-m3
    - 调用方式：本地部署
- *Reranker：*
    - 模型：BAAI/bge-reranker-base
    - 调用方式：本地部署
- *$"LLM"_"QAG"$*
    - 模型：Qwen_v2.5_7b_Instruct
    - 调用方式：本地部署
    - 微调框架：LLaMA-Factory#footnote(link("LLaMA-Factory" + "https://github.com/hiyouga/LLaMA-Factory")) @zheng2024llamafactory
    - 微调参数：batch_size：16；learning_rate：2e-5；epoch：3；peft_type: lora
    - 其他参数：temperature：0.7；max_tokens：8192 tokens；top_p：0.8；
    - 微调过程中的梯度变化（Gradient Norm）和训练损失（Train Loss）的趋势如 @loss 所示。

#picture-figure("左：微调过程中的梯度变化；右：微调过程中的训练损失。", 
            image("/src/pic/grad&loss.png",width: 100%)
            )<loss>

== UTD24_QA_5k-benchmark

为了系统评估不同大语言模型在经管学术领域中的问答理解与知识覆盖能力，本文基于 UTD‑24 数据集构建了一个标准化的四选一选择题评测集，即UTD24_QA_5k‑benchmark，其结构如 @benchmark 所示。

#picture-figure("UTD24_QA_5k-benchmark的结构，将单独的QA数据转换成单项选择题。", 
            image("/src/pic/benchmark.png",width: 100%)
            )<benchmark>

从完整的 UTD-24_QA 数据集中随机抽取了5000条QA作为 benchmark 构建基础，并按照以下流程生成标准化的选择题来评价测试集的质量。首先，使用 Qwen-plus 模型对原始问答对中的答案进行简化，得到一条简洁、表达清晰的标准答案，作为正确选项。随后，输入原问题和答案，模型被提示生成一个与该答案在长度和复杂度相似、但语义上错误的干扰项（作为选项 B）。为了进一步提升题目的多样性，将原问题重新改写，使其无法再由原答案回答，并使用模型生成该新问题的答案（作为选项 C），再基于该答案生成另一个错误选项（作为选项 D）。最终，将上述四个选项打乱顺序并记录正确选项的标签，生成结构完备的选择题。



#table-figure(
  "UTD24_QA_5k-benchmark评测结果",
  table(
  columns: 3,
  stroke: none,
  align: center,
  // 这是顶头的粗线
  table.hline(stroke: 1pt),
  table.header([模型],[得分],[调用方式]),
  // 这是中间的细线
  table.hline(stroke: 0.5pt),
  // 第一行
  [*meta-llama/Llama-4-Scout-17B-128E-Instruct*],[*77*],[*weight*],
  [Qwen/qwen2.5-32b-instruct],[76],[weight],
  [deepseek-ai/DeepSeek-R1],[74],[API],
  [OpenAI/GPT-4o- mini],[74],[API],
  [meta-llama/Llama-4-Scout-17B-16E-Instruct],[73],[weight],
  [Qwen/qwen2.5-14b-instruct],[72],[weight],
  [meta-llama/Meta-Llama-3.1-8B-Instruct],[72],[weight],
  [Qwen/qwen-max],[71],[API],
  [deepseek-ai/DeepSeek-V3],[68],[API],
  [Qwen/qwen2.5-7b-instruct-1m],[68],[weight],
  [CohereLabs/c4ai-command-r-plus-08-2024],[67],[API],
  [google/gemma-3-27b-it],[66],[weight],
  [cognitivecomputations/Dolphin3.0-Llama3.1-8B],[66],[weight],
  [meta-llama/Llama-4-Scout-17B-4E-Instruct],[65],[weight],
  [meta-llama/Llama-4-Scout-17B-8E-Instruct],[65],[weight],
  [Qwen/qwen-plus],[65],[API],
  [deepseek-ai/deepseek-r1-distill-qwen-14b],[65],[weight],
  [google/gemma-2-27b],[64],[weight],
  [deepseek-ai/deepseek-r1-distill-llama-8b],[62],[weight],
  [deepseek-ai/deepseek-r1-distill-qwen-7b],[60],[weight],
  [mistralai/Mistral-Large-Instruct-2407],[58],[weight],
  [tiiuae/Falcon3-3B-Base],[58],[weight],
  [Qwen/qwen2.5-3b-instruct],[54],[weight],
  [Qwen/qwen2.5-1.5b-instruct],[53],[weight],
  [tiiuae/Falcon3-1B-Base],[50],[weight],
  table.hline(stroke: 1pt),
  )
)<benchmark-result>

该评测集共包含 5000 道四选一题目，每道题目均来源于一个原始 QA 对，并通过结构化流程构造出逻辑合理、干扰性强的选项集合，选取了一系列大模型在UTD24_QA_5k-benchmark上进行评测（总分100），结果如 @benchmark-result 所示，所有的模型的得分都明显高于随机的准确率（约 25%），说明测试基准有不错的可回答性，其中来自meta的LLM4系列模型得分最高，Falcon3-1B-Base得分最低，同时各个模型在该测评集的表现并不完全符合分数和参数量正相关的特性，这说明目前大部分模型在进行前置训练时并没有针对经管类知识进行特殊优化，因此UTD24_QA数据集可以弥补这一空白，为模型在SFT阶段提供更丰富的训练数据。

