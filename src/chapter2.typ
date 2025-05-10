#import "../uestc-thesis-template/lib.typ":*
#set math.equation(numbering: "(1)", supplement: [  公式:  ])

= 基于RAG技术的QAG框架


== RAG技术基础

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合了信息检索技术与语言生成模型的人工智能技术。该技术通过从外部知识库中检索相关信息，并将其作为提示（Prompt）输入给大型语言模型（LLMs），以增强模型处理知识密集型任务的能力，如问答、文本摘要、内容生成等。RAG技术由Facebook AI Research（FAIR）团队于2020年首次提出 @lewis2021retrievalaugmentedgenerationknowledgeintensivenlp，并迅速成为大模型应用中的热门方案，其典型结构如 @rag-structure 所示。

#picture-figure("典型RAG结构图。左上：各类型的文档进行切分和预处理，形成文本块；左下：用户与RAG系统的前端交互；右：RAG的常规工作流程。", 
                image("/src/pic/RAG.png",width: 100%)
                )<rag-structure>
=== RAG的工作过程

在一个RAG系统运作前，首先需要构建一个向量知识库，首先将知识库中的每个文档进行分段，然后使用一个编码器（通常是一个预训练的Transformer模型）将每个段落转换为向量，这个过程叫做Embedding。接下来，将这些向量存储在一个高效的向量数据库中，以便快速检索，如 @rag-structure 所示。

构建好向量数据库后，RAG系统的工作主要有以下三个步骤：

*检索（Retrive）:*根据用户的问题，使用检索模型从向量知识库中检索出相关的文档片段。

*增强（Augment）:*将用户查询和检索到的附加上下文填充到提示模板中构成新的提示词（Prompt）。

*生成（Generate）:*最后将增强的Prompt输入大模型，得到最终的Response。

在检索的过程中，检索模型通常利用余弦相似度的方法来评估Query与文档片段之间的相似程度，公式如下：

$ "Cosine Similarity"(bold(Q_i),bold(D_i)) = (upright(bold(Q)) dot.op upright(bold(D_i)))/(parallel upright(bold(Q)) parallel parallel upright(bold(D_i)) parallel) $

#align()[
#set par(first-line-indent: 0em) 
其中，$bold(Q)$表示Query向量，$bold(D)$表示文档向量，$upright(bold(A))$表示Query向量与文档向量之间的点积，$parallel$表示向量之间的模长。
]



=== 影响RAG输出效果的因素
RAG效果受多因素影响贯穿全流程。其一，知识库质量与构建关键，含文档准确性、时效性、覆盖范围、分块策略及更新频率，影响检索相关性与回答质量；其二，嵌入模型选择与性能关乎语义表达和领域适应，向量维度与训练数据匹配度影响检索精度；其三，检索阶段中，相似度计算、向量数据库性能及Top-K参数决定检索召回率与精度；其四，生成模型规模、训练数据、微调及温度参数影响回答流畅性与准确性。

// 因此，本文在构建基于RAG的QAG框架时，综合考虑以上因素，为QAG系统打造了轻量化，高效化的向量库管理结构，设计了混合检索机制来优化召回效果，并在文本分块部分引入多级分块策略来提升检索的精度和效率。

== MiniRAG-QAG框架介绍

#picture-figure("MiniRAG-QAG框架结构图。上：整个MiniRAG-QAG框架结构，输入论文，输出QA数据集；下：基于MiniRAG-QA生成器的内部结构，可以用于生成微调LLM_QAG的种子数据。", 
                image("/src/pic/miniRAG.png",width: 100%)
                )<mini_rag-structure>

MiniRAG-QAG框架专为从科学文献中自动生成问答（QA）数据集而设计，深度融合大语言模型（LLM）技术，其构建旨在高效、精准地挖掘文本中的知识并转化为结构化的问答形式。该框架涵盖数据预处理、QA生成和QA评估三个主要模块，各模块相互协作，以实现最终的QA数据集构建目标，其框架如 @mini_rag-structure 所示。


=== 数据预处理模块

数据预处理是整个框架的起始环节，对于确保后续操作的准确性和高效性至关重要。本框架从UTD24#footnote(link("UTD24：" + "https://jsom.utdallas.edu/the-utd-top-100-business-school-research-rankings/"))数据源中精心筛选出一批文章，这些文章最初多以pdf格式呈现。我们采用MinerU#footnote(link("MinerU：" + "https://github.com/opendatalab/MinerU"))。开源框架将pdf格式文档转换为markdown格式，此过程可以形式化表示为：

$ D_("pdf") = {d_1^("pdf"), d_2^("pdf"), ..., d_n^("pdf")} arrow D_("md") = {d_1^("md"), d_2^("md"), ..., d_n^("md")} $

#align()[
#set par(first-line-indent: 0em) 
其中 $D_("pdf")$ 表示原始pdf格式文档集合，$D_("md")$ 表示转换后的markdown格式文档集合。
]


在格式转换之后，随即进行数据清洗步骤，去除文档中诸如页眉页脚、重复内容以及与核心内容无关的注释等冗余信息。通过数据清洗，得到经过预处理的干净数据集$D_("pre")$，该数据集将作为后续QA生成模块的输入数据。

=== QA生成模块<qa_generation>

QA生成模块是E&M-QAG框架的核心组件，负责从预处理后的数据中生成问答对。此模块又细分为种子QA生成和基于LLM的QA生成两个子模块。

*种子QA生成模块：*

1. 实体抽取

  对于预处理后的论文数据$D_("pre")={d_1,d_2,d_3,...,d_n}$,其中，$d_i$表示第$i$篇预处理的论文数据，其核心概念主要出现在摘要(Abstract)、介绍(Introduction)、结论(Conclusion)等章节，我们利用正则表达式等技术将以上几个文章部分提取出来，记为$d_i^("aic")$。
  借助当前先进的大语言模型（记为$"LLM"_("extract")$），对每个子文档$d_i^("aic")$进行处理，抽取其中的核心概念实体集合$E_(i)= {e_("i1"),e_("i2"),...,e_("ik")}$,可表示为：$ E_(i) = "LLM"_("extract")(d_i^("aic")) $ 所有文档的核心概念实体集合可表示为$E = {E_(1), E_(2), ..., E_(n)}$，Prompt见附录A。

2. 基于通用LLM的问题抽取

  基于抽取得到的核心概念实体集合$E_i$，利用另一个大语言模型实例（记为$"LLM"_("QG")$），生成与这些实体相关的问题集合$(Q_i = {q_("i1"), q_("i2"), ..., q_("ik")})$，prompt见附录A，该过程可表示为：$ Q_(i) = "LLM"_("QG")(E_i) $所有文档的问题集合可表示为$Q = {Q_(1), Q_(2), ..., Q_(n)}$。

3. 基于MiniRAG的答案生成

  智能文档切分与Embedding：
  通过构建了一个智能文档切分模块，将预处理后的论文数据$d_i$进行动态切分，该模块分为两层，第一层根据语义进行段落级切分，第二层根据内容类型从代码，表格，普通文本三个方面分别配置分块大小和重叠长度，经过两次切分后最终得到若干子文档集合$S_(i) = S(d_i) = {s_("i1"), s_("i2"), ..., s_("im")}$，其中$s_("ij")$表示第$j$个子文档。将预处理的文章$d_i$的若干子文档集合$S_(i) $与问题集合$Q_i$通过Embedding模型BAAI/bge-m3 @multi2024m3 $("Embedding"(dot))$转换为向量表示$S_(i) ^("emb")={s_("i1")^("emb"), s_("i2")^("emb"), ..., s_("im")^("emb"))\}$、$Q_(i) ^("emb")={q_("i1")^("emb"), q_("i2")^("emb"), ..., q_("im")^("emb"))\}$，并存储在临时的向量数据库中。

  检索与排序：本研究构建了一个混合检索与排序系统，该系统分为两个环节，初检和精排。
  
  - 初检阶段：
  
  向量检索得分：使用BAAI/bge-m3 @multi2024m3 计算问题向量$q_("iz")^("emb")$ 与子文档向量 $s_("ij")^("emb")$的相似度得分$S_("vec")(q_("iz"), s_("ij"))$。

  BM25 @robertson2009probabilistic 关键词检索得分：使用 BM25 算法计算问题 $q_("iz")$与子文档 $s_("ij")$ 的关键词匹配得分 $S_("bm25")(q_("iz"), s_("ij"))$。$ S_("bm25")(q_("iz"), s_("ij")) = 
sum_(w in q_(i z)) "IDF"(w) dot.op((k_1 + 1) f(w, s_(i j)))/(f(w, s_(i j)) + k_1 (1 - b + b(|s_(i j)|)/"avgdl")) $ 

  #align()[
  #set par(first-line-indent: 0em) 
    其中，$q_("iz")$是问题，$s_("ij")$是子文档 ,$w$代表问题$q_("iz")$分词后的单个词,$f(w, s_("ij"))$表示词$w$在子文档$s_("ij")$中的出现频率，$|s_("ij")|$是子文档$s_("ij")$的长度，通常用包含的词数衡量，$"avgdl"$是平均文档长度，通过计算所有文档长度的平均值得到，$k_1$和$b$是可调参数，一般$k_1$取值在$1.2 - 2.0$之间，$b$取值为$0.75$，$"IDF"(w)$是词$w$的逆文档频率，其计算方式为：$"IDF"(w) = log(N - n_w + 0 . 5)/(n_w + 0 . 5)$，$N$是文档总数，$n_w$是包含词$w$的文档数量。
  ]


    - 混合得分计算：对于每个子文档 $s_("ij")$，其混合得分$S_("initial")(q_("iz"), s_("ij"))$ 由向量检索得分和 BM25 关键词检索得分加权求和得到，即：
    $ S_(i n i t i a l)(q_(i z), s_(i j)) = "w"_1 times S_(v e c)(q_(i z), s_(i j)) + "w"_2 times S_(b m 25)(q_(i z), s_(i j)) $
  
  - 精排阶段：

  相关性重排序：使用 BGE-Reranker #footnote(link("BGE-Reranker：" + "https://huggingface.co/BAAI/bge-reranker-base")) 大型模型对初检结果集合 $D_("top10")(q_("iz"))$中的每个子文档 $s_(i(j_m))$ 进行相关性重排序，得到重排序得分 $S_("re")(q_("iz"), s_(i(j_m)))$。

  最终结果：将初检结果集合 $D_("top10")(q_("iz"))$ 中的子文档按照重排序得分 $S_("re")(q_("iz"), s_(i(j_m)))$ 从高到低排序，选取得分最高的前 3 个子文档，构成问题 $q_("iz")$ 的最终结果集合 $D_("final")(q_("iz"))={s_(i(m_1)), s_(i(m_2)), s_(i(m_3))}$，其中 $(m_n)$ 表示子文档在 $D_("top10")(q_("iz"))$ 集合中的索引，$(n = 1,2,3)$。    问题集合 $Q_i$ 中每个问题按照上述两阶段检索步骤都能得到对应的最终结果集合，将这些最终结果集合组合成集合，记为：$ "Final"_i = {D_(f i n a l)(q_(i 1)), D_(f i n a l)(q_(i 2)), dots.c, D_(f i n a l)(q_(i k))} $

  - 答案生成：

  通过以上步骤，完成了从问题集合到对应最终检索结果集合的转换，为后续应用提供了相关性较高的子文档接下来我们将设定好的prompt（见附录A）与参考文本集以及问题$q_("ij")$合并得到大语言模型的检索增强输入内容：$ "input"_(i z)^("RA") = "promt"_("RAG")("Final"_i, q_("iz")) $

  问题集合$Q_i$对应的输入内容集记为：$ "Input"_i^("RA") = {"input"_(i 1)^("RA"), "input"_("i2")^("RA"), . . ., "input"_("ik")^("RA")} $ 利用另一个大语言模型实例（记为$"LLM"_("AG")$），生成与这些实体相关的答案集合$(A_i = {a_("i1"), a_("i2"), ..., a_("ik")})$，该过程可表示为：$A_(i) = "LLM"_("AG")("Input"^("RA")_(i))$

    以上构成一个临时的检索增强生成（RAG）系统。对于每个问题$q_("iz")$，利用RAG系统在向量数据库中检索相关信息并生成答案$a_(i)$，可简化表示为： $ a_(i z) = R A G(q_(i z),(S_(i z)^(e m b e d d i n g))) $

  种子数据构建：问题$q_("iz")$与对应的答案$a_("iz")$共同组成种子QA对$((q_("iz"), a_("iz"))_{"seed"})$，所有种子QA对构成种子QA对集合:$ Q A_(s e e d) =
mat(delim: "[", (q_11 comma a_11)_(s e e d),(q_12 comma a_12)_(s e e d), dots.c,(q_(1 k) comma a_(1 k))_(s e e d);
(q_21 comma a_21)_(s e e d),(q_22 comma a_22)_(s e e d), dots.c,(q_(2 k) comma a_(2 k))_(s e e d);
dots.c, dots.c, dots.c, dots.c;
(q_(n 1) comma a_(n 1))_(s e e d),(q_(n 2) comma a_(n 2))_(s e e d), dots.c,(q_(n k) comma a_(n k))_(s e e d))_(n times k) $

*基于LLM的QA生成模块：*

为了进一步提高QA生成的效率，本研究利用论文数据$D_("pre")$和种子QA对集合$("QA"_("seed"))$作为训练数据，对专门用于问答生成的大语言模型$"LLM"_("QAG")$进行监督微调（Supervised Fine-Tuning，SFT）。为了进一步优化算力，本研究采用使用了LoRA @hu2021loralowrankadaptationlarge。 

LoRA（Low-Rank Adaptation）微调是一种参数高效的模型微调方法，其核心思想是假设在针对下游任务进行微调时，模型权重的更新（记作 $"ΔW"$）本质上具有低秩结构。从数学角度，$W^′ = "W" + "ΔW"$,  其中 $W$ 是原始预训练模型的权重，$"ΔW"$表示权重更新。将矩阵 $"ΔW"$ 表示为多个小矩阵的 Kronecker 积之和：

$ "ΔW" ≈ sum_(k=1)^s λ_k (W_k^((1)) ⊗ W_k^((2)) ⊗ ... ⊗ W_k^((r))) $

#align()[
#set par(first-line-indent: 0em) 
其中，$⊗$ 表示 Kronecker 积，$λ_k$ 为标量因子，而分离秩 s 控制近似的精度。当 s 较小时，这种表示能大幅减少参数量，同时利用 Kronecker 积的并行计算优势提高效率 。因此，可以将$"ΔW"$近似分解为两个低维矩阵 A 和 B 的乘积，从而大大降低需要训练的参数数量，即$"ΔW" ≈ A dot B$，其中A 与 B 的尺寸分别为 $(w_1 times r)$ 和 $(r times w_2)$，其中 r（低秩）远小于$w_1$ 和 $w_2$。
]

这种分解利用了矩阵低秩分解的思想，即一个高维矩阵可以用较少的自由度来近似表示，从而实现参数高效的微调 。这种分解不仅在保持微调性能的前提下进一步减少了参数量，还为 GPU 等硬件上进行高效并行计算提供了可能。将论文文本数据 $d_i$ 输入给大语言模型$"LLM"_("QAG")$进行正向传播得到线性层输出：

$ y_i^("QAG")=(W + α times "ΔW")d_i^("emb") $

#align()[
#set par(first-line-indent: 0em) 
其中，$d_i^("emb")$为论文文本数据 $d_i$是经过大语言模型$"LLM"_{"QAG"}$内部$"embedding"$模型得到的向量,将$Q_i,A_i$经过处理得到$d_i$的问答对文本数据$"QA"_i$。
]

使用交叉熵函数计算损失：$ l o s s_i = C r o s s E n t r o p y(y_i^(Q A G), Q A_i) = - sum l o g(f(Q A_i |y_i^(Q A G))) $

#align()[
#set par(first-line-indent: 0em) 
其中，$f(Q A_i |y_i^(Q A G))$是根据$y_i^(Q A G)$得到的关于$"QA"_i$（每一个token）的概率分布函数。
]

根据损失$"loss"_i$可以算得对低秩矩阵 A 的梯度$("∂loss"_i)/(∂A)$和对低秩矩阵 B的梯度$("∂loss"_i)/(∂B)$，然后朝着梯度下降的方向调整A和B的参数。

微调过程可以表示为：$ "LLM"_("QAG")^("ft") = "SFT"("LLM"_("QAG"), D_("pre"),"QA"_("seed")) $

微调后的模型$"LLM"_("QAG")^("ft")$具备了基于输入文本生成问答对的能力。将预处理后的数据集$D_("pre")$输入到微调后的模型中（prompt见附录A），生成更多的问答对$(q, a)_("llm")$，即：$ (q, a)_("llm") = "LLM"_("QAG")^("ft")(D_("pre")) $

=== QA评估模块<QEM>

QA评估模块用于对生成的问答对进行质量筛选，确保最终纳入QA数据集的问答对满足一定的质量标准。无论是种子QA对$(q, a)_("seed")$还是由微调后的Q.A.LLM生成的问答对$(q, a)_("llm")$，都需要经过QA评估器的评估。
Yuwei Wan等人在他们的SciQAG框架 @wan2024sciqagframeworkautogeneratedscience 中提出了RACAR综合QA评估指标，包含相关性（Relevance）、不可知性（Agnosticism）、完整性（Completeness）、准确性（Accuracy）、合理性（Reasonableness），然而他们评估方式过于依赖评估模型的能力，因此本框架在此基础上对每个指标进行了单独的数学建模，进一步推动RACAR评估体系的客观化和标准化。具体指标计算方式如下：

1. 相关性

定义：量化生成QA的质量保证对与源文章中提供的信息的相关性，此外生成的问题需要询问文章中提供的事实和知识。

设源文章内容为文本集合$A$，利用词嵌入技术将其与问题$q$、答案$a$转换为向量$A_(vec)$、$q_(vec)$、$a_("vec")$，通过公式 @Relevance 计算，取值范围为$[0,1]$，值越大表示相关性越高。

$ C o r r e l a t i o n(q, a, A) = 1/(2n) dot.op sum_(i = 1)^n ((q_(v e c) dot.op A_(v e c) [i])/(mat(delim: bar.v.double, q_(v e c)) dot.op mat(delim: bar.v.double, A_(v e c) [i])) +(a_(v e c) dot.op A_(v e c) [i])/(mat(delim: bar.v.double, a_(v e c)) dot.op mat(delim: bar.v.double, A_(v e c) [i]))) $ <Relevance>

2. 不可知性

定义：量化生成问题的上下文独立程度，即生成的问题不得引用原始文本中的演示内容，如论文中的数字或表格。

设原始文本为$d$，问题文本为$q$，利用分句器(sent_tokenize)将$d$拆分为参考句集合RS，再将分词器(word_tokenize)对RS中每一个参考句rs进行拆分得到文档中所有句子的分词列表，即参考集Reference(d)。利用分词器对$q$进行拆分得到候选集Candidate(q)。利用BLEU @papineni2002bleu 模型可以得到：$ U 1(d, q) = 1 - B L E U(R e f e r e n c e(d), C a n d i d a t e(q)) $   基于BLEU模型的特点，可以通过U1来衡量问题对原始文本的机械引用，但由于问题和原始文本可能都会包含一些重要的专有名词，单纯用U1来表示不可知性有失偏颇。于是我们建立了违禁词集合为$$FW$$，如 @FM 所示。我们定义：
$ "U2"(q) = cases(
    0 quad "if" "fw" subset q "and" forall "fw" in "FW",
    1 quad "otherwise"
) $
利用的公式 @Unawareness 可以得到不可知性的量化数值：
$ "Unawareness"(d, q) = lambda * U 1(d, q) +(1 - lambda) * U 2(q) $<Unawareness>
其中，$lambda$是$"U1"(d,q)$的权重。

#table-figure(
  "违禁词集合，通常和论文上下文内容主体相关",
  table(
  columns: 1,
  stroke: none,
  align: center,
  // 这是顶头的粗线
  table.hline(stroke: 1pt),
  table.header([违禁词集合（FW）]),
  // 这是中间的细线
  table.hline(stroke: 0.5pt),
  // 第一行
  [this、these、those、the above text、as shown in、according to],
  table.hline(stroke: 1pt),
  )
)<FM>

3. 完整性

定义：评估答案是否全面涵盖问题的所有相关方面，并有效利用论文中的细节。

将文章切分为文本集$A$，将文本集向量化为$A^("emb")$，分别计算问题向量$q^("emb")$、答案向量$a^("emb")$与文本向量集$A^("emb")$中的各证据向量$A_i^("emb")$的余弦相似度以及问题向量与答案向量的余弦相似度$"Cos"(q^("emb"), a^("emb"))$。我们定义答案的关键证据集为文章中与问题$q$相关事实集合与答案$a$证据集合的交集。我们可以发现：$sum_(i = 1)^n ("Cos"(a^(e m b), A_i^(e m b)) dot.op "Cos"(q^(e m b), A_i^(e m b)))$在文本同时与问题和答案相关（双重高相似度）时较高，可以用于度量关键证据对于答案的贡献，对其归一化得到证据覆盖度:$ E C = (sum_(i = 1)^n ("Cos"(a^(e m b), A_i^(e m b)) dot.op "Cos"(q^(e m b), A_i^(e m b))))/(sum_(i = 1)^n "Cos"(q^(e m b), A_i^(e m b))) $ 为了避免证据覆盖度高但是答案跑题的情况，利用问题向量与答案向量的余弦相似度$"Cos"(q^("emb"), a^("emb"))$对证据覆盖度进行限制：$"ECs"="EC" . "Cos"(q^("emb"), a^("emb"))$.再通过开平方平衡直接相关性与证据覆盖度得到最后的公式：
$ "Completeness"(q, a, A) = root(, "Cos"(q^(e m b) comma a^(e m b)) dot.op(sum_(i = 1)^n ("Cos"(a^(e m b) comma A_i^(e m b)) dot.op "Cos"(q^(e m b) comma A_i^(e m b))))/(sum_(i = 1)^n "Cos"(q^(e m b) comma A_i^(e m b)))) $

完整性取值为[0,1]，值越大表示完整性越高。

4. 准确性

定义：衡量生成的答案与给定论文中呈现的相关事实或信息的对应程度，答案中的任何主张或陈述都应得到论文证据的支持。

将文章切分为证据集$E$，将证据集向量化为$E^("emb")$，分别计算问题向量$q^("emb")$、答案向量$a^("emb")$与证据向量集$E^("emb")$中的各证据向量$e^("emb")$的余弦相似度。设文章中与问题$q$相关事实集合为$F = {f thin|C o s(q^(e m b), f) > alpha, f in E^(e m b)}$，答案$a$证据集合为$S = {s thin|C o s(a^(e m b), s) > beta, s in E^(e m b)}$其中，$alpha$、$beta$分别是问题向量、答案向量与证据向量集语义相关的相似度阈值。经过测试，我们使用的文本嵌入模型计算出的语义相关的文本余弦相似度在$[0.6,1]$区间内。由于答案中一般包含更多与文章相关的事实，答案向量与证据向量的相似度比对应问题向量与证据向量的相似度高，因此$alpha<beta$。为避免绝对阈值导致相关事实集合或关键证据集合为空，我们使用了动态阈值：
$ alpha = a * 0 . 6 +(1 - a) * T o p 5(C o s(q^(e m b), E^(e m b)))\
beta = b * 0 . 7 +(1 - b) * root(, T o p 5(C o s(q^(e m b) comma E^(e m b))) * T o p 5(C o s(a^(e m b) comma E^(e m b))))
 $
 公式中$a、b$是常量权重， 设相关事实集合中的相关事实数量为$n_("total")$,设答案$a$证据集合与相关事实信息集合的交集的关键证据数量为$n_("correct")$。准确性指标通过公式 @Accuracy 计算，取值范围$[0, 1]$ 。
 $ "Accuracy"(q, a) = n_(c o r r e c t)/n_(t o t a l) $<Accuracy>

5. 合理性

定义：评估答案在逻辑上是否一致而没有矛盾。

维度：
- 问题&答案契合度：回答是否切题、直击问询核心？
- 完整性与简洁性：是否既不冗余也不漏要点？
- 可辩驳性：是否承认不确定性、允许后续纠正？

合理性利用大模型进行评估打分，提示词见附录A，将问题$q$答案$a$输入大语言模型打分，记为$"Score"_("logic")="LLM"("prompt"(q,a))$，取值范围$[0, 1]$ 。


6. 综合评估

对于一组QA，我们分别计算他们的相关性、不可知性、完整性、准确性和合理性得分，将得分进行加权求和，得到最终的综合评估得分$"Score"_("QA")$，此外，我们设定了一个阈值$T$,只有当问答对的得分$"Score" > "T"$时，该问答对才被认为是合格的，从而被保留下来。所有合格的问答对最终构成QA数据集$D_("QA")$。

