#import "../uestc-thesis-template/template/thesis.typ":*

= QAG_Prompt
#block(width: 100%)[
  #set align(center + horizon)
  #set text(size: 10pt)
  #table(
    columns: 1,
    // align: left,
    stroke: none,
    // 这是顶头的粗线
    table.hline(stroke: 1pt),
    table.header([抽取实体]),
    // 这是中间的细线
    table.hline(stroke: 0.5pt),
    // 第一行
    [
      ```
## Generation Requirements:
1. Please extract 15 entities from the above text.
2. The format of the generation is as follows:
["Entity 1","Entity 2",...]
3. Do not generate any text that is not related to the generation requirements.
4. The extracted entities only include academic entities such as concepts and academic terms, and do not include entities that are not related to academics, such as names, dates, and places.
Abstract：{abstract}
Instruction：{intro}
Conclusion：{conclusion}
      ```
    ],
    table.hline(stroke: 1pt),
  )
]

#block(width: 100%)[
  #set align(center + horizon)
  #set text(size: 10pt)
  #table(
    columns: 1,
    // align: left,
    stroke: none,
    // 这是顶头的粗线
    table.hline(stroke: 1pt),
    table.header([问题生成]),
    // 这是中间的细线
    table.hline(stroke: 0.5pt),
    // 第一行
    [
      ```
## Question Generation Requirements:
1. Generate 10 independent questions based on the following academic entities. Questions must focus on the entity's definition, principles, or applications.
2. Strictly avoid mentioning paper content, author research, or contextual information.
3. Prohibit generating questions of the following types:
   - "What is the role of this entity in the paper?"
   - "How did the authors apply this entity?"
   - "Where is this entity mentioned in the paper?"
4. Question types should be diversified, including but not limited to:
   - Concept explanation (What is...)
   - Technical comparison (Differences from...)
   - Application  (How to apply)
   - Historical development (Evolution process)
   - Mathematical principles (How it works/calculates)

## Format Requirements:
["Question 1","Question 2",...,"Question 10"]
## Entity List:
{entities}
Output ONLY the JSON-formatted list of questions that meet the requirements, without any explanations.
      ```
    ],
    table.hline(stroke: 1pt),
  )
]

#block(width: 100%)[
  #set align(center + horizon)
  #set text(size: 10pt)
  #table(
    columns: 1,
    // align: left,
    stroke: none,
    // 这是顶头的粗线
    table.hline(stroke: 1pt),
    table.header([问题生成]),
    // 这是中间的细线
    table.hline(stroke: 0.5pt),
    // 第一行
    [
      ```
# Role Definition
You are the Chief Expert of a world-leading economic management think tank, possessing the following core competencies:
1. Profound expertise in cutting-edge theories, empirical research methods, and policy analysis in economic management
2. Exceptional ability to translate complex economic models into actionable business insights
3. Precision in identifying statistical significance and practical relevance in research data
4. Cross-disciplinary integration capabilities (finance, econometrics, strategic management, etc.)

# Knowledge Repository
The following knowledge units have undergone rigorous academic validation:
{context_str}

# Problem Statement
"{question}"

# Generate evidence-based response using knowledge repository
      ```
    ],
    table.hline(stroke: 1pt),
  )
]

#block(width: 100%)[
  #set align(center + horizon)
  #set text(size: 10pt)
  #table(
    columns: 1,
    // align: left,
    stroke: none,
    // 这是顶头的粗线
    table.hline(stroke: 1pt),
    table.header([基于LLM_QAG的QA生成]),
    // 这是中间的细线
    table.hline(stroke: 0.5pt),
    // 第一行
    [
      ```
# Role Definition  
You are a QAG (Question Answer Generation) expert skilled in transforming academic papers into QA pairs  
# Workflow  
Step 1. Extract 10 key entities from the article  
Step 2. Construct 10 QA pairs around these entities and the article content  
Step 3. Strictly follow the output format and only output QA pairs  
## QA Generation Principles  
1. Academic questions must focus on the entity's ​**definition, principles, or applications**  
2. Strictly avoid mentioning paper content, author research, or contextual information  
3. Prohibit generating questions of the following types:  
• "What is the role of this entity in the text?"  
• "How did the authors apply this entity?"  
• "Where does this entity appear in the paper?"  
4. Diversify question types, including but not limited to:  
• Conceptual explanation (What is...)  
• Technical comparison (Difference between... and...)  
• Application scenarios (How is it applied...)  
• Historical development (Evolution process)  
• Mathematical principles (How to calculate...)  
# Output Principles
1. Strictly follow the output format and only output QA pairs
2. Don't add any emphatic marks, any headings, any explanatory statements

# Output Format  
jsonl
[
  {{
    "Question": "……",
    "Answer": "……"
  }},
  {{
    "Question": "……",
    "Answer": "……"
  }}
]

# Paper Content
markdown
{content}
 
```
    ],
    table.hline(stroke: 1pt),
  )
]

#block(width: 100%)[
  #set align(center + horizon)
  #set text(size: 10pt)
  #table(
    columns: 1,
    // align: left,
    stroke: none,
    // 这是顶头的粗线
    table.hline(stroke: 1pt),
    table.header([合理性评估]),
    // 这是中间的细线
    table.hline(stroke: 0.5pt),
    // 第一行
    [
      ```
# Role Definition  
You are an expert QA evaluator skilled in assessing answer quality across multiple dimensions  
# Workflow
1. Receive input: <Question> and <Answer>  
2. Analyze based on four criteria:  
   - Internal Logical Consistency  
   - Question-Answer Relevance  
   - Completeness & Conciseness  
   - Rebuttability  
1. Assign 0-5 scores per dimension (5=excellent)  
2. Provide brief rationale for each score  
# Definitions of each dimension
1. Internal Logical Consistency: Is the internal response consistent and free of contradictions?
2. Question-Answer Relevance: Is the answer to the point and to the heart of the inquiry?
3. Completeness & Conciseness: Is it neither redundant nor missing points?
4. Rebuttability: Does the answer acknowledge uncertainty and allow for subsequent corrections?
# Evaluation Criteria  
**1. Internal Logical Consistency**  
5: No contradictions, coherent multi-step reasoning  
3: Minor inconsistencies but core logic holds  
0: Direct contradictions present  
**2. Question-Answer Relevance**  
5: Fully addresses question intent/scope  
3: Partially relevant with minor digressions  
0: Completely off-topic  
**3. Completeness & Conciseness**  
5: Covers essentials without redundancy  
3: Missing 1-2 key points or slight verbosity  
0: Major omissions or excessive verbosity  
**4. Rebuttability**  
5: Proper hedging when needed, no overconfidence  
3: Occasional over-assertiveness  
0: Critical errors in certainty level  
# Output Principles
1. Strictly follow the output format
2. Don't add any emphatic marks, any headings, any explanatory statements
# Output Format  
{{  
  "question": "Original question",  
  "answer": "Submitted answer",  
  "scores": {{  
    "consistency": n,  
    "relevance": n,  
    "conciseness": n,  
    "rebuttability": n  
  }},  
  "rationale": {{  
    "consistency": "text",  
    "relevance": "text",  
    "conciseness": "text",  
    "rebuttability": "text"  
  }}  
}}  
# Input Question and Answer
Question:{question}
Answer:{answer}
```
    ],
    table.hline(stroke: 1pt),
  )
]


= UTD-24_QA数据集

#block(width: 100%)[
  #set align(center + horizon)
  #set text(size: 10pt)
  #table(
    columns: 5,
    // align: left,
    stroke: none,
    // 这是顶头的粗线
    table.hline(stroke: 1pt),
    table.header([期刊名],[论文数（篇）],[QA-$"LLM"_"QAG"$（对）],[QA-MiniRAG（对）],[QA总数]),
    // 这是中间的细线
    table.hline(stroke: 0.5pt),
    // 第一行
    [Marketing_Science],[523],[4040],[932],[4972],
    [Journal_on_Computing],[613],[917],[932],[1849],
    [Information_Systems_Research],[625],[5060],[932],[5992],
    [Organization_Science],[961],[8420],[932],[9352],
    [The_Review_of_Financial_Studies],[644],[5250],[932],[6182],
    [The_Accounting_Review],[651],[5320],[932],[6252],
    [Operations_Research],[632],[8420],[932],[9352],
    [Administrative_Science_Quarterly],[615],[4960],[932],[5892],
    [Journal_of_Accounting_Research],[619],[4991],[932],[5923],
    [Journal_of_Accounting_and_Economics],[1148],[10284],[932],[11216],
    [MIS_Quarterly],[524],[5050],[932],[5982],
    [Journal_of_Marketing_Research],[1134],[10150],[932],[11082],
    [Strategic_Management_Journal],[707],[5880],[932],[6812],
    [Journal_of_Marketing],[1204],[10851],[932],[11783],
    [Production_and_Operations_Management],[591],[4720],[932],[5652],
    [Academy_of_Management_Journal],[757],[6380],[932],[7312],
    [Management_Science],[675],[5560],[932],[6492],
    [Journal_of_Operations_Management],[1043],[9281],[932],[10213],
    [Journal_of_International_Business_Studies],[634],[5141],[932],[6073],
    [The_Journal_of_Finance],[622],[5031],[932],[5963],
    [Journal_of_Consumer_Research],[529],[4100],[932],[5032],
    [Academy_of_Management_Review],[500],[3810],[932],[4742],
    [Manufacturing_and_Service_Operations],[885],[7660],[932],[8592],
    [Journal_of_Financial_Economics],[706],[9871],[933],[10804],
    [总计],[17542],[147857],[22369],[170226],

    table.hline(stroke: 1pt),
  )
]

= QAG_Systemy元数据

#block(width: 100%)[
  #set align(center + horizon)
  #set text(size: 10pt)
  #table(
    columns: 1,
    // align: left,
    stroke: none,
    // 这是顶头的粗线
    table.hline(stroke: 1pt),
    table.header([QAG_Systemy元数据（1组QA数据）]),
    // 这是中间的细线
    table.hline(stroke: 0.5pt),
    // 第一行
    [
      ```json
{
    "标题": "2405.09939v2.pdf",
    "上传时间": "2025-04-23 15:51:01",
    "大小": "1.08 MB",
    "状态": "已嵌入",
    "存储路径": "PDF/20250423155101_2405.09939v2.pdf",
    "md路径": "Markdown/2405.09939v2.md",
    "向量库路径": "vector_db/5f52e585_2405.09939v2",
    "实体数量": 15,
    "实体": [
        "SciQAG",
        "large language models (LLMs)",
        "QA generator",
        "QA evaluator",
        "science QA dataset",
        "SciQAG-24D",
        "open-ended question answering",
        "scientific tasks",
        "fine-tuning",
        "natural language processing (NLP)",
        "reading comprehension",
        "visual QA",
        "BioBERT",
        "Med-PALM",
        "Galactica"
    ],
    "QA_result": [
        {
            "question": "What is the primary function of SciQAG in generating academic questions?",
            "answer": "The primary function of SciQAG (Automatic Generation of Science Question Answering) is to automatically generate high-quality, research-level question-answer pairs from scientific articles using large language models (LLMs). It achieves this through two main components: a QA generator, which extracts relevant information and creates diverse questions and answers, and a QA evaluator, which assesses the quality of these pairs. By avoiding self-referential expressions, SciQAG ensures that the generated questions are contextually clear and suitable for closed-book applications, while maintaining a broad scope of questioning. This framework facilitates the creation of large-scale, open-ended science QA datasets, such as one containing 188,042 QA pairs extracted from 22,743 scientific papers, thereby supporting advanced educational and research purposes.",
            "reference": "来源文件: Markdown/2405.09939v2.md\n内容类型: normal\n原文片段: while closed-book indicates that no external knowledge or context is provided with the question (Roberts et al., 2020). We introduce Automatic Generation of Science Question Answering (SciQAG), a framework for automatically generating QA pairs from scientific articles (see Figure 1). SciQAG comprises two main components: a QA generator and a QA evaluator. The QA generator leverages LLMs to extract relevant information from scientific papers and generate diverse question-answer pairs while the QA evaluator\n\n来源文件: Markdown/2405.09939v2.md\n内容类型: normal\n原文片段: We introduce SciQAG, a novel framework for automatically generating high-quality science question-answer pairs from a large corpus of scientific literature based on large language models (LLMs). SciQAG consists of a QA generator and a QA evaluator, which work together to extract diverse and research-level questions and answers from scientific papers. Utilizing this framework, we construct a large-scale, high-quality, open-ended science QA dataset containing 188,042 QA pairs extracted from 22,743 scientific\n\n来源文件: Markdown/2405.09939v2.md\n内容类型: normal\n原文片段: Due to the nature of its prompting technique, self-questioning is prone to generate more generic and contextually tied questions (e.g., “What is being discussed in this paragraph?”). SciQAG is able to avoid self-referential expressions such as ‘this,’ ‘that,’ and ‘these,’ making it more suitable for crafting challenging closed-book QA datasets, though this may slightly compromise its compatibility with source papers for open-book QA applications. The scope of questions generated by SciQAG is notably broad"
        }
    ],
    "Chunks地址": "Chunks/5f52e585_2405.09939v2.json"
}
      ```
    ],
    table.hline(stroke: 1pt),
  )
]





