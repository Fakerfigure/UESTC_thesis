#import "uestc-thesis-template/lib.typ":*

// 有序列表以及无序列表的缩进
// 按照需要设置
#set enum(indent: 2em)
#set list(indent: 2em)

#show: thesis.with(info: (
  // DEBUG 开关
  // DEBUG: true,
  /*
    模板信息
    会影响输出样式

    字体相关: 建议使用思源系列字体, 请参考生成的文档
    但是如果想使用其他字体, 可以通过下面的例子修改
    另外, 有些字体可能会导致使用粗体字重过粗, 可以通过加粗粗度来解决
  */
  info-keys.打印: true,
  info-keys.匿名: false,
  // info-keys.黑体字体: "Source Han Sans SC",
  // info-keys.宋体字体: "Source Han Serif SC",
  // info-keys.加粗粗度: 150,
  /**
    封面信息
  */
  //
  // 作者基本参数 会影响总体封面效果
  //
  info-keys.申请学位级别: "本科", // 硕士、博士
  // info-keys.学位类型: "专业型", // 学术型、专业型
  //
  // 封面
  //
  info-keys.论文中文标题: "基于LLM自动生成问答数据集的框架",

  info-keys.作者学院: "信息与通信工程学院",
  info-keys.作者学科专业: "通信工程", // 学术型填写, 专业型忽略
  // info-keys.作者专业学位类别: "三头六臂", // 专业型填写, 学术型忽略
  info-keys.作者学号: "2021010903003",
  info-keys.作者中文名: "高斌",
  info-keys.指导老师中文名: "陈大江",
  // info-keys.指导老师职称中文: "金仙级教授",

  
  // 中文扉页
  
  // info-keys.分类号: "TP309.2",
  // info-keys.密级: "公开",
  // info-keys.UDC: "004.78",
  // // 标题与已经在封面中中定义
  // // 作者中文名已经在封面中定义
  // // 指导老师中文名已经在封面中定义学位论文写作指南及例子
  // // 指导老师职称中文已经在封面中定义
  // info-keys.指导老师单位: "玉虚宫 九重天",
  // info-keys.合作导师中文名: "申公豹",
  // info-keys.合作导师职称中文: "金仙级教授",
  // info-keys.合作导师单位: "玉虚宫 九重天",
  // // 申请学位级别已经在作者基本参数中定义
  // // 专业型: 专业学位类型 已经在封面中定义
  // // 学术型: 作者学科专业 已经在封面中定义
  // info-keys.专业学位领域: "大闹天宫学", // 专业型填写, 学术型忽略
  // info-keys.提交日期: "2025年3月17日",
  // info-keys.答辩日期: "2025年4月15日",
  // info-keys.学位授予单位: "电子科技大学",
  // info-keys.学位授予日期: "2025年6月1日",
  // info-keys.答辩委员会主席: "元始天尊",
  // info-keys.答辩委员会主席职称: "大罗金仙",
  // info-keys.答辩委员会成员: ("普贤道人", "广成子"),
  // info-keys.答辩委员会成员职称: ("金仙级", "金仙级"),
  //
  // 英文扉页
  //
  // info-keys.论文英文标题: "UESTC-Typst Thesis Template Usage Guide and Usage Examples",
  // info-keys.作者学科专业英文: "Imitate the Heaven and Earth", // 学术型填写, 专业型忽略
  // info-keys.作者专业学位类别英文: "Three heads and six arms", // 专业型填写, 学术型忽略
  // info-keys.作者英文名: "Nezha",
  // info-keys.指导老师英文名: "Taiyi Zhenren",
  // info-keys.指导老师职称英文: "Sage",
  // info-keys.作者学院英文: "Jinguang Cave on Qianyuan Mountain",
  
  // /**
  //   论文内容信息
  // */
  info-keys.中文摘要: include("src/摘要-中文.typ"),
  info-keys.中文摘要关键字: ("大型语言模型", "RAG（检索增强生成）", "问答数据集", "多维度评估"),
  info-keys.英文摘要: include("src/摘要-英语.typ"),
  info-keys.英文摘要关键字: ("Large Language Model", "Retrieval-Augmented Generation", "QA Dataset", " Multi-dimensional Evaluation"),
  info-keys.附录: (include("src/附录-1.typ"), include("src/附录-2.typ")),
  info-keys.致谢: include("src/致谢.typ"),
  info-keys.外文资料原文: include("src/外文资料原文.typ"),
  info-keys.外文资料译文:include("src/外文资料译文.typ"),
  info-keys.参考文献: ("src/bib/参考文献1.bib", "src/bib/参考文献2.bib"),
  // info-keys.攻读学位期间取得成果: "src/攻读学位期间取得成果.bib",
))

#include "src/chapter1.typ"
#include "src/chapter2.typ"
#include "src/chapter3.typ"
#include "src/chapter4.typ"
#include "src/chapter5.typ"
#include "src/chapter6.typ"


