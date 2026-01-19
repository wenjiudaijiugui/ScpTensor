const PptxGenJS = require("pptxgenjs");

async function createPresentation() {
    const pptx = new PptxGenJS();
    pptx.layout = 'LAYOUT_16x9';
    pptx.author = 'ScpTensor Team';
    pptx.title = 'ScpTensor 项目状态报告';

    const COLOR_PRIMARY = "003366";
    const COLOR_ACCENT = "008080";
    const COLOR_BG_GRAY = "F8F9FA";
    const COLOR_TEXT = "333333";

    const slide = pptx.addSlide();

    // Header
    slide.addShape(pptx.shapes.RECTANGLE, { x: 0, y: 0, w: '100%', h: 0.8, fill: { color: COLOR_PRIMARY } });
    slide.addText("ScpTensor 项目概览", { x: 0.3, y: 0.1, fontSize: 20, bold: true, color: "FFFFFF" });
    slide.addText("高性能单细胞蛋白质组学分析框架", { x: 0.3, y: 0.45, fontSize: 10, color: "E0E0E0" });
    slide.addText("v0.2.0 (生产就绪) | 2026年1月15日", { x: 7.0, y: 0.3, w: 2.7, fontSize: 9, color: "CCCCCC", align: "right" });

    // Left Column
    // Strategic Vision
    slide.addShape(pptx.shapes.RECTANGLE, { x: 0.3, y: 1.0, w: 3.0, h: 2.0, fill: { color: COLOR_BG_GRAY }, line: { color: "DDDDDD" } });
    slide.addShape(pptx.shapes.LINE, { x: 0.3, y: 1.0, w: 0, h: 2.0, line: { color: COLOR_PRIMARY, width: 3 } });
    slide.addText("战略愿景", { x: 0.4, y: 1.1, fontSize: 11, bold: true, color: COLOR_PRIMARY });
    slide.addText("目标：构建连接质谱数据与生物学洞察的可扩展、生产级生态系统。", { x: 0.4, y: 1.4, w: 2.8, fontSize: 9, color: COLOR_TEXT });
    slide.addText("核心理念：", { x: 0.4, y: 2.0, fontSize: 9, bold: true, color: COLOR_TEXT });
    slide.addText([
        { text: "不可变性 (Immutability)", options: { bullet: true } },
        { text: "类型安全 (Type Safety)", options: { bullet: true } },
        { text: "可扩展性 (Scalability)", options: { bullet: true } }
    ], { x: 0.4, y: 2.2, w: 2.8, fontSize: 8, color: COLOR_TEXT });

    // System Architecture
    slide.addShape(pptx.shapes.RECTANGLE, { x: 0.3, y: 3.2, w: 3.0, h: 2.0, fill: { color: COLOR_BG_GRAY }, line: { color: "DDDDDD" } });
    slide.addShape(pptx.shapes.LINE, { x: 0.3, y: 3.2, w: 0, h: 2.0, line: { color: COLOR_PRIMARY, width: 3 } });
    slide.addText("系统架构", { x: 0.4, y: 3.3, fontSize: 11, bold: true, color: COLOR_PRIMARY });
    
    // Diagram
    slide.addShape(pptx.shapes.RECTANGLE, { x: 0.5, y: 3.6, w: 2.6, h: 0.8, fill: { color: "EEF6FF" }, line: { color: "0066CC", dashType: 'dash' } });
    slide.addText("ScpContainer (容器)\n↓\nAssay (特征) → ScpMatrix (稀疏矩阵)", { x: 0.5, y: 3.6, w: 2.6, h: 0.8, align: "center", fontSize: 9, color: "004488" });

    slide.addText("溯源追踪：", { x: 0.4, y: 4.5, fontSize: 9, bold: true, color: COLOR_TEXT });
    slide.addText([
        { text: "掩码代码 (有效值, LOD, 插补值)", options: { bullet: true } },
        { text: "全流程审计日志记录", options: { bullet: true } }
    ], { x: 0.4, y: 4.7, w: 2.8, fontSize: 8, color: COLOR_TEXT });


    // Middle Column
    // Recent Milestones
    slide.addShape(pptx.shapes.RECTANGLE, { x: 3.5, y: 1.0, w: 3.0, h: 1.8, fill: { color: COLOR_BG_GRAY }, line: { color: "DDDDDD" } });
    slide.addShape(pptx.shapes.LINE, { x: 3.5, y: 1.0, w: 0, h: 1.8, line: { color: COLOR_PRIMARY, width: 3 } });
    slide.addText("近期里程碑 (v0.2.0)", { x: 3.6, y: 1.1, fontSize: 11, bold: true, color: COLOR_PRIMARY });
    slide.addText([
        { text: "高级质控：双变量分析与批次检测", options: { bullet: true } },
        { text: "统计模块：5种新差异表达方法", options: { bullet: true } },
        { text: "测试覆盖：65% (1,423项测试通过)", options: { bullet: true } },
        { text: "文档完备：8个教程 + API参考手册", options: { bullet: true } }
    ], { x: 3.6, y: 1.4, w: 2.8, fontSize: 9, color: COLOR_TEXT });

    // Module Ecosystem
    slide.addShape(pptx.shapes.RECTANGLE, { x: 3.5, y: 3.0, w: 3.0, h: 2.2, fill: { color: COLOR_BG_GRAY }, line: { color: "DDDDDD" } });
    slide.addShape(pptx.shapes.LINE, { x: 3.5, y: 3.0, w: 0, h: 2.2, line: { color: COLOR_PRIMARY, width: 3 } });
    slide.addText("模块生态", { x: 3.6, y: 3.1, fontSize: 11, bold: true, color: COLOR_PRIMARY });
    
    // Split Lists
    slide.addText("分析流程", { x: 3.6, y: 3.4, fontSize: 9, bold: true, color: COLOR_TEXT });
    slide.addText([
        { text: "标准化 (8种方法)", options: { bullet: true } },
        { text: "插补 (KNN, SVD等)", options: { bullet: true } },
        { text: "整合 (Harmony等)", options: { bullet: true } }
    ], { x: 3.6, y: 3.6, w: 1.4, fontSize: 8, color: COLOR_TEXT });

    slide.addText("计算内核", { x: 5.0, y: 3.4, fontSize: 9, bold: true, color: COLOR_TEXT });
    slide.addText([
        { text: "降维 (PCA, UMAP)", options: { bullet: true } },
        { text: "聚类 (图聚类, KMeans)", options: { bullet: true } },
        { text: "特征选择 (HVG, VST)", options: { bullet: true } }
    ], { x: 5.0, y: 3.6, w: 1.4, fontSize: 8, color: COLOR_TEXT });


    // Right Column
    // Performance
    slide.addShape(pptx.shapes.RECTANGLE, { x: 6.7, y: 1.0, w: 3.0, h: 2.0, fill: { color: COLOR_BG_GRAY }, line: { color: "DDDDDD" } });
    slide.addShape(pptx.shapes.LINE, { x: 6.7, y: 1.0, w: 0, h: 2.0, line: { color: COLOR_ACCENT, width: 3 } });
    slide.addText("性能指标", { x: 6.8, y: 1.1, fontSize: 11, bold: true, color: COLOR_ACCENT });

    // Stat Boxes
    slide.addShape(pptx.shapes.RECTANGLE, { x: 6.9, y: 1.4, w: 2.6, h: 0.5, fill: { color: COLOR_ACCENT } });
    slide.addText("代码规模", { x: 7.0, y: 1.5, fontSize: 9, color: "FFFFFF" });
    slide.addText("1.8万行", { x: 8.5, y: 1.5, fontSize: 14, bold: true, color: "FFFFFF", align: "right" });

    slide.addShape(pptx.shapes.RECTANGLE, { x: 6.9, y: 2.0, w: 2.6, h: 0.5, fill: { color: COLOR_ACCENT } });
    slide.addText("稀疏加速比", { x: 7.0, y: 2.1, fontSize: 9, color: "FFFFFF" });
    slide.addText("20倍", { x: 8.5, y: 2.1, fontSize: 14, bold: true, color: "FFFFFF", align: "right" });

    // Chart
    slide.addChart(pptx.charts.BAR, [{
        name: "执行时间",
        labels: ["加载", "质控", "标准化", "插补"],
        values: [0.5, 1.2, 0.8, 3.5]
    }], {
        x: 6.7, y: 3.2, w: 3.0, h: 2.0,
        chartColors: ["0066CC"],
        showTitle: true, title: "流水线延迟 (秒)", titleFontSize: 10,
        valAxisFontSize: 8, catAxisFontSize: 8,
        showLegend: false, barDir: 'col'
    });


    // Footer
    slide.addText("ScpTensor 团队内部资料 | 仅供评审使用", { x: 0, y: 5.3, w: '100%', align: 'center', fontSize: 8, color: "666666" });

    await pptx.writeFile({ fileName: 'ScpTensor_Status_Report_CN.pptx' });
    console.log('Presentation created successfully: ScpTensor_Status_Report_CN.pptx');
}

createPresentation().catch(err => {
    console.error(err);
    process.exit(1);
});
