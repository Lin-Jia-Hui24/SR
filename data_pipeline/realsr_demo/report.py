"""A self-contained local HTML report, with original evidence files alongside."""
import csv
import html
import json
import math
from pathlib import Path


def build_report(output):
    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    train = json.loads((output / "train/summary.json").read_text(encoding="utf-8"))
    test = json.loads((output / "test/test_summary.json").read_text(encoding="utf-8"))
    with (output / "test/test_per_image.csv").open(encoding="utf-8") as stream:
        per_image = list(csv.DictReader(stream))
    e = html.escape
    table = ""
    for group, row in test["aggregation"].items():
        bicubic_psnr, model_psnr = float(row["bicubic_psnr_rgb"]), float(row["model_psnr_rgb"])
        difference = model_psnr - bicubic_psnr
        # Infinity - Infinity is undefined; do not invent a zero improvement.
        delta = "—" if math.isnan(difference) else f"{difference:+.4f}"
        table += f'<tr><td>{e(group)}</td><td>{row["images"]}</td><td>{bicubic_psnr:.4f}</td><td>{model_psnr:.4f}</td><td>{delta}</td><td>{float(row["bicubic_ssim_rgb"]):.6f}</td><td>{float(row["model_ssim_rgb"]):.6f}</td></tr>'
    all_row = test["aggregation"]["all"]
    example_data = json.dumps(test["examples"], ensure_ascii=False).replace("<", "\\u003c")
    detail_rows = "".join(f'<tr><td>{e(row["pair_id"])}</td><td>{row["lr_width"]} × {row["lr_height"]}</td><td>{float(row["bicubic_psnr_rgb"]):.4f}</td><td>{float(row["model_psnr_rgb"]):.4f}</td><td>{float(row["bicubic_ssim_rgb"]):.6f}</td><td>{float(row["model_ssim_rgb"]):.6f}</td></tr>' for row in per_image)
    inference = e(json.dumps(test["inference_counts"], ensure_ascii=False))
    audit_links = ""
    if (output / "data_audit/audit_preview.json").exists():
        audit = json.loads((output / "data_audit/audit_preview.json").read_text(encoding="utf-8"))
        links = "".join(f'<a href="data_audit/{e(name)}" target="_blank">配对检查 {i+1}</a>' for i, name in enumerate(audit["sheets"]))
        audit_links = f'<section><h2>原始配对抽检：{audit["sample_count"]} 对</h2><p class="muted">Canon／Nikon × 官方 Train／Test 各抽 5 对。查看整图及相同坐标的局部 LR／HR 对照；这些裁剪仅用于检查，不参与本次整图训练和评测。</p><div class="links">{links}<a href="data_audit/audit_preview.json">抽样与坐标记录</a></div><p class="muted">抽查未见明显错配，不能据此证明全部图像完全没有残余配准误差。</p></section>'
    page = f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RealSR V3 ×2 · 无增强基线</title>
<style>
:root{{color-scheme:light;--ink:#14263e;--muted:#5c6d83;--line:#dae3ee;--blue:#245ee9}}*{{box-sizing:border-box}}body{{margin:0;background:#f3f6fb;color:var(--ink);font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1240px;margin:auto;padding:44px 28px 70px}}h1{{font-size:38px;line-height:1.25;margin:8px 0 16px}}h2{{font-size:23px;margin:0 0 16px}}p{{margin:10px 0}}.eyebrow{{color:var(--blue);font-weight:650;letter-spacing:2px;font-size:13px}}.muted{{color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:26px 0}}.card,section{{background:white;border:1px solid var(--line);border-radius:16px;padding:22px}}.card strong{{display:block;font-size:27px}}.card span{{color:var(--muted);font-size:14px}}section{{margin:18px 0;padding:26px}}.notice{{border-left:4px solid #e3952c;padding:12px 18px;background:#fff6e8;border-radius:4px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:13px 10px;text-align:right;border-bottom:1px solid var(--line)}}td:first-child,th:first-child{{text-align:left}}th{{color:var(--muted);font-weight:600;white-space:nowrap}}.scroll{{overflow:auto}}a{{color:var(--blue)}}select,input{{font:inherit;padding:7px 12px;border:1px solid var(--line);border-radius:8px;background:white;color:var(--ink)}}.controls{{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-bottom:16px}}#comparison{{display:block;width:100%;max-width:none}}.imagebox{{overflow:auto;border:1px solid var(--line);border-radius:10px;background:#eef1f6}}.curve{{max-width:800px;width:100%;display:block;margin:auto}}.links{{display:flex;gap:22px;flex-wrap:wrap}}pre{{white-space:pre-wrap;word-break:break-word;background:#f5f7fa;padding:16px;border-radius:10px;font-size:13px}}details{{margin-top:18px}}summary{{cursor:pointer;font-weight:650}}.protocol{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.protocol p{{padding:12px 16px;background:#f6f8fc;border-radius:9px;margin:0}}@media(max-width:760px){{main{{padding:24px 14px}}h1{{font-size:30px}}.cards,.protocol{{grid-template-columns:1fr 1fr}}section{{padding:17px}}}}
</style><main>
<div class="eyebrow">计算机视觉 · 作业八</div><h1>真实图像超分辨率 ×2<br>完整数据，无增强基线</h1>
<p class="muted">RealSR V3 / Canon + Nikon / RGB → RGB。先固定可信的数据与评测流程，再开展增强消融实验。</p>
<div class="cards"><div class="card"><strong>{manifest["counts"]["pairs"]} 对</strong><span>官方 ×2 数据</span></div><div class="card"><strong>{train["used_train_pairs"]} / {train["validation_pairs"]}</strong><span>内部训练 / 验证，按场景划分</span></div><div class="card"><strong>{test["test_images"]} 张</strong><span>官方测试，逐张整图评测</span></div><div class="card"><strong>{train["epochs"]} epochs</strong><span>{e(train["device"].upper())} · batch {train["batch_size"]} · 无增强</span></div></div>
<p class="notice">这是短程训练 Demo，已完成真实数据上的训练与测试；不能据此声称模型已收敛或增强方法有效。所有成绩来自下方同一 RGB 测试口径，不能直接与官方论文的 Y 通道成绩比较。</p>
<section><h2>完整测试集：模型与 Bicubic</h2><p><b>Bicubic：</b>{float(all_row["bicubic_psnr_rgb"]):.4f} dB / {float(all_row["bicubic_ssim_rgb"]):.6f}<br><b>TinySR：</b>{float(all_row["model_psnr_rgb"]):.4f} dB / {float(all_row["model_ssim_rgb"]):.6f}</p><p class="muted">先计算每张图的指标，再对图像取算术平均。分相机结果用于观察差异。</p><div class="scroll"><table><thead><tr><th>分组</th><th>图像数</th><th>Bicubic PSNR</th><th>TinySR PSNR</th><th>Δ PSNR / dB</th><th>Bicubic SSIM</th><th>TinySR SSIM</th></tr></thead><tbody>{table}</tbody></table></div><p class="muted">输出覆盖方式：{inference}。评分使用完整原尺寸 SR/HR；下面展示时的缩放不参与计算。</p></section>
<section><h2>整体与局部细节</h2><p class="muted">预先固定 Canon、Nikon 各前两张测试图，避免按成绩挑图。橙框对应下排同坐标中心区域；LR 用最近邻放大仅便于观察。</p><div class="controls"><label>测试图 <select id="sample"></select></label><label>展示宽度 <select id="zoom"><option value="100%">适合窗口</option><option value="1536px">1536 px</option><option value="2400px">2400 px</option></select></label><a id="original" target="_blank">打开对比大图</a></div><div class="imagebox"><img id="comparison" alt="LR、Bicubic、TinySR、真实HR的整体及局部对照"></div><p id="sampleInfo" class="muted"></p></section>
<section><h2>L1 训练与验证损失</h2><img class="curve" src="train/curve.svg" alt="按epoch统计的训练与验证L1曲线"><p class="muted">最优权重来自第 {train["best_epoch"]} 个 epoch，依据验证集逐图平均 L1={train["best_validation_l1"]:.6f}。本次训练空间模式：{"完整图像" if train["full_image_training"] else "成对裁剪"}；验证空间模式：{e(train["validation_scope"])}。测试集不参与选择权重。</p><div class="links"><a href="train/loss.csv">每 epoch 损失 CSV</a><a href="train/step_loss.csv">每一步损失及图像尺寸</a><a href="train/summary.json">完整训练配置与记录</a></div></section>
<section><h2>固定的测试口径</h2><div class="protocol"><p><b>数据与范围</b><br>原始 RGB8 ÷ 255；真实 LR 输入，真实配准 HR 为目标；输出严格为 2H × 2W × 3。</p><p><b>边界与数值</b><br>PSNR 使用全图、不主动裁边；预测截断至 [0,1]，不做 uint8 量化；目标不改色、不重配准。</p><p><b>RGB PSNR</b><br>全图 H×W×3 个误差共同求 MSE，再计算 10 log₁₀(1/MSE)；最终逐图平均。</p><p><b>RGB SSIM</b><br>三通道分别计算再平均；11×11 高斯窗口，σ=1.5，K₁=.01、K₂=.03，总体协方差。valid 窗口均值固有忽略四周 5 像素。</p><p><b>Bicubic 基线</b><br>PyTorch bicubic ×2，align_corners=False、antialias=False。与模型完全相同的目标、截断和指标。</p><p><b>训练与测试隔离</b><br>官方 Train 内按相机、场景固定 seed={train["seed"]} 划分；官方 Test 保持不动。不使用测试增强或测试图调参。</p></div>
<details><summary>逐图结果（{len(per_image)} 张）</summary><div class="controls"><input id="filter" placeholder="筛选 Canon / Nikon / 图像编号" aria-label="筛选逐图结果"></div><div class="scroll"><table id="details"><thead><tr><th>样本</th><th>LR 宽×高</th><th>Bicubic PSNR</th><th>TinySR PSNR</th><th>Bicubic SSIM</th><th>TinySR SSIM</th></tr></thead><tbody>{detail_rows}</tbody></table></div></details>
<details><summary>机器可读的协议</summary><pre>{e(json.dumps(test["protocol"], ensure_ascii=False, indent=2))}</pre></details></section>
<section><h2>可追溯产物</h2><div class="links"><a href="manifest.json">文件配对与数据划分</a><a href="test/test_per_image.csv">全部测试指标 CSV</a><a href="test/test_summary.json">测试汇总 JSON</a><a href="environment.json">环境版本</a><a href="train/best.pt">模型权重</a><a href="https://github.com/csjcai/RealSR">RealSR 官方来源</a></div><p class="muted">已有增强实现保留在源代码中，本次实验完全关闭；该结果可作为之后单因素增强实验的基线。</p></section>
</main><script>
const examples={example_data}; const sample=document.getElementById('sample');
examples.forEach((x,i)=>{{let option=document.createElement('option');option.value=i;option.textContent=x.pair_id;sample.appendChild(option);}});
function show(){{const x=examples[Number(sample.value)];if(!x)return;const url='test/'+x.comparison;document.getElementById('comparison').src=url;document.getElementById('original').href=url;document.getElementById('sampleInfo').textContent='固定 HR 局部坐标 [x,y,w,h] = '+JSON.stringify(x.hr_roi_xywh)+'；整图 RGB PSNR：Bicubic '+Number(x.metrics.bicubic_psnr_rgb).toFixed(4)+' dB / TinySR '+Number(x.metrics.model_psnr_rgb).toFixed(4)+' dB。';}}
sample.addEventListener('change',show);document.getElementById('zoom').addEventListener('change',e=>document.getElementById('comparison').style.width=e.target.value);
document.getElementById('filter').addEventListener('input',e=>document.querySelectorAll('#details tbody tr').forEach(row=>row.hidden=!row.textContent.toLowerCase().includes(e.target.value.toLowerCase())));show();
</script></html>'''
    page = page.replace("</main>", audit_links + "</main>")
    (output / "index.html").write_text(page, encoding="utf-8")
    return str(output / "index.html")
