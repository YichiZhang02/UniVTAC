"""Export measured training profiles as CSV, JSON, a chart, and a printable report."""
import argparse
import base64
import csv
import html
import json
from pathlib import Path


PRIMARY = [
    ('data_wait_ms', '等待数据'), ('host_to_device_ms', '传输到 GPU'),
    ('tps_ms', 'GPU TPS'), ('forward_ms', '前向合计'),
    ('backward_ddp_ms', '反向＋DDP'), ('clip_ms', '梯度裁剪'),
    ('optimizer_ms', 'AdamW'), ('other_ms', '其他'), ('step_wall_ms', '每步合计'),
]
LABELS = dict(mae_cpu='MAE-S / CPU TPS', mae_cuda='MAE-S / CUDA TPS',
              dinov2_cuda='DINOv2-B', ijepa_cuda='I-JEPA-S', vjepa_cuda='V-JEPA-S',
              anytouch2_cuda='AnyTouch2-B', wan_official_cuda='Wan official',
              wan_adapted_cuda='Wan adapted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--models', nargs='+', choices=list(LABELS), help='Include only the selected profiles')
    args = parser.parse_args()
    results = []
    for name, label in LABELS.items():
        if args.models and name not in args.models:
            continue
        directory = args.directory / name
        if not (directory / 'stage_profile.json').exists():
            continue
        reports = json.loads((directory / 'stage_profile.json').read_text())
        metadata = json.loads((directory / 'profile_metadata.json').read_text())
        config = __import__('yaml').safe_load((directory / 'train_config.yml').read_text())
        means = {key: sum(r['summary'][key]['mean'] for r in reports)/len(reports)
                 for key in reports[0]['summary']}
        row = dict(name=name, label=label, **metadata, frames=config['tactile_num_frames'],
                   measured_steps=reports[0]['measured_steps'], gpu=reports[0]['gpu'],
                   peak_memory_gib=max(r['peak_memory_gib'] for r in reports), means=means,
                   rank_mean_step_ms=[r['summary']['step_wall_ms']['mean'] for r in reports],
                   rank_p90_step_ms=[r['summary']['step_wall_ms']['p90'] for r in reports])
        results.append(row)
    summary = dict(
        methodology='Real HDF5 data; two A100 GPUs; per-GPU batch 16, global batch 32; '
                    '20 steps, first 5 excluded; mean across measured steps and both ranks. '
                    'CUDA synchronizes once per profiled step. Startup/statistics/validation/checkpoint '
                    'I/O excluded. Backward and DDP overlap. Four-GPU performance not measured.',
        results=results)
    (args.directory / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    fields = ['name', 'label', 'world_size', 'batch_per_gpu', 'global_batch', 'frames',
              'measured_steps', 'gpu', 'peak_memory_gib', *results[0]['means']]
    with (args.directory / 'stage_timings.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({**{key: row[key] for key in fields if key not in row['means']}, **row['means']})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(13, 6))
    offsets = np.zeros(len(results))
    names = ['Data wait', 'H2D', 'GPU TPS', 'Forward', 'Backward + DDP', 'Grad clip', 'AdamW', 'Other']
    colors = ['#e76f51', '#a8dadc', '#2a9d8f', '#457b9d', '#264653', '#e9c46a', '#f4a261', '#b8b8b8']
    for (key, _), label, color in zip(PRIMARY[:-1], names, colors):
        values = np.array([r['means'][key] for r in results]) / 1000
        ax.barh(range(len(results)), values, left=offsets, label=label, color=color)
        offsets += values
    for index, row in enumerate(results):
        ax.text(offsets[index]+.02, index, f"{row['means']['step_wall_ms']/1000:.3f}s", va='center', fontsize=9)
    ax.set_yticks(range(len(results)), [row['label'] for row in results])
    ax.invert_yaxis()
    ax.set_xlabel('Seconds per optimizer step')
    ax.set_title('Real data training: 2 x A100, batch 16 per GPU (global 32)')
    ax.set_xlim(0, max(offsets)*1.16)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, -.28), ncol=4, frameon=False)
    fig.tight_layout()
    chart = args.directory / 'stage_timings.png'
    fig.savefig(chart, dpi=160, bbox_inches='tight')
    plt.close(fig)

    def table(columns):
        head = '<tr><th>模型</th>' + ''.join(f'<th>{label}</th>' for _, label in columns) + '</tr>'
        body = []
        for row in results:
            body.append('<tr><td>'+html.escape(row['label'])+'</td>'+''.join(
                f'<td>{row["means"].get(key, 0):.1f}</td>' for key, _ in columns)+'</tr>')
        return '<table>'+head+''.join(body)+'</table>'

    before = next((r for r in results if r['name'] == 'mae_cpu'), None)
    after = next((r for r in results if r['name'] == 'mae_cuda'), None)
    outcome = ''
    if before and after:
        old, new = before['means']['step_wall_ms'], after['means']['step_wall_ms']
        outcome = f'MAE 同口径对照：{old/1000:.3f} → {new/1000:.3f} 秒/step，加速 {old/new:.2f} 倍。'
    image = base64.b64encode(chart.read_bytes()).decode()
    runtime_table = '<table><tr><th>模型</th><th>触觉历史帧</th><th>测量步数/rank</th><th>最慢 rank 的 P90/毫秒</th><th>峰值 allocated/GiB</th></tr>' + ''.join(
        f'<tr><td>{html.escape(r["label"])}</td><td>{r["frames"]}</td><td>{r["measured_steps"]}</td>'
        f'<td>{max(r["rank_p90_step_ms"]):.1f}</td><td>{r["peak_memory_gib"]:.2f}</td></tr>' for r in results) + '</table>'
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>UniVTAC 训练阶段耗时报告 · 2026-10-26</title>
<style>body{{font:15px/1.7 system-ui,sans-serif;color:#172b3a;max-width:1350px;margin:32px auto;padding:0 20px}}
table{{border-collapse:collapse;width:100%;margin:18px 0;font-variant-numeric:tabular-nums}}th,td{{padding:8px;border:1px solid #d6e0e6;text-align:right}}td:first-child,th:first-child{{text-align:left}}th{{background:#eef4f7}}img{{width:100%}}code{{background:#eef4f7;padding:2px 5px}}h1,h2{{line-height:1.3}}@media print{{body{{margin:0;font-size:11px}}}}</style>
<h1>UniVTAC 训练阶段耗时报告</h1><p>2026-10-26 · {html.escape(outcome)}</p>
<h2>测量口径</h2><p>本机 <b>2 × NVIDIA A100-SXM4-80GB</b>；每卡 batch 16，全局 batch 32；
双 RGB 相机、224×224、动作 chunk 50、触觉 pool3x3、端到端训练；8 个任务、171290 个训练样本，
每卡 4 个数据 worker。各配置运行 20 步，去掉前 5 步，统计余下 15 步及两个 rank 的算术平均。
复用已核对 episode split 的统计量。模型从原始预训练权重初始化，没有覆盖正式任务的 checkpoint。</p>
<p><b>1.sh 为 4 卡、全局 batch 64；本报告每卡负载相同，但没有实测四卡通信。</b>
这里测的是完整真实数据训练步骤，不是固定 GPU 张量的合成吞吐。
分析模式每步同步 CUDA，可能带来少量额外开销；普通训练不启用这些同步。</p>
<img alt="各训练阶段耗时" src="data:image/png;base64,{image}">
<h2>每步阶段耗时（毫秒，可按行相加）</h2>{table(PRIMARY)}
<h2>前向计算细分（毫秒，已包含在上表前向合计中）</h2>
{table([('tactile_forward_ms','触觉编码器'),('qwen_forward_ms','Qwen'),('action_forward_ms','动作头'),('forward_ms','前向合计')])}
<p>Qwen 包含图像/语言输入准备；其更细的 qwen_input_preparation_ms 在 JSON/CSV 中。
前向合计另包含路由、loss 和 DDP 前向等少量开销。反向＋DDP 包含梯度计算、重叠通信及跨 rank 等待，
不能当作纯通信时间。数据等待是主进程阻塞等待下一 batch 的可见耗时，数据 worker 的并行工作不能再重复加到每步合计。</p>
<h2>运行信息</h2>{runtime_table}
<h2>原始瓶颈与改动</h2><p>CPU 抽样分析：32 个随机样本，平均 642.6 ms/样本，触觉读取/转换占 620.2 ms，
其中 TPS 转换 547.9 ms。256 帧中 254 次密集 TPS 插值映射缓存未命中，参考 marker 网格逐帧变化导致缓存几乎无效。
每卡 batch 16、4 个 worker 时，供数压力足以把完整步骤推到数秒。</p>
<p>现在 CPU 仅完成小规模 TPS 系数求解和深度准备；GPU 分块批量求值完整形变图。
单样本 CPU 准备约 60 ms。保留重复 marker 合并、深度归一化、历史帧和形变映射。
8 个任务、两侧传感器、两种编码的真实数据对照最大绝对误差为 2.7×10⁻⁷。
新路径由 <code>TACTILE_PREPROCESSING=auto</code> 默认启用，
<code>TACTILE_PREPROCESSING=cpu</code> 可选原始路径。</p>
<h2>范围与复现</h2><p>合计不含启动、冷启动预热、数据统计、验证和 checkpoint 写盘；它们不发生在普通的每一步中，
此次没有实测写盘与验证耗时。不同 Wan checkpoint 的历史帧约定由各自配置保留；每个子目录的 train_config.yml
记录了帧数、encoder checkpoint、数据划分与完整参数。正式长跑、NAS 抖动及四卡拓扑仍可能改变吞吐。</p>
<p>可复用 <code>scripts/profile_starvla_run.py</code> 生成分 rank 原始数据，
<code>scripts/summarize_training_profiles.py</code> 汇总。
本目录包含 <code>stage_timings.csv</code>、<code>summary.json</code>、阶段图和所有分 rank 逐步样本。</p>
</html>'''
    (args.directory / 'report.html').write_text(page)
    print(json.dumps([dict(name=r['name'], step_ms=round(r['means']['step_wall_ms'], 2)) for r in results]))


if __name__ == '__main__':
    main()
