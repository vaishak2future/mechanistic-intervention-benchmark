"""Draw the submission figure from saved, completed 45-minute verifier results."""
import json
from pathlib import Path
from statistics import mean
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'sol': 'GPT-6.1 Sol', 'astra': 'GPT-6 Astra',
          'opus': 'Claude Opus 5.5', 'sonnet': 'Claude Sonnet 5.5',
          'inkling': 'Inkling-2-Small (256K)'}


def main():
    rows = json.loads((ROOT / 'runs/mechanistic-submission-45m/summary.json').read_text())['runs']
    rows += json.loads((ROOT / 'runs/inkling-256k-45m/summary.json').read_text())['runs']
    assert len(rows) == 10
    assert all(row['status'] == 'FINISHED' and row['budget_seconds'] == 2700 for row in rows)
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="440" viewBox="0 0 900 440" role="img" aria-labelledby="title description">',
             '<title id="title">Mechanistic Intervention Benchmark: 45-minute pilot</title>',
             '<desc id="description">Mean scores out of 100: Sol 9.09, Astra 4.55, Opus 0, Sonnet 0, Inkling 0. Two trials per model; dots show individual trials, not confidence intervals.</desc>',
             '<rect width="900" height="440" fill="#fbfaf7" rx="16"/>',
             '<g font-family="system-ui, sans-serif" fill="#242c28">',
             '<text x="32" y="40" font-size="22" font-weight="650">Repairing reasoning loops</text>',
             '<text x="32" y="66" font-size="13" fill="#667069">45-minute budget · one H100 · two trials per agent system</text>']
    start, width = 210, 590
    for tick in range(0, 101, 20):
        x = start + width * tick / 100
        lines += [f'<line x1="{x}" y1="100" x2="{x}" y2="345" stroke="#e0e4dc"/>',
                  f'<text x="{x}" y="367" font-size="12" text-anchor="middle" fill="#667069">{tick}</text>']
    for index, (name, label) in enumerate(LABELS.items()):
        values = [100 * row['scores']['reward'] for row in rows if row['name'] == name]
        assert len(values) == 2
        y = 124 + index * 50
        average = mean(values)
        lines += [f'<text x="32" y="{y + 5}" font-size="14">{escape(label)}</text>',
                  f'<rect x="{start}" y="{y - 11}" width="{width * average / 100}" height="22" fill="#3e7464" rx="3"/>']
        for number, value in enumerate(values):
            lines.append(f'<circle cx="{start + width * value / 100}" cy="{y + (number * 8 - 4)}" r="4" fill="#fbfaf7" stroke="#315d50" stroke-width="2"/>')
        lines.append(f'<text x="837" y="{y + 5}" font-size="14" font-weight="600" text-anchor="end">{average:.2f}</text>')
    lines += ['<text x="505" y="392" font-size="13" text-anchor="middle">Net improvement / 100</text>',
              '<text x="32" y="422" font-size="12" fill="#667069">Bars: trial mean. Dots: individual trials. A small pilot, not a statistically established ranking.</text>',
              '</g></svg>']
    (Path(__file__).parent / 'results.svg').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
