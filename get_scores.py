import json

nb = json.load(open('Traffic_Demand_Prediction.ipynb', encoding='utf-8'))
with open('scores.txt', 'w', encoding='utf-8') as f:
    for c in nb['cells']:
        if 'outputs' in c:
            for out in c['outputs']:
                if out.get('name') == 'stdout':
                    f.write(''.join(out.get('text', '')))
