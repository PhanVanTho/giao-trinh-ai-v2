import sys, json
sys.stdout.reconfigure(encoding='utf-8')
d = json.load(open(r'd:\tu_dong_giao_trinh\du_lieu\dau_ra\json\bf3899c3-6574-4eb7-bc0a-eeadf70a0e8c.json', 'r', encoding='utf-8'))
print('Keys:', list(d.keys())[:15])
for k in list(d.keys())[:10]:
    v = d[k]
    if isinstance(v, str):
        print(f'  {k}: {v[:200]}')
    elif isinstance(v, list):
        first = str(v[0])[:200] if v else "empty"
        print(f'  {k}: list[{len(v)}] first={first}')
    elif isinstance(v, dict):
        print(f'  {k}: dict keys={list(v.keys())[:10]}')
    else:
        print(f'  {k}: {type(v).__name__} = {str(v)[:100]}')
