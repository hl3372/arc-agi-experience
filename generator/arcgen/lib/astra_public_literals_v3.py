"""Recognize only complete explicit row-literal constructions, not general prose."""
import re
PREFIX=re.compile(r'^Rows top-to-bottom(?:, with slashes separating rows)?:\s*',re.I)
TOKEN=re.compile(r'@([a-z][a-z0-9_]*)')

def literal_grid(text,roles):
    match=PREFIX.match(text.strip())
    if not match:return None
    body=text.strip()[match.end():].strip().removesuffix('.').strip()
    rows=body.split('/')
    grids=[]
    for row in rows:
        row=row.strip()
        if re.fullmatch(r'[0-9\s]+',row):
            cells=[int(v) for v in row if v.isdigit()]
        elif re.fullmatch(r'(?:@[a-z][a-z0-9_]*\s*)+',row):
            refs=TOKEN.findall(row)
            if any(x not in roles for x in refs):return None
            cells=[roles[x] for x in refs]
        else:return None
        if not cells or len(cells)>30:return None
        grids.append(cells)
    if not 1<=len(grids)<=30 or len({len(r) for r in grids})!=1:return None
    return grids

def public_literal_checks(public,task):
    roles={r['role']:r['color'] for r in public['episode_overview']['functional_color_roles']}
    results=[]
    for split,key in [('train','training_pairs'),('test','test_inputs')]:
        for i,story in enumerate(public[key]):
            for side in (['input','output'] if split=='train' else ['input']):
                expected=literal_grid(story['exact_'+side+'_construction'],roles)
                if expected is None:continue
                actual=task[split][i][side]
                same_dims=len(expected)==len(actual) and len(expected[0])==len(actual[0])
                diff=([{'row_one_based':r+1,'column_one_based':c+1,'contract':v,'actual':actual[r][c]}
                       for r,row in enumerate(expected) for c,v in enumerate(row) if actual[r][c]!=v] if same_dims else [])
                results.append({'grid':f'{split}_{i+1}_{side}','matched':same_dims and not diff,
                                'dimensions_match':same_dims,'mismatches':diff})
    return results
