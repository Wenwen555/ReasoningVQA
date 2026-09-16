#!/usr/bin/env python3
"""Independent re-score of structured prediction artifacts."""
import argparse, json, re, hashlib
from collections import defaultdict
from pathlib import Path

def parse(s):
    try:return json.loads(str(s).strip())
    except Exception:
        m=re.search(r'\{.*\}',str(s),re.S)
        if m:
            try:return json.loads(m.group(0))
            except Exception:pass
    return None

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-rows',type=int);p.add_argument('--mode',required=True)
    a=p.parse_args();
    if a.output.exists():raise FileExistsError(a.output)
    ms=[json.loads(x) for x in a.manifest.read_text(encoding='utf-8').splitlines() if x.strip()]
    ps=[json.loads(x) for x in a.predictions.read_text(encoding='utf-8').splitlines() if x.strip()]
    expected_rows=len(ms) if a.expected_rows is None else a.expected_rows
    errors=[]
    if len(ms)!=expected_rows or len(ps)!=expected_rows:errors.append(f'rows:{len(ms)}/{len(ps)}')
    by={x['sample_id']:x for x in ms}; groups=defaultdict(list); recomputed={}
    for pred in ps:
        sid=pred.get('sample_id');row=by.get(sid)
        if row is None:errors.append(f'unknown:{sid}');continue
        if pred.get('mode')!=a.mode:errors.append(f'mode:{sid}')
        if a.mode=='correct-image' and pred.get('used_image_id')!=row['image_id']:errors.append(f'image:{sid}')
        if a.mode=='no-image' and pred.get('used_image_id') is not None:errors.append(f'noimage:{sid}')
        x=parse(pred.get('raw_output'));required={'visual_features','root_choice','evidence_ids','hops','answer_choice'}
        schema=int(isinstance(x,dict) and set(x)==required and isinstance(x.get('visual_features'),list) and isinstance(x.get('evidence_ids'),list) and isinstance(x.get('hops'),list))
        root=int(schema and x['root_choice']==row['gold_root_choice']);answer=int(schema and x['answer_choice']==row['gold_answer_choice'])
        pe=set(map(str,x['evidence_ids'])) if schema else set();ge=set(map(str,row['gold_evidence_ids']))
        pr=len(pe&ge)/len(pe) if pe else 0;rc=len(pe&ge)/len(ge) if ge else 0;f1=2*pr*rc/(pr+rc) if pr+rc else 0
        evexact=int(schema and x['evidence_ids']==row['gold_evidence_ids']);ph=x['hops'] if schema else [];gh=row['gold_path']
        edge=sum(int(i<len(ph) and ph[i]==e) for i,e in enumerate(gh))/len(gh)
        hops_are_objects=all(isinstance(edge,dict) for edge in ph)
        cont=int(schema and len(ph)==len(gh) and hops_are_objects and all(ph[i-1].get('object')==ph[i].get('subject') for i in range(1,len(ph))))
        pexact=int(schema and ph==gh);full=int(schema and root and evexact and pexact and answer)
        vals={'schema_valid':schema,'root_correct':root,'evidence_f1':f1,'evidence_exact':evexact,'path_edge_accuracy':edge,'path_continuous':cont,'path_exact':pexact,'answer_correct':answer,'full_chain':full}
        for k,v in vals.items():
            if isinstance(v,float): ok=abs(float(pred.get(k,-99))-v)<1e-9
            else: ok=pred.get(k)==v
            if not ok:errors.append(f'score:{sid}:{k}')
        recomputed[sid]=vals;groups[(row['group_id'],row['hop_count'])].append((row,pred))
    for key,items in groups.items():
        rot=int(len(items)==4 and {x[0]['rotation'] for x in items}=={0,1,2,3} and all(recomputed[x[0]['sample_id']]['root_correct'] and recomputed[x[0]['sample_id']]['answer_correct'] for x in items))
        for row,pred in items:
            if pred.get('rotation_consistency')!=rot:errors.append(f'rotation:{row["sample_id"]}')
    report={'status':'passed' if not errors else 'failed','errors':errors,'rows':len(ps),'mode':a.mode,'schema_valid':sum(x['schema_valid'] for x in recomputed.values()),'root_correct':sum(x['root_correct'] for x in recomputed.values()),'path_exact':sum(x['path_exact'] for x in recomputed.values()),'answer_correct':sum(x['answer_correct'] for x in recomputed.values()),'full_chain':sum(x['full_chain'] for x in recomputed.values()),'predictions_sha256':hashlib.sha256(a.predictions.read_bytes()).hexdigest()}
    a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,sort_keys=True));raise SystemExit(0 if not errors else 1)
if __name__=='__main__':main()
