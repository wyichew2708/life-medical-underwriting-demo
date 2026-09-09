"""Bounded, attributable numeric observations; no clinical interpretation."""
import re, math
from datetime import date
UNITS={'hba1c':'%','egfr':'mL/min/1.73m²','ldl':'mmol/L','systolic':'mmHg','diastolic':'mmHg'}

def review_observations(items, profile, pages=None, hashes=None):
    if not isinstance(items,list) or len(items)>100: raise ValueError('Invalid observation list.')
    rows=[]; issues=[]; seen=set()
    for o in items:
        if not isinstance(o,dict) or not isinstance(o.get('id'),str) or not re.fullmatch(r'OBS-[A-Za-z0-9_-]{1,60}',o['id']) or o['id'] in seen or o.get('field') not in UNITS: raise ValueError('Invalid or duplicate observation.')
        seen.add(o['id']); raw=o.get('raw_value'); unit=o.get('raw_unit'); observed=o.get('observed_at'); s=o.get('source')
        if not isinstance(raw,str) or len(raw)>100 or (unit is not None and (not isinstance(unit,str) or len(unit)>60)): raise ValueError('Invalid raw observation.')
        if not isinstance(s,dict) or not isinstance(s.get('document_id'),str) or type(s.get('page')) is not int or s['page']<1 or not isinstance(s.get('quote'),str) or not 1<=len(s['quote'].strip())<=1000: raise ValueError('Observation requires source attribution.')
        if pages is not None and (s['document_id'] not in pages or s['page']>pages[s['document_id']]): raise ValueError('Invalid observation page citation.')
        if observed is not None:
            if not isinstance(observed,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',observed): raise ValueError('Invalid observation date.')
            date.fromisoformat(observed)
        number=float(raw) if re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)',raw.strip()) else None
        value=number if number is not None and math.isfinite(number) and unit==UNITS[o['field']] else None
        def issue(code,text): issues.append({'code':code,'ids':[o['id']],'text':o['id']+': '+text})
        if value is None: issue('unresolved_value','verify the number and unit against the source.')
        if observed is None: issue('missing_date','obtain the observation date before comparing results.')
        if value is not None and type(profile.get(o['field'])) in (int,float) and profile[o['field']]!=value: issue('declaration_difference','differs from the current declaration; confirm timing and source.')
        source={k:s[k] for k in ('document_id','page','quote')}
        if hashes is not None: source['sha256']=hashes[s['document_id']]
        rows.append({'id':o['id'],'field':o['field'],'raw_value':raw,'raw_unit':unit,'observed_at':observed,'source':source,'normalized_value':value,'normalized_unit':UNITS[o['field']] if value is not None else None,'status':'extracted_unverified'})
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            if a['field']==b['field'] and a['observed_at'] is not None and a['observed_at']==b['observed_at'] and a['normalized_value'] is not None and b['normalized_value'] is not None and a['normalized_value']!=b['normalized_value']:
                issues.append({'code':'conflicting_results','ids':[a['id'],b['id']],'text':a['id']+' and '+b['id']+': different '+a['field']+' results on the same date; resolve with the source provider.'})
    if not rows: issues.append({'code':'no_observations','ids':[],'text':'No structured measurements extracted; review evidence requirements.'})
    return {'observations':rows,'issues':issues,'blocking':bool(issues)}

def enforce_observations(evidence,profile,pages=None,hashes=None):
    review=review_observations(evidence.get('observations',[]),profile,pages,hashes)
    return {**evidence,'observations':review['observations'],'observation_review':review,'complete':evidence.get('complete') is True and not review['blocking']}
