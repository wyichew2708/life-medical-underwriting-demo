"""Typed custom policies. They may restrict routing, never grant automatic approval."""
import json, math, re
from pathlib import Path
SCHEMA=json.loads((Path(__file__).parent/'dist/attributes.json').read_text())
FIELDS={**SCHEMA['core'],**SCHEMA['extra']}
class PolicyError(ValueError): pass

def validate_rule(r):
    if not isinstance(r,dict): raise PolicyError('Invalid rule.')
    if not isinstance(r.get('id'),str) or not re.fullmatch(r'USR-[A-Za-z0-9_-]{1,50}',r['id']): raise PolicyError('Invalid rule ID.')
    if not isinstance(r.get('title'),str) or not 1<=len(r['title'].strip())<=120: raise PolicyError('Invalid title.')
    f=FIELDS.get(r.get('field'))
    if not f: raise PolicyError('Unsupported rule field.')
    if r.get('scope') not in ('Both','Life','Medical') or r.get('action') not in ('refer','request_evidence','note') or type(r.get('enabled')) is not bool: raise PolicyError('Invalid rule action, scope or state.')
    if type(r.get('revision')) is not int or r['revision']<1: raise PolicyError('Invalid rule revision.')
    ops=('gt','gte','lt','lte','eq','ne') if f['type']=='number' else ('eq','ne')
    if r.get('operator') not in ops: raise PolicyError('Unsupported comparison.')
    v=r.get('value')
    if f['type']=='number' and (type(v) not in (int,float) or not math.isfinite(v) or not f['min']<=v<=f['max']): raise PolicyError('Invalid numeric value.')
    if f['type']=='boolean' and type(v) is not bool: raise PolicyError('Invalid Boolean value.')
    if f['type']=='enum' and v not in f['values']: raise PolicyError('Invalid comparison value.')
    return r

def evaluate_rules(profile,rules):
    if not isinstance(rules,list) or len(rules)>50: raise PolicyError('Maximum 50 rules.')
    result=[];seen=set()
    for r in rules:
        validate_rule(r)
        if r['id'] in seen: raise PolicyError('Duplicate rule ID.')
        seen.add(r['id']);actual=profile.get(r['field']);f=FIELDS[r['field']];status='inactive'
        if r['enabled'] and r['scope'] in ('Both',profile['product']):
            missing=actual is None or actual==''
            if f['type']=='number': missing=missing or type(actual) not in (int,float) or not math.isfinite(actual) or not f['min']<=actual<=f['max']
            if f['type']=='boolean': missing=missing or type(actual) is not bool
            if f['type']=='enum': missing=missing or actual not in f['values']
            if missing: status='missing'
            else:
                v=r['value'];op=r['operator']
                matched={'gt':lambda:actual>v,'gte':lambda:actual>=v,'lt':lambda:actual<v,'lte':lambda:actual<=v,'eq':lambda:actual==v,'ne':lambda:actual!=v}[op]()
                status='matched' if matched else 'not_matched'
        result.append({**r,'status':status,'actual':actual,'blocks':r['action']!='note' and status in ('matched','missing')})
    return result

def validate_policy(value,profile):
    if value is None: value={}
    if not isinstance(value,dict): raise PolicyError('Invalid policy.')
    instructions=value.get('instructions','')
    if not isinstance(instructions,str) or len(instructions)>3000: raise PolicyError('Instructions exceed 3000 characters.')
    rules=value.get('rules',[])
    results=evaluate_rules(profile,rules)
    return {'rules':rules,'instructions':instructions,'revision':value.get('revision',1),'results':results}

def policy_outcome(results):
    if any(r['blocks'] and (r['action']=='request_evidence' or r['status']=='missing') for r in results): return 'request_evidence'
    return 'refer' if any(r['blocks'] for r in results) else 'pass'

