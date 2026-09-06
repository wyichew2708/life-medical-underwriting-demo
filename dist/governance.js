import {extraAttributes,coreAttributes} from './attributes.js';
export {extraAttributes};
export const ruleFields={...coreAttributes,...extraAttributes};
export function validateRule(r){
 if(!r||typeof r!=='object')throw Error('Invalid rule.');
 if(!/^USR-[A-Za-z0-9_-]{1,50}$/.test(r.id))throw Error('Invalid rule ID.');
 if(typeof r.title!=='string'||!r.title.trim()||r.title.length>120)throw Error('Enter a rule title (1–120 characters).');
 const f=Object.hasOwn(ruleFields,r.field)?ruleFields[r.field]:null;
 if(!f)throw Error('Unsupported rule field.');
 if(!['Life','Medical','Both'].includes(r.scope)||!['refer','request_evidence','note'].includes(r.action)||typeof r.enabled!=='boolean')throw Error('Invalid scope, action or status.');
 if(!Number.isInteger(r.revision)||r.revision<1)throw Error('Invalid rule revision.');
 const ops=f.type==='number'?['gt','gte','lt','lte','eq','ne']:['eq','ne'];
 if(!ops.includes(r.operator))throw Error('Unsupported comparison.');
 if(f.type==='number'&&(!Number.isFinite(r.value)||r.value<f.min||r.value>f.max))throw Error('Invalid numeric rule value.');
 if(f.type==='boolean'&&typeof r.value!=='boolean')throw Error('Use true or false.');
 if(f.type==='enum'&&!f.values.includes(r.value))throw Error('Invalid comparison value.');
 return r;
}
export function evaluateRules(profile,rules=[]){
 if(!Array.isArray(rules)||rules.length>50)throw Error('Maximum 50 custom rules.');
 const seen=new Set();
 return rules.map(r=>{
  validateRule(r);if(seen.has(r.id))throw Error('Duplicate rule ID.');seen.add(r.id);
  let status='inactive';const actual=profile[r.field];
  if(r.enabled&&(r.scope==='Both'||r.scope===profile.product)){
   const f=ruleFields[r.field],missing=actual===null||actual===undefined||actual===''||(f.type==='number'&&(!Number.isFinite(actual)||actual<f.min||actual>f.max))||(f.type==='boolean'&&typeof actual!=='boolean')||(f.type==='enum'&&!f.values.includes(actual));
   status=missing?'missing':({gt:()=>actual>r.value,gte:()=>actual>=r.value,lt:()=>actual<r.value,lte:()=>actual<=r.value,eq:()=>actual===r.value,ne:()=>actual!==r.value}[r.operator]()?'matched':'not_matched');
  }
  return {...r,status,actual:actual??null,blocks:r.action!=='note'&&['matched','missing'].includes(status)};
 });
}
export function policyOutcome(results){
 if(results.some(r=>r.blocks&&(r.action==='request_evidence'||r.status==='missing')))return 'request_evidence';
 return results.some(r=>r.blocks)?'refer':'pass';
}
export function validateBundle(b){
 if(!b||b.schema!=='uw-policy-1'||!Array.isArray(b.rules)||typeof b.instructions!=='string'||b.instructions.length>3000)throw Error('Invalid policy bundle.');
 evaluateRules({},b.rules);
 return {schema:'uw-policy-1',rules:b.rules.map(r=>({...validateRule(r)})),instructions:b.instructions};
}
export function trustSummary(run,mode){
 const ml=run.steps.ml?.data,e=run.steps.evidence?.data,v=run.steps.verify?.data;
 return [
 ['ML class confidence',ml?.available?Math.round(ml.confidence*100)+'%':'Unavailable',mode==='mock'?'Scripted example; not measured accuracy.':'Adapter-reported probability; not the probability the final decision is correct.'],
 ['Calibration evidence',mode==='mock'?'Simulated':ml?.validation_report?'Report supplied':'Not supplied','A calibration flag alone is not independent validation.'],
 ['Document coverage',e?new Set(e.findings.map(f=>f.id)).size+' source IDs':run.route?.includes('straight-through')?'Adapter-certified':'Not assessed','Counts observed evidence; does not measure clinical completeness.'],
 ['Evidence warnings',e?String(e.warnings?.length||0):'Not assessed','Unreadable values and conflicts need source review.'],
 ['Output checks',v?v.passed===false?'Failed':'Passed':'Pending','Schema and source-ID checks do not establish factual correctness.'],
 ['Outcome validation','Not measured','No holdout mortality, claims, fairness or decision-accuracy study is bundled.']
 ];
}


export function gateChecks(p,ml,evidence,custom=[],instructions=''){
 return [
 {label:'ML service available',pass:ml.available===true},
 {label:'Standard classification',pass:ml.standard===true},
 {label:'Confidence at least 0.95',pass:Number.isFinite(ml.confidence)&&ml.confidence>=.95&&ml.confidence<=1},
 {label:'Calibration asserted by model adapter',pass:ml.calibrated===true},
 {label:'In-distribution',pass:ml.ood===false},
 {label:'Required evidence complete',pass:evidence===true},
 {label:'Built-in referral rules clear',pass:!(p.age>75||(p.product==='Life'&&p.cover>1000000)||p.smoker||p.condition!=='none')},
 {label:'Custom restrictions clear',pass:policyOutcome(custom)==='pass'},
 {label:'No additional reasoning instructions',pass:!instructions.trim()}
 ];
}

