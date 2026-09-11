const measurementLabels={hba1c:'Average blood sugar (HbA1c)',egfr:'Kidney function (eGFR)',ldl:'LDL cholesterol',systolic:'Systolic blood pressure',diastolic:'Diastolic blood pressure'};
export function reviewItems(run){
 const evidence=run.steps.evidence?.data||run.syntheticEvidence;
 const observations=evidence?.observation_review?.timeline||[];
 const items=observations.map(o=>({key:o.id,label:measurementLabels[o.field]||o.field,date:o.observed_at,value:o.raw_value+' '+(o.raw_unit||'(unit unknown)'),source:o.source,kind:'measurement'}));
 (evidence?.findings||[]).forEach((f,i)=>items.push({key:'FINDING-'+i,label:f.text,date:null,value:f.source,source:{document_id:f.id,page:f.page,quote:f.quote},kind:'finding',findingIndex:i}));
 const issues=(evidence?.observation_review?.issues||[]).map((x,i)=>({key:'ISSUE-'+i,text:x.text,targets:x.ids||[],blocking:true}));
 (evidence?.warnings||[]).forEach((text,i)=>issues.push({key:'WARNING-'+i,text,targets:[],blocking:false}));
 (run.customRuleResults||[]).filter(r=>r.blocks).forEach(r=>issues.push({key:'RULE-'+r.id,text:r.title+': '+(r.status==='missing'?'required input is unknown':'restriction matched')+'. '+r.action,targets:[],blocking:true}));
 if(evidence?.complete===false&&!issues.some(x=>x.blocking))issues.push({key:'INCOMPLETE',text:'Evidence is incomplete. Obtain the outstanding information and reassess.',targets:[],blocking:true});
 if(run.steps.verify?.data?.passed===false)issues.push({key:'VERIFY',text:'Output checks failed. Inspect the original sources before relying on this recommendation.',targets:[],blocking:true});
 return {evidence,items,issues:issues.sort((a,b)=>Number(b.blocking)-Number(a.blocking))};
}
export function sourceFileIndex(source,files){
 const match=/^DOC-([1-9]\d*)$/.exec(source?.document_id||'');
 const index=match?Number(match[1])-1:-1;
 return index>=0&&index<files.length?index:-1;
}
