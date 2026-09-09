// Structured observations are evidence, never underwriting thresholds.
export const units={hba1c:'%',egfr:'mL/min/1.73m²',ldl:'mmol/L',systolic:'mmHg',diastolic:'mmHg'};
export function assessObservations(observations=[],profile={}){
 const issues=[],seen=new Set();
 if(!Array.isArray(observations)||observations.length>100)throw Error('Invalid observation list.');
 const rows=observations.map(o=>{
  if(!o||typeof o.id!=='string'||!/^OBS-[A-Za-z0-9_-]{1,60}$/.test(o.id)||seen.has(o.id)||!Object.hasOwn(units,o.field))throw Error('Invalid or duplicate observation.');
  seen.add(o.id);
  if(typeof o.raw_value!=='string'||o.raw_value.length>100||!(o.raw_unit===null||typeof o.raw_unit==='string'&&o.raw_unit.length<=60))throw Error('Invalid raw observation.');
  if(!o.source||typeof o.source.document_id!=='string'||!Number.isInteger(o.source.page)||o.source.page<1||typeof o.source.quote!=='string'||!o.source.quote.trim()||o.source.quote.trim().length>1000)throw Error('Observation requires source attribution.');
  const date=o.observed_at;
  if(date!==null&&(typeof date!=='string'||!/^\d{4}-\d{2}-\d{2}$/.test(date)||!Number.isFinite(Date.parse(date))||new Date(date).toISOString().slice(0,10)!==date))throw Error('Invalid observation date.');
  const validNumber=/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(o.raw_value.trim());
  const value=validNumber?Number(o.raw_value):null;
  const normalized=Number.isFinite(value)&&o.raw_unit===units[o.field]?value:null;
  if(normalized===null)issues.push({code:'unresolved_value',ids:[o.id],text:o.id+': verify the number and unit against the source.'});
  if(date===null)issues.push({code:'missing_date',ids:[o.id],text:o.id+': obtain the observation date before comparing results.'});
  if(normalized!==null&&typeof profile[o.field]==='number'&&profile[o.field]!==normalized)issues.push({code:'declaration_difference',ids:[o.id],text:o.id+': differs from the current declaration; confirm timing and source.'});
  return {...o,normalized_value:normalized,normalized_unit:normalized===null?null:units[o.field],status:'extracted_unverified'};
 });
 for(let i=0;i<rows.length;i++)for(let j=i+1;j<rows.length;j++){
  const a=rows[i],b=rows[j];
  if(a.field===b.field&&a.observed_at!==null&&a.observed_at===b.observed_at&&a.normalized_value!==null&&b.normalized_value!==null&&a.normalized_value!==b.normalized_value)issues.push({code:'conflicting_results',ids:[a.id,b.id],text:a.id+' and '+b.id+': different '+a.field+' results on the same date; resolve with the source provider.'});
 }
 return {observations:rows,issues,blocking:issues.length>0,timeline:[...rows].sort((a,b)=>(a.observed_at||'9999').localeCompare(b.observed_at||'9999')||a.id.localeCompare(b.id))};
}
export function syntheticObservations(profile,scenario='as_declared'){
 const rows=Object.keys(units).filter(k=>profile[k]!==null&&profile[k]!==undefined).map((field,i)=>({id:'OBS-'+(i+1),field,raw_value:String(profile[field]),raw_unit:units[field],observed_at:'2026-08-01',source:{document_id:'DOC-2',page:1,quote:'Synthetic '+field+': '+profile[field]+' '+units[field]+'; observed 2026-08-01'},synthetic:true}));
 if(rows.length&&scenario==='conflict')rows.push({...structuredClone(rows[0]),id:'OBS-CONFLICT',raw_value:String(Number(rows[0].raw_value)+1),source:{document_id:'DOC-3',page:1,quote:'Synthetic conflicting '+rows[0].field+': '+(Number(rows[0].raw_value)+1)+' '+rows[0].raw_unit+'; observed 2026-08-01'}});
 if(rows.length&&scenario==='missing_date'){rows[0].observed_at=null;rows[0].source.quote='Synthetic '+rows[0].field+': '+rows[0].raw_value+' '+rows[0].raw_unit+'; date absent';}
 if(rows.length&&scenario==='unknown_unit'){rows[0].raw_unit=null;rows[0].source.quote='Synthetic '+rows[0].field+': '+rows[0].raw_value+'; unit absent';}
 if(rows.length&&scenario==='timeline'){const o=rows[0];rows.push({...structuredClone(o),id:'OBS-HISTORY',raw_value:String(Number(o.raw_value)+.5),observed_at:'2026-02-01',source:{document_id:'DOC-3',page:1,quote:'Synthetic earlier '+o.field+': '+(Number(o.raw_value)+.5)+' '+o.raw_unit+'; observed 2026-02-01'}});}
 return rows;
}
export function prepareEvidence(evidence,profile){
 const review=assessObservations(evidence.observations||[],profile);
 if(!review.observations.length)review.issues.push({code:'no_observations',ids:[],text:'No structured measurements were extracted. Review narrative evidence and obtain measurements only if required by the applicable policy.'});
 review.blocking=review.issues.length>0;
 return {...evidence,observations:review.observations,observation_review:review,complete:evidence.complete===true&&!review.blocking};
}
