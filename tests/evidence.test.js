import test from 'node:test';
import assert from 'node:assert/strict';
import {assessObservations,syntheticObservations,prepareEvidence} from '../dist/evidence.js';
import {profiles,mockML} from '../dist/engine.js';
import {gateChecks} from '../dist/governance.js';
const p=profiles[0];
test('consistent measurements preserve raw evidence and permit existing mock gates',()=>{
 const e=prepareEvidence({complete:true,observations:syntheticObservations(p)},p);
 assert.equal(e.complete,true);assert.equal(e.observations[0].status,'extracted_unverified');
 assert.ok(gateChecks(p,mockML(p),e.complete).every(g=>g.pass));
});
test('conflicts and missing metadata block mock STP despite high confidence',()=>{
 for(const scenario of ['conflict','missing_date','unknown_unit','timeline']){
  const e=prepareEvidence({complete:true,observations:syntheticObservations(p,scenario)},p);
  assert.equal(e.complete,false,scenario);assert.equal(gateChecks(p,mockML(p),e.complete).every(g=>g.pass),false);
 }
});
test('different dates are chronology, not same-date contradictions',()=>{
 const r=assessObservations(syntheticObservations(p,'timeline'),p);
 assert.equal(r.issues.some(x=>x.code==='conflicting_results'),false);
 assert.equal(r.timeline[0].observed_at,'2026-02-01');
});
test('unknown units and inequality values are not silently converted',()=>{
 const o=syntheticObservations(p)[0];
 for(const raw_value of ['<6','NaN','','1e3'])assert.equal(assessObservations([{...o,raw_value}],{}).observations[0].normalized_value,null);
 assert.equal(assessObservations([{...o,raw_unit:'mmol/mol'}],{}).observations[0].normalized_value,null);
});
test('no observations, duplicate IDs, invalid dates and missing sources do not pass',()=>{
 assert.equal(prepareEvidence({complete:true},p).complete,false);
 const o=syntheticObservations(p)[0];
 for(const rows of [[o,o],[{...o,observed_at:'2026-02-30'}],[{...o,source:null}]])assert.throws(()=>assessObservations(rows,p));
});
