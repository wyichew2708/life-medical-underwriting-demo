import test from 'node:test';
import assert from 'node:assert/strict';
import {reviewItems,sourceFileIndex} from '../dist/review-model.js';
test('blocking issues precede advisory warnings and source checks cannot resolve them',()=>{
 const run={sourceChecks:['OBS-1'],steps:{evidence:{data:{complete:false,findings:[],warnings:['Inspect legibility'],observation_review:{timeline:[],issues:[{text:'Conflicting values',ids:['OBS-1']}]}}}},customRuleResults:[]};
 const r=reviewItems(run);assert.equal(r.issues[0].blocking,true);assert.equal(r.issues[1].blocking,false);assert.deepEqual(r.issues[0].targets,['OBS-1']);
});
test('incomplete evidence and failed output checks remain visible without observation issues',()=>{
 const r=reviewItems({steps:{evidence:{data:{complete:false,findings:[]}},verify:{data:{passed:false}}}});
 assert.deepEqual(r.issues.map(x=>x.key),['INCOMPLETE','VERIFY']);
});
test('viewer only resolves strictly formatted document IDs to existing uploads',()=>{
 const files=[{},{}];assert.equal(sourceFileIndex({document_id:'DOC-2'},files),1);
 for(const id of ['DOC-0','DOC-3','DOC-1evil','https://example.com','DOC-01'])assert.equal(sourceFileIndex({document_id:id},files),-1);
});
test('fast path supports synthetic evidence but never fabricates local extracted records',()=>{
 assert.equal(reviewItems({steps:{}}).items.length,0);
 assert.equal(reviewItems({steps:{},syntheticEvidence:{findings:[{id:'DOC-1',page:1,text:'Example'}]}}).items[0].source.document_id,'DOC-1');
});
