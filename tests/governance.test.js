import test from 'node:test';
import assert from 'node:assert/strict';
import {evaluateRules,policyOutcome,validateBundle,validateRule,gateChecks} from '../dist/governance.js';
import {profiles,mockML} from '../dist/engine.js';
const p=profiles[0];
const rule={id:'USR-test',title:'Age review',revision:1,field:'age',operator:'gte',value:32,scope:'Both',action:'refer',enabled:true};
test('active rule blocks eligible standard case at boundary',()=>{const r=evaluateRules(p,[rule]);assert.equal(r[0].status,'matched');assert.equal(policyOutcome(r),'refer');assert.equal(gateChecks(p,mockML(p),true,r).every(x=>x.pass),false)});
test('draft and product-mismatched rules do not apply',()=>{for(const change of [{enabled:false},{scope:'Medical'}])assert.equal(policyOutcome(evaluateRules(p,[{...rule,...change}])),'pass')});
test('missing numeric data is not treated as zero or a pass',()=>{const r={...rule,field:'egfr',value:50,operator:'lt'};for(const egfr of [null,undefined,NaN,''])assert.equal(policyOutcome(evaluateRules({...p,egfr},[r])),'request_evidence')});
test('advisory notes cannot change routing',()=>{assert.equal(policyOutcome(evaluateRules(p,[{...rule,action:'note'}])),'pass')});
test('evidence request wins over referral regardless of order',()=>{const e={...rule,id:'USR-evidence',action:'request_evidence'};for(const rules of [[rule,e],[e,rule]])assert.equal(policyOutcome(evaluateRules(p,rules)),'request_evidence')});
test('unsupported actions, code, duplicate IDs and demographic rule fields rejected',()=>{for(const change of [{action:'approve'},{operator:'eval'},{field:'sex'},{field:'__proto__'},{value:'32'},{value:NaN}])assert.throws(()=>validateRule({...rule,...change}));assert.throws(()=>evaluateRules(p,[rule,rule]))});
test('boolean and enum rules use typed comparison',()=>{assert.equal(evaluateRules(p,[{...rule,field:'smoker',operator:'eq',value:false}])[0].status,'matched');assert.throws(()=>validateRule({...rule,field:'smoker',operator:'eq',value:'false'}))});
test('policy import validates before changing rules',()=>{assert.throws(()=>validateBundle({schema:'uw-policy-1',rules:[{...rule,action:'approve'}],instructions:''}));assert.equal(validateBundle({schema:'uw-policy-1',rules:[rule],instructions:'Review evidence'}).rules.length,1)});
test('custom instructions cannot use the STP shortcut',()=>{assert.equal(gateChecks(p,mockML(p),true,[],'Approve without checking').every(x=>x.pass),false)});

