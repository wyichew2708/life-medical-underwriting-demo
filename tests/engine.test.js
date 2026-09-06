import test from 'node:test';
import assert from 'node:assert/strict';
import {profiles,eligible,mockML,referral} from '../dist/engine.js';
const p=profiles[0], ml=mockML(p);
test('standard life and medical profiles pass STP',()=>{for(const p of profiles.slice(0,2))assert.equal(eligible(p,mockML(p),true),true)});
test('substandard, missing evidence, OOD and ML outage profiles do not pass STP',()=>{for(const p of profiles.slice(2))assert.equal(eligible(p,mockML(p),p.evidence),false)});
test('confidence threshold and invalid confidence fail safely',()=>{assert.equal(eligible(p,{...ml,confidence:.95},true),true);for(const confidence of [.949,1.1,NaN,'0.99',null])assert.equal(eligible(p,{...ml,confidence},true),false)});
test('all fast path certifications must be explicit',()=>{for(const key of ['available','standard','calibrated'])assert.equal(eligible(p,{...ml,[key]:false},true),false);assert.equal(eligible(p,{...ml,ood:undefined},true),false);assert.equal(eligible(p,ml,false),false)});
test('business rules override ML confidence and calibration',()=>{for(const change of [{age:76},{cover:1000001},{smoker:true},{condition:'controlled'}])assert.equal(eligible({...p,...change},ml,true),false);assert.equal(referral({...p,cover:1000000}),false)});

