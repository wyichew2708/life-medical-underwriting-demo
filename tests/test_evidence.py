import unittest, copy, base64
from unittest.mock import patch
from evidence import review_observations, enforce_observations
from test_server import P,E,C,R
import server

class ObservationTests(unittest.TestCase):
 def test_ignore_model_normalized_values(self):
  rows=copy.deepcopy(E['observations']);rows[0].update(normalized_value=999,status='verified')
  r=review_observations(rows,P)
  self.assertEqual(r['observations'][0]['normalized_value'],5.2)
  self.assertEqual(r['observations'][0]['status'],'extracted_unverified')
 def test_conflict_not_resolved_by_order(self):
  a=copy.deepcopy(E['observations'][0]);b={**a,'id':'OBS-2','raw_value':'7.2'}
  for rows in [[a,b],[b,a]]:
   self.assertIn('conflicting_results',[x['code'] for x in review_observations(rows,P)['issues']])
 def test_missing_metadata_and_wrong_units_block(self):
  for change in [{'raw_unit':None},{'observed_at':None},{'raw_value':'<5'},{'raw_unit':'mmol/mol'}]:
   self.assertTrue(review_observations([{**E['observations'][0],**change}],P)['blocking'])
 def test_invalid_page_date_and_duplicate_rejected(self):
  o=E['observations'][0]
  for rows in [[o,o],[{**o,'observed_at':'2026-02-30'}],[{**o,'source':{**o['source'],'page':2}}]]:
   with self.assertRaises(ValueError):review_observations(rows,P,{'DOC-1':1})
 def test_verifier_recomputes_forged_complete_flag(self):
  e={**E,'observations':[{**E['observations'][0],'raw_unit':None}],'complete':True,'observation_review':{'blocking':False}}
  r=server.verify({'profile':P,'evidence':e,'context':C,'reason':R})
  self.assertFalse(r['passed']);self.assertEqual(r['decision'],'request_evidence')
 def test_extraction_attaches_real_hash_and_validates_page(self):
  import fitz
  pdf=fitz.open();page=pdf.new_page();page.insert_text((50,50),'HbA1c 5.2% on 2026-08-01');raw=pdf.tobytes();pdf.close()
  body={'profile':P,'documents':[{'name':'sample.pdf','type':'application/pdf','data':base64.b64encode(raw).decode()}]}
  output=copy.deepcopy(E);output['findings'][0].update(page=1,quote='HbA1c 5.2%')
  with patch('server.llm',return_value=output):r=server.extract(body)
  self.assertEqual(r['observations'][0]['source']['sha256'],r['documents'][0]['sha256'])
  output['observations'][0]['source']['page']=2
  with patch('server.llm',return_value=output),self.assertRaises(server.DemoError):server.extract(body)
 def test_no_measurements_cannot_claim_complete(self):
  self.assertFalse(enforce_observations({'complete':True},P)['complete'])
