import unittest, base64, json, os
from unittest.mock import patch
import server

P={'name':'Test Profile','age':35,'cover':500000,'bmi':23,'sex':'Not specified','occupation':'Engineer','product':'Life','smoker':False,'condition':'none'}
E={'complete':True,'findings':[{'id':'DOC-1','text':'Test finding'}],'warnings':[]}
C={'sources':[]}
R={'recommendation':'refer','explanation':'Human review needed.','reasons':['Evidence requires review.'],'citations':['DOC-1'],'missing_information':[]}
class SafetyTests(unittest.TestCase):
 def test_input_bounds(self):
  for field,value in [('age',float('nan')),('age',True),('cover',-1),('bmi',99)]:
   with self.assertRaises(server.DemoError):server.validate_profile({**P,field:value})
 def test_file_type_spoof(self):
  with self.assertRaises(server.DemoError):server.documents([{'name':'x.pdf','type':'application/pdf','data':base64.b64encode(b'not a pdf').decode()}])
 def test_unknown_citation_and_approval(self):
  self.assertTrue(server.reasoning_errors({**R,'citations':['FAKE']},E,C))
  self.assertTrue(server.reasoning_errors({**R,'recommendation':'approve'},E,C))
 def test_evidence_gap_contradiction(self):
  self.assertTrue(server.reasoning_errors(R,{**E,'complete':False},C))
 def test_valid_recommendation(self):
  self.assertEqual(server.reasoning_errors(R,E,C),[])
 def test_harness_is_bounded(self):
  with patch('server.llm',return_value={}) as model:
   result=server.reason({'profile':P,'evidence':E,'context':C})
   self.assertEqual(model.call_count,2);self.assertEqual(result['recommendation'],'refer');self.assertTrue(result['validation_errors'])
 def test_harness_can_repair(self):
  with patch('server.llm',side_effect=[{**R,'citations':['FAKE']},R]):
   self.assertEqual(server.reason({'profile':P,'evidence':E,'context':C})['iterations'],2)
 def test_ml_not_configured_disables_stp(self):
  with patch.dict(os.environ,{'UW_ML_URL':''}):self.assertFalse(server.ml_screen({'profile':P})['available'])
 def test_ml_cannot_certify_unknown_uploads(self):
  d={'standard':True,'calibrated':True,'ood':False,'evidence_complete':True,'confidence':.99,'model':'v1','verified_document_hashes':[]}
  doc={'name':'t.pdf','type':'application/pdf','data':base64.b64encode(b'%PDF-1.4 test').decode()}
  with patch.dict(os.environ,{'UW_ML_URL':'http://example.test/predict'}),patch('server.remote_json',return_value=d):
   self.assertFalse(server.ml_screen({'profile':P,'documents':[doc]})['evidence_complete'])
 def test_complex_path_never_autoapproves(self):
  out=server.verify({'profile':P,'evidence':E,'context':C,'reason':R})
  self.assertTrue(out['passed']);self.assertEqual(out['decision'],'refer')
 def test_duplicate_context_ids_rejected(self):
  source={'id':'S1','type':'internal','title':'Rule','excerpt':'Rule text'}
  with patch.dict(os.environ,{'UW_CONTEXT_URL':'http://example.test/search'}),patch('server.remote_json',return_value={'sources':[source,source]}):
   with self.assertRaises(server.DemoError):server.context_retrieval({'profile':P})
if __name__=='__main__':unittest.main()

