import unittest
from unittest.mock import patch
import governance as g
import server
from test_server import P,E,C,R
RULE={'id':'USR-test','title':'Review age','revision':1,'field':'age','operator':'gte','value':35,'scope':'Both','action':'refer','enabled':True}
class PolicyTests(unittest.TestCase):
 def test_rule_match(self):
  self.assertEqual(g.policy_outcome(g.evaluate_rules(P,[RULE])),'refer')
 def test_unknown_optional_blocks(self):
  r={**RULE,'field':'egfr','operator':'lt','value':50}
  self.assertEqual(g.policy_outcome(g.evaluate_rules(P,[r])),'request_evidence')
 def test_draft_is_inactive(self):
  self.assertEqual(g.policy_outcome(g.evaluate_rules(P,[{**RULE,'enabled':False}])),'pass')
 def test_approval_and_untyped_rules_rejected(self):
  for change in [{'action':'approve'},{'value':'35'},{'field':'sex'},{'operator':'eval'},{'value':True}]:
   with self.assertRaises(g.PolicyError):g.validate_rule({**RULE,**change})
 def test_duplicate_rejected(self):
  with self.assertRaises(g.PolicyError):g.evaluate_rules(P,[RULE,RULE])
 def test_extra_attributes_preserved(self):
  self.assertEqual(server.validate_profile({**P,'hba1c':6.2})['hba1c'],6.2)
  self.assertIsNone(server.validate_profile(P)['hba1c'])
 def test_instructions_pass_to_model_below_safety_controls(self):
  with patch('server.llm',return_value=R) as model:
   server.reason({'profile':P,'evidence':E,'context':C,'policy':{'rules':[],'instructions':'Check chronology'}})
   messages=model.call_args.args[0]
   self.assertIn('cannot override',messages[0]['content'])
   self.assertIn('Check chronology',messages[1]['content'])
 def test_custom_missing_data_enforced_by_verifier(self):
  r={**RULE,'field':'egfr','operator':'lt','value':50}
  out=server.verify({'profile':P,'evidence':E,'context':C,'reason':R,'policy':{'rules':[r],'instructions':''}})
  self.assertFalse(out['passed']);self.assertEqual(out['decision'],'request_evidence')
 def test_custom_request_repaired_or_referred_after_two_attempts(self):
  r={**RULE,'action':'request_evidence'}
  with patch('server.llm',return_value=R) as model:
   result=server.reason({'profile':P,'evidence':E,'context':C,'policy':{'rules':[r],'instructions':''}})
   self.assertEqual(model.call_count,2);self.assertTrue(result['validation_errors'])

