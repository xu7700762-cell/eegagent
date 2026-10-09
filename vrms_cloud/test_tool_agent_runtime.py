import io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

from .tool_agent_v1 import call_agent,validate_final
from .evaluate_tool_agent import normalize_tool_citations


class Runtime:
    query_index=0
    def __init__(self):self.calls=[];self.timings={}
    def initial_evidence(self):return dict(id="qsingle",baseline=dict(raw_class="High",raw_high_probability=.6))
    def execute(self,name):
        if name not in self.calls:self.calls.append(name);self.timings[name]=1.
        return dict(tool=name,raw_learned_class="Low")


def final():
    return dict(id="qsingle",final_class="Low",confidence="low",tools_used=["femba_features"],
                supporting_evidence=[],conflicting_evidence=[],missing_evidence=[],explanation="Based on requested evidence")


class AgentRuntimeContracts(unittest.TestCase):
    def test_real_function_roundtrip_shape_and_lazy_execution(self):
        runtime=Runtime();seen=[]
        def server(request,timeout):
            body=json.loads(request.data);seen.append(body)
            if len(seen)==1:
                self.assertEqual(runtime.calls,[])
                output=[dict(type="function_call",name="femba_features",arguments=json.dumps(dict(query="qsingle")),call_id="call1")]
            else:
                self.assertEqual(runtime.calls,["femba_features"])
                self.assertEqual(body["input"][-1]["type"],"function_call_output")
                self.assertEqual(body["input"][-1]["call_id"],"call1")
                output=[dict(type="message",content=[dict(type="output_text",text=json.dumps(final()))])]
            return io.StringIO(json.dumps(dict(status="completed",model="test",output=output)))
        with tempfile.TemporaryDirectory() as folder,patch("urllib.request.urlopen",side_effect=server):
            destination=Path(folder)/"call.json"
            log=call_agent(dict(model="test",base_url="http://localhost/v1",key="PRIVATE_KEY"),runtime,destination)
            self.assertTrue(log["success"]);self.assertEqual(log["final"]["final_class"],"Low")
            self.assertEqual(log["actual_tool_calls"],["femba_features"])
            self.assertNotIn("PRIVATE_KEY",destination.read_text(encoding="utf-8"))

    def test_citation_fix_cannot_change_decision_or_invent_tool(self):
        value=dict(final(),tools_used=["spectral / spatial_covariance"])
        result=normalize_tool_citations(value,["spectral","spatial_covariance"])
        self.assertEqual(result["final_class"],value["final_class"])
        self.assertEqual(result["explanation"],value["explanation"])
        with self.assertRaises(ValueError):normalize_tool_citations(value,["spectral"])

    def test_uncertain_and_unexecuted_tool_are_rejected(self):
        with self.assertRaises(ValueError):validate_final(dict(final(),final_class="uncertain"),["femba_features"])
        with self.assertRaises(ValueError):validate_final(final(),[])


if __name__=="__main__":unittest.main()
