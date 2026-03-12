"""Tests for api_doc_parser — parse HTML API docs from Ascend pytorch repo."""

from __future__ import annotations

from diffusion_agent.tools.api_doc_parser import OpEntry, build_op_matrix, parse_api_doc


# ---------------------------------------------------------------------------
# Sample data — inline HTML mimicking the real format
# ---------------------------------------------------------------------------

SAMPLE_DOC = '''# API of PyTorch Ascend Adapter

## torch

<table><thead><tr><th>PyTorch API</th><th>Compatibility</th><th>Limitations</th></tr></thead>
<tbody>
<tr id="row1"><td><p><a name="a"></a><a name="b"></a>torch.zeros</p></td>
<td><p>Y</p></td>
<td><p>Only support fp16, fp32</p></td></tr>
<tr id="row2"><td><p><a name="c"></a>torch.default_generator</p></td>
<td>&nbsp;</td>
<td>&nbsp;</td></tr>
</tbody></table>

## torch.nn

<table><thead><tr><th>PyTorch API</th><th>Compatibility</th><th>Limitations</th></tr></thead>
<tbody>
<tr id="row3"><td><p><a name="d"></a><a name="e"></a>torch.nn.Conv2d</p></td>
<td><p>Y</p></td>
<td>&nbsp;&nbsp;</td></tr>
<tr id="row4"><td><p><a name="f"></a>torch.nn.Linear</p></td>
<td><p>Y</p></td>
<td><p>Only support fp16\uff0cfp32\uff0cint64\uff0cbool</p></td></tr>
</tbody></table>
'''

# Real escaped format as it appears in the actual doc file
REAL_FORMAT_DOC = (
    '## torch\\n\\n<a name="table..."></a>\\n'
    '<table><thead align="left"><tr id="row0">'
    '<th class="cellrowborder" valign="top" width="59.61%" id="mcps1.1.4.1.1">'
    '<p id="p0"><a name="p0"></a><a name="p0"></a>PyTorch API</p>\\n</th>\\n'
    '<th class="cellrowborder" valign="top" width="5.74%" id="mcps1.1.4.1.2">'
    '<p id="p1"><a name="p1"></a><a name="p1"></a>Compatibility</p>\\n</th>\\n'
    '<th class="cellrowborder" valign="top" width="34.65%" id="mcps1.1.4.1.3">'
    '<p id="p2"><a name="p2"></a><a name="p2"></a>Limitations</p>\\n</th>\\n'
    '</tr>\\n</thead>\\n'
    '<tbody>'
    '<tr id="row1"><td class="cellrowborder" valign="top" width="59.61%" '
    'headers="mcps1.1.4.1.1 "><p id="p3"><a name="p3"></a><a name="p3"></a>'
    'torch.SymInt</p>\\n</td>\\n'
    '<td class="cellrowborder" valign="top" width="5.74%" '
    'headers="mcps1.1.4.1.2 "><p id="p4"><a name="p4"></a><a name="p4"></a>'
    'Y</p>\\n</td>\\n'
    '<td class="cellrowborder" valign="top" width="34.65%" '
    'headers="mcps1.1.4.1.3 ">&nbsp;&nbsp;</td>\\n</tr>\\n'
    '<tr id="row2"><td class="cellrowborder" valign="top" width="59.61%" '
    'headers="mcps1.1.4.1.1 "><p id="p5"><a name="p5"></a><a name="p5"></a>'
    'torch.default_generator</p>\\n</td>\\n'
    '<td class="cellrowborder" valign="top" width="5.74%" '
    'headers="mcps1.1.4.1.2 ">&nbsp;&nbsp;</td>\\n'
    '<td class="cellrowborder" valign="top" width="34.65%" '
    'headers="mcps1.1.4.1.3 ">&nbsp;&nbsp;</td>\\n</tr>\\n'
    '</tbody></table>'
)


# ---------------------------------------------------------------------------
# parse_api_doc — individual op parsing
# ---------------------------------------------------------------------------


class TestParseSupported:
    def test_parse_supported_op(self):
        entries = parse_api_doc(SAMPLE_DOC)
        zeros = [e for e in entries if e.api_name == "torch.zeros"]
        assert len(zeros) == 1
        assert zeros[0].compatible is True

    def test_parse_untested_op(self):
        entries = parse_api_doc(SAMPLE_DOC)
        gen = [e for e in entries if e.api_name == "torch.default_generator"]
        assert len(gen) == 1
        assert gen[0].compatible is False

    def test_parse_limitations(self):
        entries = parse_api_doc(SAMPLE_DOC)
        zeros = [e for e in entries if e.api_name == "torch.zeros"]
        assert zeros[0].limitations == "Only support fp16, fp32"

    def test_parse_no_limitations(self):
        entries = parse_api_doc(SAMPLE_DOC)
        conv = [e for e in entries if e.api_name == "torch.nn.Conv2d"]
        assert conv[0].limitations == ""


class TestParseSections:
    def test_parse_sections(self):
        entries = parse_api_doc(SAMPLE_DOC)
        torch_entries = [e for e in entries if e.section == "torch"]
        nn_entries = [e for e in entries if e.section == "torch.nn"]
        assert len(torch_entries) == 2
        assert len(nn_entries) == 2

    def test_section_names_correct(self):
        entries = parse_api_doc(SAMPLE_DOC)
        zeros = [e for e in entries if e.api_name == "torch.zeros"][0]
        assert zeros.section == "torch"
        linear = [e for e in entries if e.api_name == "torch.nn.Linear"][0]
        assert linear.section == "torch.nn"


class TestParseEdgeCases:
    def test_parse_empty_content(self):
        entries = parse_api_doc("")
        assert entries == []

    def test_parse_no_tables(self):
        entries = parse_api_doc("## torch\n\nSome text without any tables.")
        assert entries == []


class TestParseRealFormat:
    def test_parse_real_format(self):
        """Test with the actual escaped format from the Ascend pytorch repo."""
        entries = parse_api_doc(REAL_FORMAT_DOC)
        assert len(entries) == 2

        sym = [e for e in entries if e.api_name == "torch.SymInt"]
        assert len(sym) == 1
        assert sym[0].compatible is True
        assert sym[0].limitations == ""
        assert sym[0].section == "torch"

        gen = [e for e in entries if e.api_name == "torch.default_generator"]
        assert len(gen) == 1
        assert gen[0].compatible is False


class TestOpEntry:
    def test_fields(self):
        e = OpEntry(api_name="torch.zeros", compatible=True, limitations="fp16 only", section="torch")
        assert e.api_name == "torch.zeros"
        assert e.compatible is True
        assert e.limitations == "fp16 only"
        assert e.section == "torch"


# ---------------------------------------------------------------------------
# build_op_matrix — convert entries to checker-compatible format
# ---------------------------------------------------------------------------


class TestBuildOpMatrix:
    def test_build_op_matrix_supported(self):
        entries = [OpEntry("torch.zeros", True, "", "torch")]
        matrix = build_op_matrix(entries)
        assert matrix["ops"]["torch.zeros"]["status"] == "supported"

    def test_build_op_matrix_untested(self):
        entries = [OpEntry("torch.default_generator", False, "", "torch")]
        matrix = build_op_matrix(entries)
        assert matrix["ops"]["torch.default_generator"]["status"] == "unknown"
        assert matrix["ops"]["torch.default_generator"]["note"] == "Not tested on NPU"

    def test_build_op_matrix_with_limitations(self):
        entries = [OpEntry("torch.ones", True, "Only support fp16, fp32, int64", "torch")]
        matrix = build_op_matrix(entries)
        assert matrix["ops"]["torch.ones"]["note"] == "Only support fp16, fp32, int64"

    def test_build_op_matrix_empty_patterns(self):
        entries = [OpEntry("torch.zeros", True, "", "torch")]
        matrix = build_op_matrix(entries)
        assert matrix["patterns"] == {}

    def test_build_op_matrix_has_ops_and_patterns(self):
        entries = [OpEntry("torch.zeros", True, "", "torch")]
        matrix = build_op_matrix(entries)
        assert "ops" in matrix
        assert "patterns" in matrix

    def test_build_op_matrix_empty_entries(self):
        matrix = build_op_matrix([])
        assert matrix["ops"] == {}
        assert matrix["patterns"] == {}
