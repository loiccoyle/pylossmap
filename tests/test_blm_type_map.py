import unittest

from pylossmap.blm_type_map import name_to_type


class TestTypeMap(unittest.TestCase):
    def test_name_to_type_map_cold(self):
        inp = [
            "BLMQI.16R8.B1I10_MQ",
            "BLMQI.16R8.B2E10_MQ",
            "BLMQI.16R8.B1I30_MQ",
            "BLMBI.17R8.B0T10_MBB-MBA",
            "BLMBI.17R8.B0T20_MBA-MBB",
            "BLMQI.17R8.B2E30_MQ",
            "BLMQI.17R8.B1I10_MQ",
            "BLMQI.17R8.B2E10_MQ",
            "BLMQI.17R8.B1I30_MQ",
            "BLMBI.18R8.B0T10_MBA-MBB",
            "BLMBI.18R8.B0T20_MBB-MBA",
            "BLMQI.18R8.B2E30_MQ",
            "BLMQI.18R8.B1I10_MQ",
            "BLMQI.18R8.B2E10_MQ",
            "BLMQI.18R8.B1I30_MQ",
            "BLMBI.19R8.B0T10_MBB-MBA",
            "BLMBI.19R8.B0T20_MBA-MBB",
            "BLMQI.19R8.B2E30_MQ",
            "BLMQI.19R8.B1I10_MQ",
            "BLMQI.19R8.B2E10_MQ",
            "BLMQI.19R8.B1I30_MQ",
            "BLMBI.20R8.B0T10_MBA-MBB",
            "BLMBI.20R8.B0T20_MBB-MBA",
            "BLMQI.20R8.B2E30_MQ",
            "BLMQI.20R8.B1I10_MQ",
            "BLMQI.20R8.B2E10_MQ",
        ]
        for b in inp:
            self.assertEqual(name_to_type(b), "cold")

    def test_name_to_type_map_coll(self):
        inp = [
            "BLMTI.04R1.B2I10_TCTPV.4R1.B2",
            "BLMTI.04R1.B2I10_TCTPH.4R1.B2",
            "BLMTI.04R1.B1E10_TCL.4R1.B1",
            "BLMTI.05R1.B1E10_TCL.5R1.B1",
            "BLMTI.06R1.B1E10_TCL.6R1.B1",
            "BLMTL.04L2.B1E10_TCTPH.4L2.B1",
            "BLMTI.04L2.B1E10_TCTPH.4L2.B1",
            "BLMTI.04L2.B1E10_TCTPV.4L2.B1",
            "BLMTL.04L2.B1E10_TCTPV.4L2.B1",
            "BLMTI.04L2.B2I10_TDI.4L2.B2",
            "BLMTL.04L2.B2I10_TDI.4L2.B2",
            "BLMTI.04L2.B1E10_TDI.4L2.B1",
            "BLMTL.04L2.B1E10_TDI.4L2.B1",
            "BLMTI.04L2.B1E20_TDI.4L2.B1",
            "BLMTL.04L2.B1E20_TDI.4L2.B1",
            "BLMTI.04L2.B1E10_TCDD.4L2",
            "BLMTL.04L2.B1E10_TCDD.4L2",
            "BLMTI.04R2.B1I10_TCLIA.4R2",
            "BLMTL.04R2.B1I10_TCLIA.4R2",
            "BLMTI.04R2.B2E10_TCTPV.4R2.B2",
            "BLMTL.04R2.B2E10_TCTPV.4R2.B2",
            "BLMTL.04R2.B2E10_TCTPH.4R2.B2",
            "BLMTI.04R2.B2E10_TCTPH.4R2.B2 ",
        ]
        for b in inp:
            self.assertEqual(name_to_type(b), "coll")

    def test_name_to_type_map_warm(self):
        inp = [
            "BLMEL.01R1.B2I10_BPMSW.1R1",
            "BLMTI.04R1.B1E10_TANAR.4R1",
            "BLMEL.06L2.B1E10_MSIB",
            "BLMEI.06L2.B1E10_MSIB",
            "BLMEL.06L2.B1E20_MSIB",
            "BLMEI.06L2.B1E20_MSIB",
            "BLMEL.06L2.B1E30_MSIB",
            "BLMEI.06L2.B1E30_MSIB",
            "BLMEL.06L2.B1E10_MSIA",
            "BLMEI.06L2.B1E10_MSIA",
            "BLMEL.06L2.B1E20_MSIA",
            "BLMEI.06L2.B1E20_MSIA",
            "BLMEL.06L2.B1E30_MSIA",
            "BLMEI.06L2.B1E30_MSIA",
            "BLMEI.05L2.B1E10_MKI.D5L2.B1",
            "BLMEI.05L2.B1E20_MKI.C5L2.B1",
        ]
        for b in inp:
            self.assertEqual(name_to_type(b), "warm")

    def test_name_to_type_map_xrp(self):
        inp = [
            "BLMQI.07R1.B2I30_MQM_XRP",
            "BLMQI.06L5.B2E22_MQML_XRP",
            "BLMQI.06L5.B1I20_MQML_XRP",
            "BLMQI.06L5.B2E10_MQML_XRP",
            "BLMQI.06L5.B1I30_MQML_XRP",
            "BLMQI.06R5.B2I30_MQML_XRP",
            "BLMQI.06R5.B2I20_MQML_XRP",
            "BLMQI.06R5.B1E20_MQML_XRP",
            "BLMQI.07L1.B2E10_MQM_XRP",
            "BLMQI.07L1.B1I30_MQM_XRP",
            "BLMEI.07R1.B1E10_XRP",
            "BLMEI.07R1.B1E20_XRP-RUN1",
            "BLMQI.07R1.B1E10_MQM_XRP",
        ]
        for b in inp:
            self.assertEqual(name_to_type(b), "xrp")

    def test_name_to_type_map_other(self):
        self.assertEqual(name_to_type("asdadad", on_fail="foo"), "foo")
