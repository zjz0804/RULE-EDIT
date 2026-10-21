"""Data integrity and mathematical consistency checks for RULE-EDIT."""
import copy
import importlib.util
import math
import random
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_data",ROOT/"data"/"prepare_data.py")
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)
RAW = ROOT/"data"/"raw"


class DataPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = p.read_json(RAW/"rules.json")
        cls.by_id = {r["rule_id"]:r for r in cls.rules}
        cls.hierarchy = p.read_json(RAW/"hierarchy.json")
        cls.outputs,cls.manifest = p.build(RAW)

    def test_statistics_and_rule_ids(self):
        self.assertEqual((len(self.rules),self.manifest["rules_2d"],self.manifest["rules_3d"]),(79,44,35))
        self.assertEqual(len(self.manifest["subjects"]),15)
        self.assertEqual(self.manifest["hierarchical_questions"],22)
        self.assertEqual([r["case_id"] for r in self.rules],list(range(79)))
        self.assertEqual(len(self.by_id),79)
        self.assertEqual(self.rules[12]["rule_id"],"right_triangle_area")

    def test_checked_in_release_rebuilds_exactly(self):
        for name,rows in dict(self.outputs,**{"manifest.json":self.manifest}).items():
            self.assertEqual(p.json_bytes(rows),(ROOT/"data"/"processed"/name).read_bytes(),name)

    def test_all_632_assignments_are_valid_disjoint_and_nontrivial(self):
        for rule,entry in zip(self.rules,self.manifest["sampling"]):
            seen = set()
            for sample in entry["samples"]:
                v = sample["values"]
                self.assertTrue(p.geometry_valid(rule,v,sample["witness"]))
                self.assertNotIn(p.sample_key(rule,v),seen)
                seen.add(p.sample_key(rule,v))
                expanded = p.enrich_values(rule,v)
                gold = p.evaluate_expression(rule["original_expression"],expanded)
                target = p.evaluate_expression(rule["edited_expression"],expanded)
                self.assertFalse(math.isclose(gold,target,rel_tol=1e-10,abs_tol=1e-10))
            self.assertEqual(len(seen),8)

    def test_pi_and_fraction_arithmetic(self):
        self.assertEqual(p.exact_text(p.exact_value("2π * r",{"r":10})),"20π")
        self.assertEqual(p.exact_value("(1/3) * π * r^2 * h",{"r":1,"h":1}),(Fraction(1,3),1))
        self.assertEqual(p.evaluate_expression("sqrt(x^2 + y^2)",{"x":3,"y":4}),5)
        self.assertEqual(p.exact_text(p.exact_value("r / 2",{"r":7})),"3.5")
        with self.assertRaises(ValueError): p.exact_value("π + 1",{})

    def test_symmetric_triangle_and_cuboid_copies_share_identity(self):
        rule = self.by_id['triangle_circumradius']
        self.assertEqual(p.sample_key(rule,{"d":3,"b":4,"c":5,"S":6}),
                         p.sample_key(rule,{"d":5,"b":3,"c":4,"S":6}))
        rule = self.by_id['cuboid_surface_area']
        self.assertEqual(p.sample_key(rule,{"l":2,"w":3,"h":4}),
                         p.sample_key(rule,{"l":4,"w":2,"h":3}))

    def test_code_execution_and_excessive_exponents_are_rejected(self):
        for expression in ("__import__('os').getcwd()","(1).__class__","[x for x in (1,)]",
                           "2**100000","True + 1","sqrt(x=2)"):
            with self.subTest(expression=expression),self.assertRaises(ValueError):
                p.evaluate_expression(expression,{})

    def test_isosceles_trapezoid_actual_diagonal(self):
        rule,v = self.by_id['isosceles_trapezoid_diagonal'],{"x":10,"b":4,"c":5}
        # Height is 4; the diagonal's horizontal projection is 7.
        self.assertAlmostEqual(p.evaluate_expression(rule["original_expression"],v),math.sqrt(65))
        self.assertAlmostEqual(p.reference_value(rule,v),math.sqrt(65))

    def test_bad_triangles_and_inconsistent_area_are_rejected(self):
        self.assertFalse(p.geometry_valid(self.by_id['triangle_perimeter'],{"x":1,"y":2,"z":9},{}))
        self.assertFalse(p.geometry_valid(self.by_id['isosceles_triangle_height'],{"x":2,"y":4},{}))
        self.assertTrue(p.geometry_valid(self.by_id['triangle_circumradius'],{"d":3,"b":4,"c":5,"S":6},{}))
        self.assertFalse(p.geometry_valid(self.by_id['triangle_circumradius'],{"d":3,"b":4,"c":5,"S":7},{}))
        witness = {"triangle_sides":[3,4,5]}
        self.assertTrue(p.geometry_valid(self.by_id['triangle_inradius'],{"B":6,"s":6},witness))
        self.assertFalse(p.geometry_valid(self.by_id['triangle_inradius'],{"B":7,"s":6},witness))

    def test_trapezoids_and_angle_domains(self):
        self.assertTrue(p.geometry_valid(self.by_id['trapezoid_perimeter'],{"e":10,"b":4,"c":5,"d":5},{}))
        self.assertFalse(p.geometry_valid(self.by_id['trapezoid_perimeter'],{"e":10,"b":4,"c":1,"d":1},{}))
        self.assertFalse(p.geometry_valid(self.by_id['isosceles_trapezoid_diagonal'],{"x":10,"b":4,"c":2},{}))
        self.assertFalse(p.geometry_valid(self.by_id['parallelogram_area_angle'],{"c":2,"b":3,"θ":4},{}))
        self.assertTrue(p.geometry_valid(self.by_id['sector_arc_length'],{"r":2,"θ":6},{}))

    def test_pyramid_and_prism_geometry_constraints(self):
        witness = {"base_side":2,"slant_height":5}
        self.assertTrue(p.geometry_valid(self.by_id['pyramid_surface_area'],{"B":4,"L":20},witness))
        self.assertFalse(p.geometry_valid(self.by_id['pyramid_surface_area'],{"B":4,"L":19},witness))
        self.assertFalse(p.geometry_valid(self.by_id['pyramid_lateral_area'],{"P":16,"l":1},{}))
        witness = {"base_sides":[2,3]}
        self.assertTrue(p.geometry_valid(self.by_id['prism_surface_area'],{"B":6,"p":10,"h":4},witness))
        self.assertFalse(p.geometry_valid(self.by_id['prism_surface_area'],{"B":6,"p":9,"h":4},witness))

    def test_verbal_targets_express_formula_operations(self):
        diamond = p.rule_views(self.by_id['rhombus_side_diagonals'])["Description"]["target_new"]
        self.assertIn("sum",diamond)
        self.assertNotIn("square",diamond)
        sphere = p.rule_views(self.by_id['sphere_diameter'])["Description"]["target_new"]
        self.assertEqual(sphere,"(the radius) divided by (two).")
        contained = p.rule_views(self.by_id['cuboid_maximum_sphere_radius'])["Description"]["target_new"]
        self.assertNotIn("half",contained)
        self.assertIn("sum",contained)

    def test_templates_preserve_words_and_bind_all_inputs(self):
        rendered = p.render_instance(self.by_id['triangle_area_heron'],{"values":{"x":3,"y":4,"z":5},"style":"step"})
        self.assertIn("Heron's",rendered["prompt"])
        self.assertIn("semiperimeter 6",rendered["prompt"])
        rendered = p.render_instance(self.by_id['triangle_circumradius'],{"values":{"d":3,"b":4,"c":5,"S":6},"style":"step"})
        self.assertIn("circumradius",rendered["prompt"])
        self.assertIn("area 6",rendered["rephrase_prompt"])
        self.assertNotIn("sides a",rendered["rephrase_prompt"])

    def test_icl_paraphrase_contains_demonstration(self):
        item = self.outputs["instance_editing.json"][0]
        demo = item["prompt"][0]+" "+item["target_new"][0]+" "
        icl = self.outputs["instance_editing_icl.json"][0]
        self.assertTrue(icl["prompt"].startswith(demo))
        self.assertTrue(icl["rephrase_prompt"].startswith(demo))

    def test_hierarchy_rectangle_and_lateral_faces(self):
        row = self.outputs["hierarchical.json"][1]
        self.assertEqual(row["ground_truth"][0],"A = x * y.")
        self.assertTrue(row["hierarchical"]["target_new"][-1].endswith("= 56."))
        self.assertTrue(row["hierarchical"]["target_new"][3].endswith("= 88."))

    def test_missing_hierarchical_face_is_rejected(self):
        broken = copy.deepcopy(self.hierarchy)
        broken[1]["downstream"][-1]["dependencies"][0]["factor"] = 1
        with self.assertRaises(ValueError): p.build_hierarchy(broken,self.rules)

    def test_seed_reproducibility_and_global_rng_is_unchanged(self):
        state = random.getstate()
        other,manifest = p.build(RAW,43)
        again,manifest_again = p.build(RAW,43)
        self.assertEqual(other,again)
        self.assertEqual(manifest,manifest_again)
        self.assertNotEqual(other["instance_editing.json"],self.outputs["instance_editing.json"])
        self.assertEqual(state,random.getstate())

    def check_rejected(self,mutate):
        outputs = copy.deepcopy(self.outputs)
        mutate(outputs)
        with self.assertRaises(ValueError):
            p.validate(outputs,self.rules,self.manifest["sampling"],self.hierarchy)

    def test_corrupt_numerical_label_is_rejected(self):
        self.check_rejected(lambda o:o["formula_editing.json"][0]["Instance"]["target_new"].__setitem__(0,"wrong"))

    def test_corrupt_cross_view_target_is_rejected(self):
        self.check_rejected(lambda o:o["instance_editing.json"][0]["Rule_Understanding"]["target_new"].__setitem__(0,"wrong"))

    def test_self_locality_is_rejected(self):
        def mutate(o):
            row = o["formula_editing.json"][0]
            row["locality"]["neighborhood"]["prompt"][0] = row["prompt"]
            row["locality"]["neighborhood"]["ground_truth"][0] = row["ground_truth"][0]
        self.check_rejected(mutate)

    def test_bad_context_and_original_annotation_are_rejected(self):
        bad = copy.deepcopy(self.rules)
        bad[1]["context"] = "A rectangle"
        with self.assertRaises(ValueError): p.validate_catalog(bad)
        bad = copy.deepcopy(self.by_id['isosceles_trapezoid_diagonal'])
        bad["original_expression"] = "c + x + b"
        with self.assertRaises(ValueError): p.sample_instances(bad,random.Random(42))


if __name__ == "__main__":
    unittest.main()
