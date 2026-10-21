"""Build and validate RULE-EDIT datasets (standard library only)."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import random
import re
import string
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERSION = "1.0.0"
FILES = ("formula_editing.json", "description_editing.json", "instance_editing.json",
         "instance_editing_icl.json", "hierarchical.json")
PLANE_SHAPES = {"rectangle", "square", "triangle", "parallelogram", "diamond", "trapezoid", "circle", "sector"}
FUNCTIONS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "min": min}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def parse_expression(expression):
    expression = re.sub(r"(?<=\d)\s*π", " * π", expression)
    tree = ast.parse(expression.replace("^", "**"), mode="eval")
    require(sum(1 for _ in ast.walk(tree)) <= 200, "Expression is too large")
    for node in ast.walk(tree):
        require(isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name,
                                  ast.Load, ast.Call, ast.Add, ast.Sub, ast.Mult, ast.Div,
                                  ast.Pow, ast.USub, ast.UAdd)), "Unsupported expression syntax")
        if isinstance(node, ast.Call):
            require(isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS and not node.keywords,
                    "Only sqrt, sin, cos and min calls are supported")
        if isinstance(node, ast.Constant):
            require(type(node.value) in (int, float), "Only numeric constants are supported")
    return tree


def evaluate_expression(expression, values):
    """Restricted arithmetic evaluator; dataset strings are never executed."""
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return math.pi if node.id == "π" else float(values[node.id])
        if isinstance(node, ast.Call):
            return FUNCTIONS[node.func.id](*(visit(a) for a in node.args))
        if isinstance(node, ast.UnaryOp):
            return visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        left, right = visit(node.left), visit(node.right)
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
        if isinstance(node.op, ast.Mult): return left * right
        if isinstance(node.op, ast.Div): return left / right
        require(abs(right) <= 12, "Exponent exceeds the supported range")
        return left ** right

    value = visit(parse_expression(expression))
    require(not isinstance(value, complex) and math.isfinite(value), "Non-finite arithmetic result")
    return value


def exact_value(expression, values):
    """Return a rational coefficient and integer pi power, without floating pi."""
    def visit(node):
        if isinstance(node, ast.Expression): return visit(node.body)
        if isinstance(node, ast.Constant): return Fraction(str(node.value)), 0
        if isinstance(node, ast.Name):
            return (Fraction(1), 1) if node.id == "π" else (Fraction(str(values[node.id])), 0)
        if isinstance(node, ast.UnaryOp):
            value, degree = visit(node.operand)
            return value * (-1 if isinstance(node.op, ast.USub) else 1), degree
        require(isinstance(node, ast.BinOp), "Exact output supports rational arithmetic and powers of pi")
        a, p = visit(node.left)
        b, q = visit(node.right)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            require(p == q, "Cannot combine different powers of pi in one exact coefficient")
            return a + b * (-1 if isinstance(node.op, ast.Sub) else 1), p
        if isinstance(node.op, ast.Mult): return a*b, p+q
        if isinstance(node.op, ast.Div): return a/b, p-q
        require(q == 0 and b.denominator == 1 and abs(b) <= 12, "Invalid exact exponent")
        return a ** int(b), p * int(b)

    return visit(parse_expression(expression))


def rational_text(value):
    value = Fraction(value)
    if value.denominator == 1: return str(value.numerator)
    denominator = value.denominator
    for factor in (2, 5):
        while denominator % factor == 0: denominator //= factor
    if denominator == 1: return format(float(value), ".12g")
    return f"({value.numerator}/{value.denominator})"


def exact_text(value):
    coefficient, power = value
    if not power: return rational_text(coefficient)
    prefix = "" if coefficient == 1 else "-" if coefficient == -1 else rational_text(coefficient)
    return prefix + ("π" if power == 1 else f"π^{power}")


def substitute_expression(expression, values):
    return re.sub(r"\b([a-zA-Zθ])\b", lambda m: rational_text(Fraction(str(values[m[1]])))
                  if m[1] in values else m[1], expression)


def describe(expression, labels):
    """Derive an unambiguous verbal calculation from the same expression AST."""
    numbers = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 6: "six", 16: "sixteen"}

    def visit(node):
        if isinstance(node, ast.Expression): return visit(node.body)
        if isinstance(node, ast.Constant): return numbers.get(node.value, str(node.value))
        if isinstance(node, ast.Name): return "π" if node.id == "π" else "the " + labels[node.id]
        if isinstance(node, ast.Call):
            args = [visit(a) for a in node.args]
            if node.func.id == "min": return "the minimum of (" + "), (".join(args) + ")"
            name = {"sqrt": "square root", "sin": "sine", "cos": "cosine"}[node.func.id]
            return f"the {name} of ({args[0]})"
        if isinstance(node, ast.UnaryOp):
            return ("the negative of " if isinstance(node.op, ast.USub) else "") + f"({visit(node.operand)})"
        a, b = visit(node.left), visit(node.right)
        if isinstance(node.op, ast.Pow):
            if isinstance(node.right, ast.Constant) and node.right.value in (2, 3):
                return f"the {'square' if node.right.value == 2 else 'cube'} of ({a})"
            return f"({a}) raised to the power ({b})"
        if isinstance(node.op, ast.Div): return f"({a}) divided by ({b})"
        operation = {ast.Add: "sum", ast.Sub: "difference", ast.Mult: "product"}[type(node.op)]
        return f"the {operation} of ({a}) and ({b})"

    return visit(parse_expression(expression)) + "."


def enrich_values(rule, values):
    result = dict(values)
    for name, definition in rule.get("derived", {}).items():
        number = float(evaluate_expression(definition["expression"], result))
        result[name] = int(number) if number.is_integer() else number
    return result


def context_text(rule, values=None):
    if values is None:
        mapping = {name: name for name in rule["variables"]}
        mapping.update({name: name for name in rule.get("derived", {})})
    else:
        mapping = {name: rational_text(Fraction(str(value))) for name, value in enrich_values(rule, values).items()}
    return rule["context"].format_map(mapping)


def rule_views(rule):
    context = context_text(rule)
    quantity, symbol = rule["quantity"], rule["output_symbol"]
    original, edited = rule["original_expression"], rule["edited_expression"]
    labels = dict(rule["variables"], **{k: v["label"] for k, v in rule.get("derived", {}).items()})
    return {
        "Formula": {"prompt": f"{context}. Its {quantity} can be calculated by the formula:",
                    "target_new": f"{symbol} = {edited}.",
                    "ground_truth": [f"{symbol} = {original}.", f"{original}."],
                    "rephrase_prompt": f"{context}. Using these measurements, the formula for its {quantity} is"},
        "Description": {"prompt": f"{context}. In words, its {quantity} can be calculated by",
                        "target_new": describe(edited, labels),
                        "ground_truth": [describe(original, labels)],
                        "rephrase_prompt": f"{context}. Describe how to calculate its {quantity} in words:"},
    }


def numerical_answer(rule, values, original=False, direct=False):
    expression = rule["original_expression" if original else "edited_expression"]
    values = enrich_values(rule, values)
    try:
        answer = exact_text(exact_value(expression, values))
        equality = "="
    except ValueError:
        require(original, "Counterfactual target must be exactly representable")
        answer = format(evaluate_expression(expression, values), ".12g")
        equality = "≈"
    if direct: return ("approximately " if equality == "≈" else "") + answer + "."
    return f"{rule['output_symbol']} = {substitute_expression(expression, values)} {equality} {answer}."


def render_instance(rule, sample):
    values, style = sample["values"], sample["style"]
    context = context_text(rule, values)
    quantity = rule["quantity"]
    prompt = (f"{context}. What is its {quantity}? The answer is" if style == "direct" else
              f"{context}. Calculate its {quantity}, showing the substitution and final result:")
    rephrase = f"Using the following measurements, determine its {quantity}: {context}. The result is"
    return {"prompt": prompt, "target_new": numerical_answer(rule, values, direct=style == "direct"),
            "ground_truth": numerical_answer(rule, values, original=True, direct=style == "direct"),
            "rephrase_prompt": rephrase}


def triangle_area(a, b, c):
    require(min(a,b,c) > 0 and 2*max(a,b,c) < a+b+c, "Invalid triangle sides")
    s = (a+b+c)/2
    return math.sqrt(s*(s-a)*(s-b)*(s-c))


def geometry_valid(rule, values, witness):
    if set(values) != set(rule["variables"]) or not all(math.isfinite(v) and v > 0 for v in values.values()):
        return False
    g, v = rule["geometry"], values
    try:
        if g == "triangle_sides": triangle_area(v["x"],v["y"],v["z"])
        elif g == "isosceles_triangle": require(v["y"] < 2*v["x"], "Degenerate isosceles triangle")
        elif g == "triangle_area_semiperimeter":
            a,b,c = witness["triangle_sides"]
            require(math.isclose(v["B"],triangle_area(a,b,c)) and v["s"] == (a+b+c)/2,
                    "Triangle area and semiperimeter disagree")
        elif g == "triangle_sides_area":
            require(math.isclose(v["S"],triangle_area(v["d"],v["b"],v["c"])), "Inconsistent triangle area")
        elif g == "interior_angle": require(v["θ"] < math.pi, "Interior angle must be between zero and pi")
        elif g == "sector": require(v["θ"] < 2*math.pi, "Sector angle must be below two pi")
        elif g == "trapezoid_sides":
            delta = abs(v["e"]-v["b"])
            require(abs(v["c"]-v["d"]) < delta < v["c"]+v["d"], "Impossible trapezoid")
        elif g == "trapezoid_bases": require(v["c"] != v["b"], "Use distinct parallel bases")
        elif g == "isosceles_trapezoid":
            require(0 < abs(v["x"]-v["b"]) < 2*v["c"], "Impossible isosceles trapezoid")
        elif g == "square_pyramid_areas":
            a,slant = witness["base_side"],witness["slant_height"]
            require(slant > a/2 and v["B"] == a*a and v["L"] == 2*a*slant, "Impossible square pyramid")
        elif g == "regular_pyramid_slant": require(v["l"] > v["P"]/8, "Slant height must exceed the square-base apothem")
        elif g == "right_prism_base":
            a,b = witness["base_sides"]
            require(v["B"] == a*b and v["p"] == 2*(a+b), "Inconsistent prism base")
        else: require(g == "positive", f"Unknown geometry constraint: {g}")
        return True
    except (ValueError,KeyError,ArithmeticError):
        return False


def reference_value(rule, v):
    """Geometry oracle independent of the annotation expression evaluator."""
    rule_id = rule["rule_id"]
    x,y,z = (v.get(k,0) for k in ("x","y","z"))
    l,w,h = (v.get(k,0) for k in ("l","w","h"))
    p,q,r,d = (v.get(k,0) for k in ("p","q","r","d"))
    b,c,s = (v.get(k,0) for k in ("b","c","s"))
    B,P,angle = (v.get(k,0) for k in ("B","P","θ"))
    if rule_id in ('rectangle_perimeter', 'parallelogram_perimeter'): return 2*(x+y)
    if rule_id in ('rectangle_area', 'parallelogram_area'): return x*y
    if rule_id in ('rectangle_diagonal', 'right_triangle_hypotenuse'): return math.hypot(x,y)
    if rule_id == 'rectangle_diagonal_squared': return x*x+y*y
    if rule_id == 'rectangle_aspect_ratio': return x/y
    if rule_id == 'square_perimeter': return 4*x
    if rule_id == 'square_area': return x*x
    if rule_id == 'square_diagonal': return math.hypot(x,x)
    if rule_id == 'square_diagonal_squared': return 2*x*x
    if rule_id == 'square_area_from_perimeter': return (P/4)**2
    if rule_id in ('triangle_area', 'right_triangle_area'): return x*y/2
    if rule_id == 'triangle_perimeter': return x+y+z
    if rule_id == 'equilateral_triangle_perimeter': return 3*x
    if rule_id == 'equilateral_triangle_area': return x*math.sqrt(x*x-(x/2)**2)/2
    if rule_id == 'isosceles_triangle_perimeter': return 2*x+y
    if rule_id == 'isosceles_triangle_height': return math.sqrt(x*x-y*y/4)
    if rule_id == 'isosceles_triangle_area': return y*math.sqrt(x*x-y*y/4)/2
    if rule_id == 'triangle_area_heron': return triangle_area(x,y,z)
    if rule_id == 'triangle_inradius': return B/s
    if rule_id == 'triangle_circumradius': return d*b*c/(4*v["S"])
    if rule_id == 'parallelogram_diagonal': return math.hypot(c-b*math.cos(angle),b*math.sin(angle))
    if rule_id == 'parallelogram_area_angle': return c*b*math.sin(angle)
    if rule_id == 'rhombus_area_diagonals': return p*q/2
    if rule_id == 'rhombus_perimeter': return 4*x
    if rule_id == 'rhombus_area_angle': return x*x*math.sin(angle)
    if rule_id == 'rhombus_perimeter_diagonals': return 4*math.hypot(p/2,q/2)
    if rule_id == 'rhombus_side_diagonals': return math.hypot(p/2,q/2)
    if rule_id == 'trapezoid_area': return (c+b)*h/2
    if rule_id == 'trapezoid_perimeter': return v["e"]+b+c+d
    if rule_id == 'trapezoid_midsegment': return (c+b)/2
    if rule_id == 'isosceles_trapezoid_diagonal':
        height = math.sqrt(c*c-((x-b)/2)**2)
        return math.hypot(height,(x+b)/2)
    if rule_id == 'circle_circumference': return 2*math.pi*r
    if rule_id == 'circle_circumference_diameter': return math.pi*d
    if rule_id == 'circle_area': return math.pi*r*r
    if rule_id == 'circle_area_diameter': return math.pi*(d/2)**2
    if rule_id == 'circle_diameter': return 2*r
    if rule_id == 'circle_radius': return d/2
    if rule_id == 'sector_arc_length': return r*angle
    if rule_id == 'sector_area': return r*r*angle/2
    if rule_id == 'sector_perimeter': return 2*r+r*angle
    if rule_id == 'cube_space_diagonal': return math.sqrt(3*l*l)
    if rule_id == 'cube_space_diagonal_squared': return 3*v["m"]**2
    if rule_id == 'cube_inradius': return p/2
    if rule_id == 'cube_circumradius': return math.sqrt(3*q*q)/2
    if rule_id == 'cube_base_area': return s*s
    if rule_id == 'cuboid_surface_area': return 2*(l*w+l*h+w*h)
    if rule_id == 'cuboid_volume': return l*w*h
    if rule_id == 'cuboid_edge_sum': return 4*(l+w+h)
    if rule_id == 'cuboid_lateral_area': return 2*h*(l+w)
    if rule_id == 'cuboid_base_area': return l*w
    if rule_id == 'cuboid_space_diagonal': return math.sqrt(l*l+w*w+h*h)
    if rule_id == 'cuboid_space_diagonal_squared': return l*l+w*w+h*h
    if rule_id == 'cuboid_maximum_sphere_radius': return min(l,w,h)/2
    if rule_id == 'cuboid_circumradius': return math.sqrt(l*l+w*w+h*h)/2
    if rule_id == 'cylinder_surface_area': return 2*math.pi*r*r+2*math.pi*r*h
    if rule_id == 'cylinder_volume': return math.pi*r*r*h
    if rule_id in ('cylinder_base_area', 'cone_base_area', 'sphere_great_circle_area'): return math.pi*r*r
    if rule_id == 'cylinder_lateral_area': return 2*math.pi*r*h
    if rule_id == 'cone_surface_area': return math.pi*r*r+math.pi*r*math.hypot(r,h)
    if rule_id == 'cone_volume': return math.pi*r*r*h/3
    if rule_id == 'cone_lateral_area': return math.pi*r*math.hypot(r,h)
    if rule_id == 'sphere_surface_area': return 4*math.pi*r*r
    if rule_id == 'sphere_volume': return 4*math.pi*r*r*r/3
    if rule_id == 'sphere_diameter': return 2*r
    if rule_id == 'sphere_radius': return d/2
    if rule_id == 'pyramid_volume': return B*h/3
    if rule_id == 'pyramid_surface_area': return B+v["L"]
    if rule_id == 'pyramid_lateral_area': return P*l/2
    if rule_id == 'square_pyramid_volume': return s*s*h/3
    if rule_id == 'prism_volume': return B*h
    if rule_id == 'prism_surface_area': return 2*B+p*h
    if rule_id == 'prism_lateral_area': return p*h
    if rule_id == 'equilateral_triangular_prism_volume': return s*math.sqrt(s*s-s*s/4)*h/2
    raise ValueError(f"No geometry oracle for rule {rule_id}")


def draw_values(rule, rng):
    v = {name:rng.randint(1,3 if name == "θ" and rule["geometry"] == "interior_angle"
                          else 6 if name == "θ" else 15) for name in rule["variables"]}
    witness = {}
    g = rule["geometry"]
    if g in ("triangle_area_semiperimeter","triangle_sides_area"):
        a,b,c = (rng.randint(1,15) for _ in range(3))
        try: area = triangle_area(a,b,c)
        except ValueError: return None,None
        if not math.isclose(area,round(area),abs_tol=1e-12): return None,None
        if g == "triangle_area_semiperimeter":
            v = {"B":round(area),"s":(a+b+c)/2}
            witness = {"triangle_sides":[a,b,c]}
        else: v = {"d":a,"b":b,"c":c,"S":round(area)}
    elif g == "square_pyramid_areas":
        a,slant = rng.randint(1,15),rng.randint(1,15)
        v = {"B":a*a,"L":2*a*slant}
        witness = {"base_side":a,"slant_height":slant}
    elif g == "right_prism_base":
        a,b = rng.randint(1,15),rng.randint(1,15)
        v.update(B=a*b,p=2*(a+b))
        witness = {"base_sides":[a,b]}
    return v,witness


def sample_instances(rule, rng):
    samples,seen = [],set()
    styles = ("direct","direct","step","step","step","step","step","step")
    for _ in range(50000):
        v,witness = draw_values(rule,rng)
        if v is None or not geometry_valid(rule,v,witness): continue
        key = sample_key(rule,v)
        if key in seen: continue
        enriched = enrich_values(rule,v)
        original = evaluate_expression(rule["original_expression"],enriched)
        require(math.isclose(original,reference_value(rule,v),rel_tol=1e-10,abs_tol=1e-10),
                f"Incorrect original formula for rule {rule['rule_id']}")
        edited = evaluate_expression(rule["edited_expression"],enriched)
        if math.isclose(original,edited,rel_tol=1e-10,abs_tol=1e-10): continue
        seen.add(key)
        samples.append({"values":v,"witness":witness,"style":styles[len(samples)]})
        if len(samples) == 8: return samples
    raise ValueError(f"Unable to obtain eight valid, distinct, nontrivial samples for rule {rule['case_id']}")


def sample_key(rule, values):
    """Keep relabelled copies of the same geometric measurements out of other splits."""
    rule_id,v = rule["rule_id"],values
    if rule_id in ('rectangle_perimeter', 'rectangle_area', 'rectangle_diagonal', 'rectangle_diagonal_squared', 'rectangle_aspect_ratio', 'parallelogram_perimeter'): return tuple(sorted((v["x"],v["y"])))
    if rule_id in ('triangle_perimeter', 'triangle_area_heron'): return tuple(sorted((v["x"],v["y"],v["z"])))
    if rule_id == 'triangle_circumradius': return tuple(sorted((v["d"],v["b"],v["c"])))
    if rule_id in ('parallelogram_diagonal', 'parallelogram_area_angle'): return tuple(sorted((v["c"],v["b"])))+(v["θ"],)
    if rule_id in ('rhombus_area_diagonals', 'rhombus_perimeter_diagonals', 'rhombus_side_diagonals'): return tuple(sorted((v["p"],v["q"])))
    if rule_id == 'trapezoid_area': return tuple(sorted((v["c"],v["b"])))+(v["h"],)
    if rule_id == 'trapezoid_perimeter': return tuple(sorted((v["e"],v["b"])))+tuple(sorted((v["c"],v["d"])))
    if rule_id == 'trapezoid_midsegment': return tuple(sorted((v["c"],v["b"])))
    if rule_id == 'isosceles_trapezoid_diagonal': return tuple(sorted((v["x"],v["b"])))+(v["c"],)
    if rule_id in ('cuboid_surface_area', 'cuboid_volume', 'cuboid_edge_sum', 'cuboid_space_diagonal', 'cuboid_space_diagonal_squared', 'cuboid_maximum_sphere_radius', 'cuboid_circumradius'): return tuple(sorted((v["l"],v["w"],v["h"])))
    if rule_id == 'cuboid_lateral_area': return tuple(sorted((v["l"],v["w"])))+(v["h"],)
    if rule_id == 'cuboid_base_area': return tuple(sorted((v["l"],v["w"])))
    return tuple(sorted(v.items()))


def validate_catalog(rules):
    require(len(rules) == 79,"Expected 79 unique rules")
    require([r["case_id"] for r in rules] == list(range(79)),"Invalid rule IDs")
    require(len({r["rule_id"] for r in rules}) == 79,"Repeated rule ID")
    prompts = []
    for rule in rules:
        names = set(rule["variables"]) | set(rule.get("derived",{}))
        placeholders = {field for _,field,_,_ in string.Formatter().parse(rule["context"]) if field}
        require(placeholders == names,f"Context does not declare every input: {rule['case_id']}")
        for expression in (rule["original_expression"],rule["edited_expression"]):
            used = {n.id for n in ast.walk(parse_expression(expression)) if isinstance(n,ast.Name)}
            require(not used-names-{"π"}-set(FUNCTIONS),"Unbound formula symbol")
        views = rule_views(rule)
        prompts.extend(views[v]["prompt"] for v in ("Formula","Description"))
    require(len(set(prompts)) == len(prompts),"Ambiguous identical rule prompts")


def propagate(expression, dependencies):
    terms = []
    for term in dependencies:
        bindings = term["bindings"]
        bound = re.sub(r"\b([a-zA-Zθ])\b",lambda m:f"({bindings[m[1]]})" if m[1] in bindings else m[1],expression)
        factor = term.get("factor",1)
        terms.append(f"{factor} * ({bound})" if factor != 1 else f"({bound})")
    return " + ".join(terms)


def build_hierarchy(cases, rules):
    require([case["case_id"] for case in cases] == list(range(5)), "Expected five unique hierarchical cases")
    by_id = {r["rule_id"]:r for r in rules}
    result = []
    for case in cases:
        rule = copy.deepcopy(by_id[case["rule_id"]])
        rule["edited_expression"] = case.get("edited_expression",rule["edited_expression"])
        row = {"case_id":case["case_id"],"subject":rule["subject"],"rule":rule["rule"],**rule_views(rule)["Formula"]}
        downstream = {"prompt":[],"target_new":[],"ground_truth":[]}
        for question in case["downstream"]:
            q = dict(question,original_expression=propagate(rule["original_expression"],question["dependencies"]),
                     edited_expression=propagate(rule["edited_expression"],question["dependencies"]),derived={})
            views = rule_views(q)["Formula"]
            downstream["prompt"].append(views["prompt"])
            downstream["target_new"].append(views["target_new"])
            downstream["ground_truth"].append(views["ground_truth"][0])
            sample = {"values":question["values"],"witness":question.get("witness",{}),"style":"step"}
            require(geometry_valid(q,sample["values"],sample["witness"]),"Invalid hierarchical geometry")
            require(math.isclose(evaluate_expression(q["original_expression"],sample["values"]),
                                 hierarchy_reference(question),rel_tol=1e-10,abs_tol=1e-10),
                    "Hierarchy dependencies omit or misrepresent a geometric face")
            require(not math.isclose(evaluate_expression(q["original_expression"],sample["values"]),
                                     evaluate_expression(q["edited_expression"],sample["values"])),
                    "Hierarchical counterfactual accidentally equals original")
            rendered = render_instance(q,sample)
            for field in ("prompt","target_new","ground_truth"): downstream[field].append(rendered[field])
        row["hierarchical"] = downstream
        result.append(row)
    return result


def hierarchy_reference(question):
    """Check dependency graphs against ordinary solid geometry, independently."""
    v = question["values"]
    kind = question["reference_kind"]
    if kind == "cube_base": return v["s"]**2
    if kind == "cube_surface": return 6*v["x"]**2
    if kind == "cuboid_base": return v["l"]*v["w"]
    if kind == "cuboid_surface": return 2*(v["l"]*v["w"]+v["l"]*v["h"]+v["w"]*v["h"])
    if kind == "cuboid_lateral": return 2*v["h"]*(v["l"]+v["w"])
    if kind == "pyramid_lateral": return v["P"]*v["l"]/2
    if kind == "triangle_base": return v["b"]*v["h"]/2
    if kind == "equilateral_base":
        side = v.get("s",v.get("x"))
        return math.sqrt(3)*side**2/4
    if kind == "circular_base": return math.pi*v["r"]**2
    raise ValueError(f"Unknown hierarchical reference: {kind}")


def make_locality(rule, rules, views, rng, task):
    result = {}
    for kind,candidates in (("neighborhood",[r for r in rules if r["subject"] == rule["subject"] and r["case_id"] != rule["case_id"]]),
                            ("distracting",[r for r in rules if r["subject"] != rule["subject"]])):
        selected = rng.sample(candidates,2)
        refs = [views[r["case_id"]][("Formula","Description")[i%2] if task == "instance" else task]
                for i,r in enumerate(selected)]
        result[kind] = {"prompt":[r["prompt"] for r in refs],"ground_truth":[r["ground_truth"][0] for r in refs]}
    return result


def pack(rendered, fields):
    return {field:[r[field] for r in rendered] for field in fields}


def as_icl(item):
    icl = copy.deepcopy(item)
    demonstration = item["prompt"][0]+" "+item["target_new"][0]+" "
    for field in ("prompt","rephrase_prompt"): icl[field] = demonstration+item[field][1]
    icl["target_new"] = item["target_new"][1]
    icl["ground_truth"] = item["ground_truth"][1]
    return icl


def build(raw_dir=ROOT/"raw", seed=42):
    raw_dir = Path(raw_dir)
    rules,hierarchy = read_json(raw_dir/"rules.json"),read_json(raw_dir/"hierarchy.json")
    validate_catalog(rules)
    rng = random.Random(seed)
    views = [rule_views(r) for r in rules]
    outputs,sampling = {name:[] for name in FILES},[]
    for rule in rules:
        samples = sample_instances(rule,rng)
        sampling.append({"case_id":rule["case_id"],"samples":samples})
        header = {k:rule[k] for k in ("case_id","subject","rule")}
        rendered = [render_instance(rule,s) for s in samples]
        for name,task,other in ((FILES[0],"Formula","Description"),(FILES[1],"Description","Formula")):
            row = dict(header,**copy.deepcopy(views[rule["case_id"]][task]))
            row["Rule_Understanding"] = copy.deepcopy(views[rule["case_id"]][other])
            row["Instance"] = pack(rendered[:4],("prompt","target_new","ground_truth"))
            row["locality"] = make_locality(rule,rules,views,rng,task)
            outputs[name].append(row)
        row = dict(header,**pack(rendered[4:6],("prompt","target_new","ground_truth","rephrase_prompt")))
        row["Rule_Understanding"] = {
            "prompt":[views[rule["case_id"]][v]["prompt"] for v in ("Formula","Description")],
            "target_new":[views[rule["case_id"]][v]["target_new"] for v in ("Formula","Description")],
            "ground_truth":[views[rule["case_id"]][v]["ground_truth"][0] for v in ("Formula","Description")],
        }
        row["Instance"] = pack(rendered[6:],("prompt","target_new","ground_truth"))
        row["locality"] = make_locality(rule,rules,views,rng,"instance")
        outputs[FILES[2]].append(row)
        outputs[FILES[3]].append(as_icl(row))
    outputs[FILES[4]] = build_hierarchy(hierarchy,rules)
    validate(outputs,rules,sampling,hierarchy)
    sources = [raw_dir/name for name in ("rules.json","hierarchy.json") if (raw_dir/name).exists()]
    manifest = {
        "dataset_version":VERSION,"seed":seed,"rules":len(rules),
        "rules_2d":sum(r["subject"] in PLANE_SHAPES for r in rules),
        "rules_3d":sum(r["subject"] not in PLANE_SHAPES for r in rules),
        "subjects":dict(Counter(r["subject"] for r in rules)),
        "rule_level_instances":4*len(rules),"instance_editing_samples":4*len(rules),
        "hierarchical_cases":len(hierarchy),"hierarchical_questions":sum(len(r["hierarchical"]["prompt"]) for r in outputs[FILES[4]]),
        "raw_sources":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "datasets":{name:{"records":len(rows),"sha256":hashlib.sha256(json_bytes(rows)).hexdigest()} for name,rows in outputs.items()},
        "validation":{"valid_geometric_instances":8*len(rules),"counterfactual_equals_original":0,
                      "repeated_assignments_within_rule":0,"symmetry_equivalent_assignments":0,"editing_portability_overlap":0,
                      "ambiguous_rule_prompts":0,"invalid_locality_references":0,
                      "original_formulas_checked_against_geometry_oracle":len(rules)},
        "sampling":sampling,
        "notes":["Angles are in radians; primitive integers are 1-15, interior angles 1-3 and sector angles 1-6.",
                 "Derived areas and perimeters can exceed the primitive sampling range.",
                 "Formula and description targets share one expression AST; verbal grouping is explicit.",
                 "Eight assignments per rule are disjoint across rule-level evaluation, instance editing and IP, including symmetric side/dimension relabellings.",
                 "Hierarchical labels propagate the lower rule compositionally to every dependent face.",
                 "Exact rational/pi counterfactual results; irrational original results use symbolic substitution and a 12-digit approximation."],
    }
    return outputs,manifest


def validate(outputs, rules, sampling, hierarchy):
    require(set(outputs) == set(FILES),"Missing or unexpected task dataset")
    for name in FILES[:4]: require(len(outputs[name]) == len(rules),"Wrong task record count")
    views = [rule_views(r) for r in rules]
    prompt_index = {views[r["case_id"]][v]["prompt"]:(r,views[r["case_id"]][v]) for r in rules for v in ("Formula","Description")}
    for i,rule in enumerate(rules):
        samples = sampling[i]["samples"]
        require(len(samples) == 8 and len({sample_key(rule,s["values"]) for s in samples}) == 8,
                "Repeated or missing numerical assignments")
        for sample in samples:
            v = sample["values"]
            require(geometry_valid(rule,v,sample["witness"]),"Invalid geometric instance")
            expanded = enrich_values(rule,v)
            original = evaluate_expression(rule["original_expression"],expanded)
            require(math.isclose(original,reference_value(rule,v),rel_tol=1e-10,abs_tol=1e-10),"Original rule contradicts geometry oracle")
            require(not math.isclose(original,evaluate_expression(rule["edited_expression"],expanded),rel_tol=1e-10,abs_tol=1e-10),
                    "Counterfactual equals original for this instance")
        rendered = [render_instance(rule,s) for s in samples]
        f,d,ins,icl = (outputs[name][i] for name in FILES[:4])
        for name in FILES[:4]:
            row = outputs[name][i]
            require(row["case_id"] == i and row["subject"] == rule["subject"] and row["rule"] == rule["rule"],"Invalid task identity")
            require(set(row["locality"]) == {"neighborhood","distracting"},"Invalid locality groups")
            for kind,part in row["locality"].items():
                require(len(part["prompt"]) == len(part["ground_truth"]) == 2,"Unpaired locality")
                ids = []
                for prompt,truth in zip(part["prompt"],part["ground_truth"]):
                    require(prompt in prompt_index,"Unknown locality prompt")
                    neighbor,view = prompt_index[prompt]
                    require(truth == view["ground_truth"][0],"Incorrect locality original target")
                    require(neighbor["case_id"] != i,"Self locality")
                    require((neighbor["subject"] == rule["subject"]) == (kind == "neighborhood"),"Wrong locality scope")
                    ids.append(neighbor["case_id"])
                require(len(set(ids)) == 2,"Repeated locality neighbor")
        for row,view,other in ((f,"Formula","Description"),(d,"Description","Formula")):
            require(all(row[k] == value for k,value in views[i][view].items()),"Formula/description target or prompt mismatch")
            require(row["Rule_Understanding"] == views[i][other],"Incorrect cross-view reference")
            require(row["Instance"] == pack(rendered[:4],("prompt","target_new","ground_truth")),"Incorrect rule-level numerical target")
        require(all(ins[k] == v for k,v in pack(rendered[4:6],("prompt","target_new","ground_truth","rephrase_prompt")).items()),
                "Incorrect editing instance or paraphrase")
        require(ins["Instance"] == pack(rendered[6:],("prompt","target_new","ground_truth")),"Incorrect IP target")
        expected_ru = {"prompt":[views[i][v]["prompt"] for v in ("Formula","Description")],
                       "target_new":[views[i][v]["target_new"] for v in ("Formula","Description")],
                       "ground_truth":[views[i][v]["ground_truth"][0] for v in ("Formula","Description")]}
        require(ins["Rule_Understanding"] == expected_ru,"Instance RU formula/description mismatch")
        require(not set(ins["prompt"]) & set(ins["Instance"]["prompt"]),"Editing/IP overlap")
        require(icl == as_icl(ins),"Inconsistent ICL conversion or missing paraphrase demonstration")
    require(outputs[FILES[4]] == build_hierarchy(hierarchy,rules),"Incorrect compositional hierarchy label")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("build","validate"))
    parser.add_argument("--raw-dir",type=Path,default=ROOT/"raw")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"processed")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--overwrite",action="store_true")
    args = parser.parse_args()
    try:
        src,dst = args.raw_dir.resolve(),args.output_dir.resolve()
        require(src != dst and src not in dst.parents and dst not in src.parents,"Source and output paths must not overlap")
        outputs,manifest = build(args.raw_dir,args.seed)
        artifacts = dict(outputs,**{"manifest.json":manifest})
        if args.command == "validate":
            for name,value in artifacts.items():
                require((dst/name).read_bytes() == json_bytes(value),f"Rebuild mismatch: {name}")
            print("Validated 79 rules, 632 geometric instances and 22 hierarchical questions; all output bytes match.")
        else:
            for name,value in artifacts.items():
                path = dst/name
                require(args.overwrite or not path.exists() or path.read_bytes() == json_bytes(value),
                        f"Refusing to replace changed output {path}; use --overwrite")
            for name,value in artifacts.items(): write_json(dst/name,value)
            print(f"Built RULE-EDIT {VERSION} in {dst} (seed {args.seed}).")
    except (ValueError,KeyError,IndexError,OSError,SyntaxError,ArithmeticError) as error:
        parser.exit(1,f"Error: {error}\n")


if __name__ == "__main__":
    main()
