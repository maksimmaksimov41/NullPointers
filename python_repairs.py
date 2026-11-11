"""Limits on model edits and low-cost candidates for simple Python bugs."""
import ast
from copy import copy, deepcopy
import io
from itertools import product
import re
import tokenize


SAFE_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int",
    "len", "list", "map", "max", "min", "range", "reversed", "round", "set",
    "sorted", "str", "sum", "tuple", "zip",
}
SAFE_METHODS = {
    "append", "capitalize", "casefold", "copy", "count", "endswith", "extend",
    "find", "fromkeys", "get", "index", "isalnum", "isalpha", "isascii", "isdecimal", "isdigit",
    "isnumeric", "isspace", "items", "join", "keys", "lower", "lstrip", "pop",
    "replace", "reverse", "rstrip", "sort", "split", "startswith", "strip",
    "upper", "values", "add", "discard", "setdefault", "update",
}
# Function rewrites permit ordinary calculations and container operations.
PURE_NODES = (
    ast.Assert, ast.Assign, ast.AugAssign, ast.BinOp, ast.BoolOp, ast.Break,
    ast.Call, ast.Compare, ast.Constant, ast.Continue, ast.Dict, ast.DictComp,
    ast.Expr, ast.For, ast.FormattedValue, ast.GeneratorExp, ast.If, ast.IfExp,
    ast.JoinedStr, ast.List, ast.ListComp, ast.Load, ast.Name, ast.Pass,
    ast.Raise, ast.Return, ast.Set, ast.SetComp, ast.Slice, ast.Starred, ast.Store,
    ast.Subscript, ast.Tuple, ast.UnaryOp, ast.While, ast.boolop, ast.cmpop,
    ast.comprehension, ast.keyword, ast.operator, ast.unaryop,
)
ERROR_TYPES = {"IndexError", "KeyError", "TypeError", "ValueError", "ZeroDivisionError"}
OPPOSITES = {
    "<": ">", ">": "<", "<=": ">=", ">=": "<=", "==": "!=", "!=": "==",
    "+=": "-=", "-=": "+=", "min": "max", "max": "min",
    "sorted": "reversed", "reversed": "sorted",
}


def same_syntax(old, new):
    return ast.dump(old) == ast.dump(new)


def declaration(function):
    header = copy(function)
    header.body = []
    return header


def pure_body(function):
    args = function.args
    locals_ = {arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    locals_.update(arg.arg for arg in (args.vararg, args.kwarg) if arg)
    nodes = [node for statement in function.body for node in ast.walk(statement)]
    locals_.update(node.id for node in nodes if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store))
    receivers = set()
    keys = {}
    key_syntax = set()
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            if node.func.attr not in SAFE_METHODS:
                return False
            receivers.add(node.func)
        supports_key = (
            isinstance(node.func, ast.Name) and node.func.id in {"min", "max", "sorted"}
            or isinstance(node.func, ast.Attribute) and node.func.attr == "sort"
        )
        for keyword in node.keywords:
            if not isinstance(keyword.value, ast.Lambda):
                continue
            key = keyword.value
            args = key.args
            if (not supports_key or keyword.arg != "key" or len(args.args) != 1
                    or args.posonlyargs or args.defaults or args.kwonlyargs or args.vararg or args.kwarg):
                return False
            key_syntax.update((key, args, args.args[0]))
            for member in ast.walk(key.body):
                if isinstance(member, ast.Lambda):
                    return False
                keys[member] = args.args[0].arg
    for node in nodes:
        if node in key_syntax:
            continue
        if isinstance(node, ast.Attribute):
            if node not in receivers:
                return False
        elif not isinstance(node, PURE_NODES):
            return False
        if isinstance(node, ast.Name):
            if node.id.startswith("_") and node.id != "_":
                return False
            if node.id != keys.get(node) and node.id not in locals_ | SAFE_BUILTINS | ERROR_TYPES:
                return False
        if isinstance(node, ast.Call) and node.func not in receivers:
            if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_BUILTINS | ERROR_TYPES:
                return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == keys.get(node) or node.func.id in locals_:
                return False
    return True


def acceptable_patch(old, new):
    """Keep module-level code and signatures intact. Validate each changed body."""
    before, after = ast.parse(old), ast.parse(new)
    if len(before.body) != len(after.body):
        return False
    for original, changed in zip(before.body, after.body):
        if same_syntax(original, changed):
            continue
        if not isinstance(original, ast.FunctionDef) or not isinstance(changed, ast.FunctionDef):
            return False
        if not same_syntax(declaration(original), declaration(changed)):
            return False
        if not pure_body(changed):
            return False
    return True


def numbered_lines(source):
    return "\n".join(f"{number}|{line}" for number, line in enumerate(source.splitlines(), 1))


def apply_numbered_edits(source, edits):
    lines = source.splitlines(keepends=True)
    replacements = []
    for edit in edits:
        if not isinstance(edit, list) or len(edit) != 3:
            raise ValueError("Expected [first line, last line, replacement].")
        start, end, replacement = edit
        if type(start) is not int or type(end) is not int or not isinstance(replacement, str):
            raise ValueError("Invalid line edit.")
        if not 1 <= start <= end <= len(lines):
            raise ValueError("Line range is outside the source.")
        if replacement and not replacement.endswith("\n"):
            replacement += "\n"
        replacements.append((start - 1, end, replacement))
    result = []
    offset = 0
    for start, end, replacement in sorted(replacements):
        if start < offset:
            raise ValueError("Line edits overlap.")
        result.extend(lines[offset:start])
        result.append(replacement)
        offset = end
    return "".join(result + lines[offset:])


def conflicting_assertions(test_sources):
    for text in test_sources:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef) or not function.name.startswith("test"):
                continue
            previous = None
            for statement in function.body:
                current = literal_assertion(statement)
                if current is not None and previous is not None:
                    if current[0] == previous[0] and current[1] != previous[1]:
                        return True
                previous = current
    return False


def literal_assertion(statement):
    match statement:
        case ast.Expr(value=ast.Call(
            func=ast.Attribute(value=ast.Name(id="self"), attr="assertEqual"),
            args=[ast.Call(func=ast.Name(id=function), args=args, keywords=[]), expected],
            keywords=[],
        )):
            try:
                positional = []
                for arg in args:
                    if isinstance(arg, ast.Starred):
                        if not isinstance(arg.value, (ast.List, ast.Tuple)):
                            return None
                        positional.extend(arg.value.elts)
                    else:
                        positional.append(arg)
                for arg in positional:
                    ast.literal_eval(arg)
                return (function, [ast.dump(arg) for arg in positional]), ast.literal_eval(expected)
            except (ValueError, TypeError):
                return None
        case _:
            return None


def structural_candidates(function, instruction):
    """Propose only edits supported by both the existing expression and the task."""
    inclusive = re.search(r"\b(?:at least|at most|inclusive|no (?:more|less) than)\b", instruction)
    stable = re.search(r"\b(?:first (?:appearance|occurrence)|first-seen|preserv\w* (?:the )?(?:input|original) order)\b", instruction)
    empty_zero = re.search(r"\b(?:zero|0)\b", instruction) and re.search(r"\bempty\b", instruction)
    backwards = re.search(r"\breverse (?:input|original) order\b", instruction)
    keep_falsy = re.search(r"\b(?:keep|preserve|retain|include)\b[^.?!]{0,40}\b(?:falsy|falsey|0|false)\b", instruction)
    balance_parentheses = (
        re.search(r"\bparentheses\b", instruction)
        and re.search(r"\b(?:balanced|paired)\b", instruction)
        and re.search(r"\bignore other characters\b", instruction)
    )
    clean_comma_parts = (
        re.search(r"\bsplit\b[^.!?]*\bcommas?\b", instruction)
        and re.search(r"\bstrip whitespace\b", instruction)
        and re.search(r"\b(?:discard|drop|remove|omit) empty parts\b", instruction)
    )
    limit_after_blanks = (
        re.search(r"\bnonblank comma-separated (?:pieces|parts)\b", instruction)
        and re.search(r"\bafter blank (?:pieces|parts) are removed\b", instruction)
        and re.search(r"\bnonpositive limit\b", instruction)
    )
    min_without_none = (
        re.search(r"\b(?:smallest|minimum)\b", instruction)
        and re.search(r"\b(?:ignoring|excluding|ignore|exclude) none\b", instruction)
        and re.search(r"\breturn none for empty input or if all (?:entries|values|elements) are none\b", instruction)
    )
    reverse_pairs = (
        re.search(r"\bconsecutive non[- ]?overlapping pairs?\b", instruction)
        and re.search(r"\breverse (?:the two elements|each pair)\b", instruction)
        and re.search(r"\b(?:preserve|keep) (?:any|the) final unpaired element\b", instruction)
        and re.search(r"\breturn a new list\b", instruction)
    )
    isolated_indices = (
        re.search(r"\bindices\b", instruction)
        and re.search(r"\bdiffers? from every immediate neighbou?r that exists\b", instruction)
        and re.search(r"\ba single element qualifies\b", instruction)
        and re.search(r"\bempty input (?:gives|returns) \[\]", instruction)
    )
    for statement in function.body:
        if inclusive:
            for node in ast.walk(statement):
                if isinstance(node, ast.Compare) and len(node.ops) == 1:
                    replacement = {ast.Lt: ast.LtE, ast.Gt: ast.GtE}.get(type(node.ops[0]))
                    if replacement is not None:
                        edited = copy(node)
                        edited.ops = [replacement()]
                        yield node, ast.unparse(edited)
        if not isinstance(statement, ast.Return) or len(function.body) != 1:
            continue
        value = statement.value
        match value:
            case ast.Call(func=ast.Name(id="min"), args=[ast.Name(id=sequence)], keywords=[]) if min_without_none:
                item = "item" if sequence != "item" else "element"
                yield value, f"min(({item} for {item} in {sequence} if {item} is not None), default=None)"
            case ast.Subscript(
                value=ast.Name(id=sequence), slice=ast.Slice(lower=None, upper=None,
                    step=ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1))),
            ) if reverse_pairs:
                index = "index" if sequence != "index" else "offset"
                item = "item" if sequence != "item" else "element"
                yield value, (
                    f"[{item} for {index} in range(0, len({sequence}), 2) "
                    f"for {item} in {sequence}[{index}:{index} + 2][::-1]]"
                )
            case ast.ListComp(
                elt=ast.Name(id=index),
                generators=[ast.comprehension(
                    target=ast.Name(id=target), ifs=[condition], is_async=0,
                    iter=ast.Call(func=ast.Name(id="range"), keywords=[], args=[
                        ast.Constant(value=1),
                        ast.BinOp(left=ast.Call(func=ast.Name(id="len"), args=[ast.Name(id=sequence)], keywords=[]),
                                  op=ast.Sub(), right=ast.Constant(value=1)),
                    ]),
                )],
            ) if isolated_indices and index == target and sequence != index:
                old_condition = ast.parse(
                    f"{sequence}[{index}] != {sequence}[{index} - 1] and {sequence}[{index}] != {sequence}[{index} + 1]",
                    mode="eval",
                ).body
                if ast.dump(condition) == ast.dump(old_condition):
                    yield value, (
                        f"[{index} for {index} in range(len({sequence})) "
                        f"if ({index} == 0 or {sequence}[{index}] != {sequence}[{index} - 1]) "
                        f"and ({index} + 1 == len({sequence}) or {sequence}[{index}] != {sequence}[{index} + 1])]"
                    )
            case ast.ListComp(
                elt=ast.Call(func=ast.Attribute(value=ast.Name(id=stripped), attr="strip"), args=[], keywords=[]),
                generators=[ast.comprehension(target=ast.Name(id=part), ifs=[], is_async=0,
                    iter=ast.Subscript(
                        value=ast.Call(func=ast.Attribute(value=ast.Name(id=text), attr="split"),
                                       args=[ast.Constant(value=",")], keywords=[]),
                        slice=ast.Slice(lower=None, upper=ast.Name(id=limit), step=None)))],
            ) if limit_after_blanks and stripped == part:
                yield value, f"[{part}.strip() for {part} in {text}.split(',') if {part}.strip()][:max(0, {limit})]"
            case ast.Call(
                func=ast.Attribute(value=ast.Call(
                    func=ast.Attribute(value=ast.Name(id=text), attr="strip"), args=[], keywords=[]),
                    attr="split"),
                args=[ast.Constant(value=",")], keywords=[],
            ) if clean_comma_parts:
                part = "part" if text != "part" else "piece"
                yield value, f"[{part}.strip() for {part} in {text}.split(',') if {part}.strip()]"
            case ast.Compare(
                left=ast.Call(func=ast.Attribute(value=ast.Name(id=text), attr="count"),
                              args=[ast.Constant(value="(")], keywords=[]),
                ops=[ast.Eq()],
                comparators=[ast.Call(func=ast.Attribute(value=ast.Name(id=other), attr="count"),
                                      args=[ast.Constant(value=")")], keywords=[])],
            ) if balance_parentheses and text == other and statement.lineno > function.lineno:
                depth = "depth" if text != "depth" else "balance"
                char = "char" if text != "char" else "symbol"
                yield statement, (
                    f"{depth} = 0\n"
                    f"for {char} in {text}:\n"
                    f"    if {char} == '(':\n"
                    f"        {depth} += 1\n"
                    f"    elif {char} == ')':\n"
                    f"        {depth} -= 1\n"
                    f"        if {depth} < 0:\n"
                    "            return False\n"
                    f"return {depth} == 0"
                )
            case ast.Call(
                func=ast.Name(id="sorted" | "list"),
                args=[ast.Call(func=ast.Name(id="set"), args=[ast.Name(id=name)], keywords=[])],
                keywords=[],
            ) if stable:
                if all(re.search(pattern, instruction) for pattern in (r"\btrim\w*\b", r"\blowercase\b", r"\bnonblank\b")):
                    item = "item" if name != "item" else "element"
                    yield value, f"list(dict.fromkeys({item}.strip().lower() for {item} in {name} if {item}.strip()))"
                yield value, f"list(dict.fromkeys({name}))"
            case ast.BinOp(
                left=ast.Call(func=ast.Name(id="sum"), args=[ast.Name(id=left)], keywords=[]),
                op=ast.Div(),
                right=ast.Call(func=ast.Name(id="len"), args=[ast.Name(id=right)], keywords=[]),
            ) if empty_zero and left == right:
                yield value, f"{ast.unparse(value)} if {left} else 0"
            case ast.ListComp() if backwards:
                yield value, f"{ast.unparse(value)}[::-1]"
            case ast.ListComp(elt=ast.Name(id=item)) if keep_falsy:
                edited = deepcopy(value)
                for generator in edited.generators:
                    generator.ifs = [guard for guard in generator.ifs
                                     if not isinstance(guard, ast.Name) or guard.id != item]
                if ast.dump(edited) != ast.dump(value):
                    yield value, ast.unparse(edited)


def source_edit(name, lines, line_starts, node, replacement):
    # AST columns are UTF-8 byte offsets, while Python strings use characters.
    start_line, end_line = node.lineno - 1, node.end_lineno - 1
    start = len(lines[start_line].encode("utf-8")[:node.col_offset].decode("utf-8"))
    end = len(lines[end_line].encode("utf-8")[:node.end_col_offset].decode("utf-8"))
    if isinstance(node, ast.stmt) and "\n" in replacement:
        indentation = lines[start_line][:start]
        if indentation.isspace():
            replacement = replacement.replace("\n", "\n" + indentation)
    return name, line_starts[start_line] + start, line_starts[end_line] + end, replacement


def simple_candidates(sources, report, max_candidates=10, instruction=""):
    """Try a few generic, small corrections in functions named by failing tests."""
    failing = set(re.findall(r"(?m)^test_(\w+)\s+\([^\r\n]*\)\s+\.\.\.\s+(?:FAIL|ERROR)\b", report))
    if not failing:
        return
    alternatives = {}
    for name, source in sources.items():
        try:
            tree = ast.parse(source)
            tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
        except (SyntaxError, tokenize.TokenError):
            continue
        lines = source.splitlines(keepends=True)
        line_starts = [0]
        for line in lines:
            line_starts.append(line_starts[-1] + len(line))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        for function in functions:
            named = function.name in failing
            if not named and (len(sources) != 1 or len(functions) != 1):
                continue
            changes = []
            for token_index, token in enumerate(tokens):
                row, column = token.start
                if not named or not function.body[0].lineno <= row <= function.end_lineno:
                    continue
                replacement = OPPOSITES.get(token.string)
                if token.string in {"True", "False"} and token_index >= 2:
                    if [t.string for t in tokens[token_index-2:token_index]] == ["reverse", "="]:
                        replacement = "False" if token.string == "True" else "True"
                if replacement is not None:
                    start = line_starts[row - 1] + column
                    changes.append((name, start, start + len(token.string), replacement))
            structural = list(structural_candidates(function, instruction.casefold()))
            if any(isinstance(node, ast.Return) for node, _ in structural):
                changes = []
            extra = [source_edit(name, lines, line_starts, node, replacement) for node, replacement in structural]
            if changes or extra:
                alternatives[function.name] = list(dict.fromkeys([*extra[:2], *changes[:4]]))
    if not alternatives or len(alternatives) > 3:
        return
    if any(name not in alternatives for name in failing) and alternatives.keys() & failing:
        return
    for count, edits in enumerate(product(*alternatives.values())):
        if count >= max_candidates:
            return
        updated = sources.copy()
        for name, start, end, new in sorted(edits, key=lambda item: (item[0], -item[1])):
            text = updated[name]
            updated[name] = text[:start] + new + text[end:]
        if all(acceptable_patch(sources[name], updated[name]) for name in {edit[0] for edit in edits}):
            yield updated

