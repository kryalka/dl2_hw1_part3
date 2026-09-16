import torch
import torch.fx as fx
import re
import os 

from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch._subclasses.fake_tensor import FakeTensorMode

from training_ana_val import Trompt


BATCH_SIZE = 8
N_COLUMNS = 984
N_PROMPTS = 128
D_MODEL = 128
N_CYCLES = 6


def get_shape(node):
    value = node.meta.get("val")

    if isinstance(value, torch.Tensor):
        return tuple(value.shape)

    return None


def collect_input_nodes(obj):
    result = []

    if isinstance(obj, fx.Node):
        result.append(obj)

    elif isinstance(obj, (tuple, list)):
        for x in obj:
            result.extend(collect_input_nodes(x))

    elif isinstance(obj, dict):
        for x in obj.values():
            result.extend(collect_input_nodes(x))

    return result


def get_target_name(gm, node):
    if node.op == "call_module":
        module = gm.get_submodule(node.target)
        return f"{node.target} ({module.__class__.__name__})"

    if node.op == "call_function":
        return getattr(node.target, "__name__", str(node.target))

    return str(node.target)

def get_source_line(node):
    stack_trace = node.stack_trace

    if not stack_trace:
        return "-"

    pattern = r'File "([^"]+)", line (\d+), in ([^\n]+)\n\s*(.*)'
    matches = re.findall(pattern, stack_trace)

    if not matches:
        return "-"

    for filename, line_number, function, code in reversed(matches):
        if "site-packages/torch" not in filename:
            filename = os.path.basename(filename)

            return f"{filename}:{line_number} | {code.strip()}"

    return "-"

def make_report(gm):
    lines = []

    header = (
        f"{'NODE':<22}"
        f"{'OP':<15}"
        f"{'TARGET':<30}"
        f"{'INPUTS':<55}"
        f"{'OUTPUT SHAPE':<25}"
        f"{'SOURCE'}"
    )

    lines.append(header)
    lines.append("-" * 210)

    for node in gm.graph.nodes:
        output_shape = get_shape(node)

        input_nodes = (collect_input_nodes(node.args) + collect_input_nodes(node.kwargs))

        inputs = []

        for inp in input_nodes:
            shape = get_shape(inp)

            if shape is not None:
                inputs.append(f"{inp.name}{shape}")
            else:
                inputs.append(inp.name)

        inputs_str = ", ".join(inputs)

        target = get_target_name(gm, node)

        source = get_source_line(node)

        lines.append(
            f"{node.name:<22}"
            f"{node.op:<15}"
            f"{target:<30}"
            f"{inputs_str:<55}"
            f"{str(output_shape):<25}"
            f"{source}"
        )

    return "\n".join(lines)


if __name__ == "__main__":

    model = Trompt(n_columns=N_COLUMNS, n_prompts=N_PROMPTS, d_model=D_MODEL, n_cycles=N_CYCLES,)

    cell = model.tcells[0]

    tracer = fx.Tracer()
    tracer.record_stack_traces = True

    graph = tracer.trace(cell)
    traced = fx.GraphModule(cell, graph)

    x = torch.empty(BATCH_SIZE, N_COLUMNS,)

    prev_cell_out = torch.empty(BATCH_SIZE, N_PROMPTS, D_MODEL,)

    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

    FakeTensorProp(traced, mode=fake_mode,).propagate(x, prev_cell_out,)

    report = make_report(traced)

    print(report)

    with open("shapes.txt", "w") as f:
        f.write(report)