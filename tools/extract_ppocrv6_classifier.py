#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import helper
from onnx import numpy_helper


@dataclass(frozen=True)
class ClassifierTail:
    hidden_name: str
    matmul_name: str
    matmul_output: str
    add_name: str
    logits_name: str
    weight_name: str
    bias_name: str


def find_classifier_tail(
    graph: onnx.GraphProto,
    *,
    hidden_size: int,
    vocab_size: int,
) -> ClassifierTail:
    initializer_shapes = {init.name: tuple(init.dims) for init in graph.initializer}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    candidates: list[ClassifierTail] = []
    for node in graph.node:
        if node.op_type != "MatMul":
            continue
        weight_names = [
            name
            for name in node.input
            if initializer_shapes.get(name) == (hidden_size, vocab_size)
        ]
        if len(weight_names) != 1:
            continue

        weight_name = weight_names[0]
        hidden_inputs = [name for name in node.input if name != weight_name]
        if len(hidden_inputs) != 1 or len(node.output) != 1:
            continue

        matmul_output = node.output[0]
        for consumer in consumers.get(matmul_output, []):
            if consumer.op_type != "Add" or len(consumer.output) != 1:
                continue
            bias_names = [
                name
                for name in consumer.input
                if initializer_shapes.get(name) == (vocab_size,)
            ]
            if len(bias_names) != 1:
                continue
            candidates.append(
                ClassifierTail(
                    hidden_name=hidden_inputs[0],
                    matmul_name=node.name,
                    matmul_output=matmul_output,
                    add_name=consumer.name,
                    logits_name=consumer.output[0],
                    weight_name=weight_name,
                    bias_name=bias_names[0],
                )
            )

    if not candidates:
        raise RuntimeError(
            f"could not find MatMul/Add classifier tail with "
            f"weight [{hidden_size},{vocab_size}] and bias [{vocab_size}]"
        )
    if len(candidates) > 1:
        details = ", ".join(f"{c.matmul_name}/{c.add_name}" for c in candidates)
        raise RuntimeError(f"found multiple classifier tails: {details}")
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the PP-OCRv6 recognition CTC classifier weight and bias."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/ppocrv6-medium/source/rec/inference.onnx"),
        help="Baseline recognition ONNX that still contains the final classifier.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ppocrv6-medium/derived/classifier.npz"),
        help="Output .npz path.",
    )
    parser.add_argument(
        "--hidden-output",
        type=Path,
        default=Path("artifacts/ppocrv6-medium/derived/rec-hidden.onnx"),
        help="Output recognizer ONNX cut before the classifier.",
    )
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--vocab-size", type=int, default=18710)
    return parser.parse_args()


def value_info_by_name(graph: onnx.GraphProto) -> dict[str, onnx.ValueInfoProto]:
    return {value.name: value for value in [*graph.input, *graph.value_info, *graph.output]}


def make_fallback_hidden_value_info(name: str, hidden_size: int) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(
        name,
        onnx.TensorProto.FLOAT16,
        ["batch", "timesteps", hidden_size],
    )


def cast_output_to_fp16(
    model: onnx.ModelProto,
    *,
    source_name: str,
    hidden_size: int,
) -> None:
    value_info = value_info_by_name(model.graph)
    source_value = value_info.get(
        source_name,
        make_fallback_hidden_value_info(source_name, hidden_size),
    )
    source_type = source_value.type.tensor_type
    if source_type.elem_type == onnx.TensorProto.FLOAT16:
        return

    cast_output_name = f"{source_name}.fp16"
    model.graph.node.append(
        helper.make_node(
            "Cast",
            inputs=[source_name],
            outputs=[cast_output_name],
            name="CastHiddenOutputToFloat16",
            to=onnx.TensorProto.FLOAT16,
        )
    )

    fp16_output = onnx.ValueInfoProto()
    fp16_output.CopyFrom(source_value)
    fp16_output.name = cast_output_name
    fp16_output.type.tensor_type.elem_type = onnx.TensorProto.FLOAT16
    del model.graph.output[:]
    model.graph.output.append(fp16_output)


def prune_to_output(model: onnx.ModelProto, output_name: str, hidden_size: int) -> onnx.ModelProto:
    producer_by_output: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for name in node.output:
            producer_by_output[name] = node

    graph_inputs = {value.name for value in model.graph.input}
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    needed_nodes: set[str] = set()
    needed_tensors = {output_name}

    def visit_tensor(name: str) -> None:
        if name in graph_inputs or name in initializer_names:
            needed_tensors.add(name)
            return
        node = producer_by_output.get(name)
        if node is None:
            needed_tensors.add(name)
            return
        node_key = node.name or "\0".join(node.output)
        if node_key in needed_nodes:
            return
        needed_nodes.add(node_key)
        for input_name in node.input:
            if input_name:
                needed_tensors.add(input_name)
                visit_tensor(input_name)
        for output in node.output:
            if output:
                needed_tensors.add(output)

    visit_tensor(output_name)

    hidden = onnx.ModelProto()
    hidden.CopyFrom(model)
    del hidden.graph.node[:]
    for node in model.graph.node:
        node_key = node.name or "\0".join(node.output)
        if node_key in needed_nodes:
            hidden.graph.node.append(node)

    used_inputs = set()
    used_outputs = {output_name}
    for node in hidden.graph.node:
        used_inputs.update(name for name in node.input if name)
        used_outputs.update(name for name in node.output if name)

    del hidden.graph.initializer[:]
    for initializer in model.graph.initializer:
        if initializer.name in used_inputs:
            hidden.graph.initializer.append(initializer)

    value_info = value_info_by_name(model.graph)
    del hidden.graph.output[:]
    hidden.graph.output.append(
        value_info.get(output_name, make_fallback_hidden_value_info(output_name, hidden_size))
    )

    keep_value_names = (used_inputs | used_outputs) - graph_inputs
    del hidden.graph.value_info[:]
    for item in model.graph.value_info:
        if item.name in keep_value_names and item.name != output_name:
            hidden.graph.value_info.append(item)

    return hidden


def main() -> None:
    args = parse_args()
    model = onnx.load(args.model, load_external_data=True)
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as exc:  # pragma: no cover - best-effort metadata polish
        print(f"warning: ONNX shape inference failed: {exc}")
    graph = model.graph
    tail = find_classifier_tail(
        graph,
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size,
    )

    initializers = {init.name: init for init in graph.initializer}
    weight = numpy_helper.to_array(initializers[tail.weight_name])
    bias = numpy_helper.to_array(initializers[tail.bias_name])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        weight=weight,
        bias=bias,
        hidden_name=np.array(tail.hidden_name),
        matmul_name=np.array(tail.matmul_name),
        matmul_output=np.array(tail.matmul_output),
        add_name=np.array(tail.add_name),
        logits_name=np.array(tail.logits_name),
        weight_name=np.array(tail.weight_name),
        bias_name=np.array(tail.bias_name),
    )

    hidden_model = prune_to_output(model, tail.hidden_name, args.hidden_size)
    cast_output_to_fp16(
        hidden_model,
        source_name=tail.hidden_name,
        hidden_size=args.hidden_size,
    )
    args.hidden_output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(hidden_model, args.hidden_output)

    print(f"model: {args.model}")
    print(f"output: {args.output}")
    print(f"hidden_output: {args.hidden_output}")
    print(f"hidden: {tail.hidden_name}")
    print(f"matmul: {tail.matmul_name} -> {tail.matmul_output}")
    print(f"add: {tail.add_name} -> {tail.logits_name}")
    print(f"weight: {tail.weight_name} {weight.shape} {weight.dtype}")
    print(f"bias: {tail.bias_name} {bias.shape} {bias.dtype}")


if __name__ == "__main__":
    main()
