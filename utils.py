from copy import deepcopy

import numpy as np


def hist(img: np.ndarray, symbs: list):
    probs = deepcopy(symbs)
    numel = len(img.flatten())
    for i, it in enumerate(symbs):
        probs[i] = np.sum(img == it) / numel
    new_symbs, new_probs = [], []
    for i in range(len(probs)):
        if probs[i] != 0.:
            new_symbs.append(symbs[i])
            new_probs.append(probs[i])
    return new_symbs, new_probs


def huffman(symbs: list, probs: list):
    if not len(symbs) == len(probs):
        raise ValueError("engths of probs and symbols should be equal")
    HTree = HuffmanTree(symbs, probs)
    return HTree.generate_code()


class Node:
    def __init__(self, symb, prob):
        self.symb = symb
        self.prob = prob
        self.left = None
        self.right = None


class HuffmanTree:
    def __init__(self, symbs, probs):
        self.symbs = symbs
        self.probs = probs
        self._build_tree()

    def _build_tree(self):
        nodes = []
        for i in range(len(self.probs)):
            nodes.append(
                Node(self.symbs[i], self.probs[i])
            )
        while True:
            nodes = sorted(nodes, key=lambda x: x.prob)
            min_node_1 = nodes.pop(0)
            min_node_2 = nodes.pop(0)
            new_node = self._merge_nodes(
                node_1=min_node_1,
                node_2=min_node_2,
            )
            nodes.append(new_node)
            if len(nodes) == 1:
                self.root = nodes[0]
                break

    def _merge_nodes(self, node_1: Node, node_2: Node) -> Node:
        new_node = Node(
            symb=None,
            prob=node_1.prob+node_2.prob,
        )
        if node_1.prob < node_2.prob:
            # Higher weight on the left
            new_node.left, new_node.right = node_1, node_2
        else:
            new_node.left, new_node.right = node_2, node_1
        return new_node

    def generate_code(self) -> dict:
        self.codes = [None] * len(self.symbs)
        self.codes = dict(zip(self.symbs, self.codes))
        prefix = tuple()
        self._traverse(self.root, prefix)
        return self.codes

    def _traverse(self, node: Node, prefix: tuple):
        if node.symb is not None:
            self.codes[node.symb] = prefix
        else:
            self._traverse(node.left, prefix+(0,))
            self._traverse(node.right, prefix+(1,))


def uint2bin(x: int, depth: int):
    if x > (2**depth - 1) or x < 0:
        raise ValueError(
            "x = {} not in range of [0, {}]".format(x, 2**depth - 1))
    byte = tuple()
    while x:
        byte = (x % 2,) + byte
        x //= 2
    if len(byte) < depth:
        byte = (0,) * (depth - len(byte)) + byte
    return byte


class outer:
    def __init__(self, path: str):
        self.ofile = open(path, "wb")
        self.per8counter = 0
        self.current_byte = 0

    def out(self, bits):
        for b in bits:
            self.per8counter += 1
            self.current_byte = (self.current_byte << 1) | b
            if self.per8counter == 8:
                to_out = bytes((self.current_byte,))
                self.ofile.write(to_out)
                self.per8counter = 0
                self.current_byte = 0

    def close(self):
        if self.per8counter != 0:
            self.out(
                (0,) * (8 - self.per8counter)
            )
        self.ofile.close()


class inner:
    def __init__(self, path: str):
        self.ifile = open(path, "rb")
        self.per8counter = 0
        self.current_byte = 0

    def in_(self):
        if self.current_byte == -1:
            return tuple()
        if self.per8counter == 0:
            byte = self.ifile.read(1)
            if len(byte) == 0:
                self.current_byte = -1
                return tuple()
            self.current_byte = byte[0]
            self.per8counter = 8
        self.per8counter -= 1
        return ((self.current_byte >> self.per8counter) & 1,)

    def close(self):
        self.ifile.close()
        self.current_byte = -1
        self.per8counter = 0


def bin2uint(bits: tuple):
    depth = len(bits)
    x = 0
    for b in bits:
        x += (b * 2**(depth-1))
        depth -= 1
    return x
