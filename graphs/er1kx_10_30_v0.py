
import torch
import torch.nn as nn
from src.elementary_modules import Input, Node, Reduce, Output
from numpy import array
import sympy as sym

class Net(nn.Module):
    
    # Information about the original graph               
    meta = {'type': 'er', 'n': 30, 'p': 0.1, 'stages': [10, 10], 'seed1': 1621, 'dag': 'x', 'emb': 'kamada'}
    stages = [10, 10]
    num_nodes = 30
    num_units = 43
    edges = [(0, 1), (0, 2), (1, 2), (2, 3), (0, 4), (1, 4), (3, 4), (4, 5), (2, 6), (3, 6), (4, 6), (5, 6), (6, 7), (3, 8), (5, 8), (5, 9), (6, 10), (4, 11), (9, 12), (8, 13), (9, 13), (11, 14), (12, 14), (7, 15), (14, 16), (15, 16), (9, 17), (13, 18), (16, 18), (10, 19), (15, 19), (18, 19), (11, 20), (13, 20), (17, 20), (16, 21), (18, 21), (20, 21), (14, 22), (17, 22), (20, 22), (21, 22), (19, 23), (20, 24), (22, 24), (15, 25), (19, 26), (23, 26), (24, 26), (20, 27), (22, 27), (20, 28), (27, 28), (25, 29), (26, 29), (28, 29)]
    pos = {0: array([-1.        , -0.02463815]), 1: array([-0.95832211,  0.15710124]), 2: array([-0.91454352, -0.305244  ]), 3: array([-0.66126456, -0.1818607 ]), 4: array([-0.59590662,  0.07268835]), 5: array([-0.51750539,  0.26168601]), 6: array([-0.48506903, -0.27328763]), 7: array([-0.34468673, -0.59161287]), 8: array([-0.34219406,  0.06499026]), 9: array([-0.21703885,  0.54251566]), 10: array([-0.19915759, -0.58849296]), 11: array([-0.17881223,  0.29656542]), 12: array([-0.14724511,  0.806806  ]), 13: array([0.024988  , 0.19263863]), 14: array([0.05858233, 0.43153993]), 15: array([ 0.07943111, -0.48923705]), 16: array([ 0.09707244, -0.05150466]), 17: array([0.19225816, 0.68950875]), 18: array([ 0.20675204, -0.21133735]), 19: array([ 0.23560003, -0.55637647]), 20: array([0.32077736, 0.3106736 ]), 21: array([0.39695711, 0.01688399]), 22: array([0.42202838, 0.51925579]), 23: array([ 0.50533815, -0.7377455 ]), 24: array([0.58215284, 0.11113601]), 25: array([ 0.58403081, -0.5814292 ]), 26: array([ 0.58795652, -0.34444809]), 27: array([0.66723234, 0.51408615]), 28: array([0.75282704, 0.19220634]), 29: array([ 0.84776114, -0.24306751])}
    
    
    def __init__(self, C, restype="C", blocktype="simple", num_outputs=10):
        super().__init__()

        self.restype = restype
        self.blocktype = blocktype
        
        # Neural Network proper
        self.n0   = Input(C, restype=self.restype, blocktype=self.blocktype)
        self.n1   = Node(C, 1, restype=self.restype, blocktype=self.blocktype)
        self.n2   = Node(C, 2, restype=self.restype, blocktype=self.blocktype)
        self.n3   = Node(C, 1, restype=self.restype, blocktype=self.blocktype)
        self.n4   = Node(C, 3, restype=self.restype, blocktype=self.blocktype)
        self.r4x2 = Reduce(C, C*2, 2, restype=self.restype, blocktype=self.blocktype)
        self.n5   = Node(C, 1, restype=self.restype, blocktype=self.blocktype)
        self.n6   = Node(C, 4, restype=self.restype, blocktype=self.blocktype)
        self.r6x2 = Reduce(C, C*2, 2, restype=self.restype, blocktype=self.blocktype)
        self.n7   = Node(C, 1, restype=self.restype, blocktype=self.blocktype)
        self.r7x2 = Reduce(C, C*2, 2, restype=self.restype, blocktype=self.blocktype)
        self.n8   = Node(C, 2, restype=self.restype, blocktype=self.blocktype)
        self.r8x2 = Reduce(C, C*2, 2, restype=self.restype, blocktype=self.blocktype)
        self.n9   = Node(C, 1, restype=self.restype, blocktype=self.blocktype)
        self.r9x2 = Reduce(C, C*2, 2, restype=self.restype, blocktype=self.blocktype)
        self.n10   = Node(C*2, 1, restype=self.restype, blocktype=self.blocktype)
        self.n11   = Node(C*2, 1, restype=self.restype, blocktype=self.blocktype)
        self.r11x2 = Reduce(C*2, C*4, 2, restype=self.restype, blocktype=self.blocktype)
        self.n12   = Node(C*2, 1, restype=self.restype, blocktype=self.blocktype)
        self.n13   = Node(C*2, 2, restype=self.restype, blocktype=self.blocktype)
        self.r13x2 = Reduce(C*2, C*4, 2, restype=self.restype, blocktype=self.blocktype)
        self.n14   = Node(C*2, 2, restype=self.restype, blocktype=self.blocktype)
        self.r14x2 = Reduce(C*2, C*4, 2, restype=self.restype, blocktype=self.blocktype)
        self.n15   = Node(C*2, 1, restype=self.restype, blocktype=self.blocktype)
        self.r15x2 = Reduce(C*2, C*4, 2, restype=self.restype, blocktype=self.blocktype)
        self.n16   = Node(C*2, 2, restype=self.restype, blocktype=self.blocktype)
        self.r16x2 = Reduce(C*2, C*4, 2, restype=self.restype, blocktype=self.blocktype)
        self.n17   = Node(C*2, 1, restype=self.restype, blocktype=self.blocktype)
        self.r17x2 = Reduce(C*2, C*4, 2, restype=self.restype, blocktype=self.blocktype)
        self.n18   = Node(C*2, 2, restype=self.restype, blocktype=self.blocktype)
        self.r18x2 = Reduce(C*2, C*4, 2, restype=self.restype, blocktype=self.blocktype)
        self.n19   = Node(C*2, 3, restype=self.restype, blocktype=self.blocktype)
        self.r19x2 = Reduce(C*2, C*4, 2, restype=self.restype, blocktype=self.blocktype)
        self.n20   = Node(C*4, 3, restype=self.restype, blocktype=self.blocktype)
        self.n21   = Node(C*4, 3, restype=self.restype, blocktype=self.blocktype)
        self.n22   = Node(C*4, 4, restype=self.restype, blocktype=self.blocktype)
        self.n23   = Node(C*4, 1, restype=self.restype, blocktype=self.blocktype)
        self.n24   = Node(C*4, 2, restype=self.restype, blocktype=self.blocktype)
        self.n25   = Node(C*4, 1, restype=self.restype, blocktype=self.blocktype)
        self.n26   = Node(C*4, 3, restype=self.restype, blocktype=self.blocktype)
        self.n27   = Node(C*4, 2, restype=self.restype, blocktype=self.blocktype)
        self.n28   = Node(C*4, 2, restype=self.restype, blocktype=self.blocktype)
        self.n29   = Node(C*4, 3, restype=self.restype, blocktype=self.blocktype)
        self.out   = Output(C*4, num_outputs)

    
    def forward(self, x):
        x0   = self.n0(x)
        x1   = self.n1(x0)
        x2   = self.n2([ x0, x1,])
        x3   = self.n3(x2)
        x4   = self.n4([ x0, x1, x3,])
        x4_2 = self.r4x2(x4)
        x5   = self.n5(x4)
        x6   = self.n6([ x2, x3, x4, x5,])
        x6_2 = self.r6x2(x6)
        x7   = self.n7(x6)
        x7_2 = self.r7x2(x7)
        x8   = self.n8([ x3, x5,])
        x8_2 = self.r8x2(x8)
        x9   = self.n9(x5)
        x9_2 = self.r9x2(x9)
        x10   = self.n10(x6_2)
        x11   = self.n11(x4_2)
        x11_2 = self.r11x2(x11)
        x12   = self.n12(x9_2)
        x13   = self.n13([ x8_2, x9_2,])
        x13_2 = self.r13x2(x13)
        x14   = self.n14([ x11, x12,])
        x14_2 = self.r14x2(x14)
        x15   = self.n15(x7_2)
        x15_2 = self.r15x2(x15)
        x16   = self.n16([ x14, x15,])
        x16_2 = self.r16x2(x16)
        x17   = self.n17(x9_2)
        x17_2 = self.r17x2(x17)
        x18   = self.n18([ x13, x16,])
        x18_2 = self.r18x2(x18)
        x19   = self.n19([ x10, x15, x18,])
        x19_2 = self.r19x2(x19)
        x20   = self.n20([ x11_2, x13_2, x17_2,])
        x21   = self.n21([ x16_2, x18_2, x20,])
        x22   = self.n22([ x14_2, x17_2, x20, x21,])
        x23   = self.n23(x19_2)
        x24   = self.n24([ x20, x22,])
        x25   = self.n25(x15_2)
        x26   = self.n26([ x19_2, x23, x24,])
        x27   = self.n27([ x20, x22,])
        x28   = self.n28([ x20, x27,])
        x29   = self.n29([ x25, x26, x28,])
        return self.out(x29)


    def edge_weights(self):    
        lst = []   # list holding edge weights in the same order as edges
        lst.append(1.0)      # (0, 1)
        lst.append(torch.sigmoid(self.n2.weights[0]).item())    # (0, 2)
        lst.append(torch.sigmoid(self.n2.weights[1]).item())    # (1, 2)
        lst.append(1.0)      # (2, 3)
        lst.append(torch.sigmoid(self.n4.weights[0]).item())    # (0, 4)
        lst.append(torch.sigmoid(self.n4.weights[1]).item())    # (1, 4)
        lst.append(torch.sigmoid(self.n4.weights[2]).item())    # (3, 4)
        lst.append(1.0)      # (4, 5)
        lst.append(torch.sigmoid(self.n6.weights[0]).item())    # (2, 6)
        lst.append(torch.sigmoid(self.n6.weights[1]).item())    # (3, 6)
        lst.append(torch.sigmoid(self.n6.weights[2]).item())    # (4, 6)
        lst.append(torch.sigmoid(self.n6.weights[3]).item())    # (5, 6)
        lst.append(1.0)      # (6, 7)
        lst.append(torch.sigmoid(self.n8.weights[0]).item())    # (3, 8)
        lst.append(torch.sigmoid(self.n8.weights[1]).item())    # (5, 8)
        lst.append(1.0)      # (5, 9)
        lst.append(1.0)      # (6, 10)
        lst.append(1.0)      # (4, 11)
        lst.append(1.0)      # (9, 12)
        lst.append(torch.sigmoid(self.n13.weights[0]).item())    # (8, 13)
        lst.append(torch.sigmoid(self.n13.weights[1]).item())    # (9, 13)
        lst.append(torch.sigmoid(self.n14.weights[0]).item())    # (11, 14)
        lst.append(torch.sigmoid(self.n14.weights[1]).item())    # (12, 14)
        lst.append(1.0)      # (7, 15)
        lst.append(torch.sigmoid(self.n16.weights[0]).item())    # (14, 16)
        lst.append(torch.sigmoid(self.n16.weights[1]).item())    # (15, 16)
        lst.append(1.0)      # (9, 17)
        lst.append(torch.sigmoid(self.n18.weights[0]).item())    # (13, 18)
        lst.append(torch.sigmoid(self.n18.weights[1]).item())    # (16, 18)
        lst.append(torch.sigmoid(self.n19.weights[0]).item())    # (10, 19)
        lst.append(torch.sigmoid(self.n19.weights[1]).item())    # (15, 19)
        lst.append(torch.sigmoid(self.n19.weights[2]).item())    # (18, 19)
        lst.append(torch.sigmoid(self.n20.weights[0]).item())    # (11, 20)
        lst.append(torch.sigmoid(self.n20.weights[1]).item())    # (13, 20)
        lst.append(torch.sigmoid(self.n20.weights[2]).item())    # (17, 20)
        lst.append(torch.sigmoid(self.n21.weights[0]).item())    # (16, 21)
        lst.append(torch.sigmoid(self.n21.weights[1]).item())    # (18, 21)
        lst.append(torch.sigmoid(self.n21.weights[2]).item())    # (20, 21)
        lst.append(torch.sigmoid(self.n22.weights[0]).item())    # (14, 22)
        lst.append(torch.sigmoid(self.n22.weights[1]).item())    # (17, 22)
        lst.append(torch.sigmoid(self.n22.weights[2]).item())    # (20, 22)
        lst.append(torch.sigmoid(self.n22.weights[3]).item())    # (21, 22)
        lst.append(1.0)      # (19, 23)
        lst.append(torch.sigmoid(self.n24.weights[0]).item())    # (20, 24)
        lst.append(torch.sigmoid(self.n24.weights[1]).item())    # (22, 24)
        lst.append(1.0)      # (15, 25)
        lst.append(torch.sigmoid(self.n26.weights[0]).item())    # (19, 26)
        lst.append(torch.sigmoid(self.n26.weights[1]).item())    # (23, 26)
        lst.append(torch.sigmoid(self.n26.weights[2]).item())    # (24, 26)
        lst.append(torch.sigmoid(self.n27.weights[0]).item())    # (20, 27)
        lst.append(torch.sigmoid(self.n27.weights[1]).item())    # (22, 27)
        lst.append(torch.sigmoid(self.n28.weights[0]).item())    # (20, 28)
        lst.append(torch.sigmoid(self.n28.weights[1]).item())    # (27, 28)
        lst.append(torch.sigmoid(self.n29.weights[0]).item())    # (25, 29)
        lst.append(torch.sigmoid(self.n29.weights[1]).item())    # (26, 29)
        lst.append(torch.sigmoid(self.n29.weights[2]).item())    # (28, 29)
        return lst

    
    @staticmethod
    def params_count_sym(restype="C", blocktype="simple"):
        C = sym.symbols('C')
        total = Input.params_count_sym(3,C,restype=restype, blocktype=blocktype)
        total += Node.params_count_sym(C, C, 1, restype=restype, blocktype=blocktype )
        total += Node.params_count_sym(C, C, 2, restype=restype, blocktype=blocktype )
        total += Node.params_count_sym(C, C, 1, restype=restype, blocktype=blocktype )
        total += Node.params_count_sym(C, C, 3, restype=restype, blocktype=blocktype )
        total += Reduce.params_count_sym(C, C*2, restype=restype, blocktype=blocktype)
        total += Node.params_count_sym(C, C, 1, restype=restype, blocktype=blocktype )
        total += Node.params_count_sym(C, C, 4, restype=restype, blocktype=blocktype )
        total += Reduce.params_count_sym(C, C*2, restype=restype, blocktype=blocktype)
        total += Node.params_count_sym(C, C, 1, restype=restype, blocktype=blocktype )
        total += Reduce.params_count_sym(C, C*2, restype=restype, blocktype=blocktype)
        total += Node.params_count_sym(C, C, 2, restype=restype, blocktype=blocktype )
        total += Reduce.params_count_sym(C, C*2, restype=restype, blocktype=blocktype)
        total += Node.params_count_sym(C, C, 1, restype=restype, blocktype=blocktype )
        total += Reduce.params_count_sym(C, C*2, restype=restype, blocktype=blocktype)
        total += Node.params_count_sym(C*2, C*2, 1, restype=restype, blocktype=blocktype )
        total += Node.params_count_sym(C*2, C*2, 1, restype=restype, blocktype=blocktype )
        total += Reduce.params_count_sym(C*2, C*4, restype=restype, blocktype=blocktype)
        total += Node.params_count_sym(C*2, C*2, 1, restype=restype, blocktype=blocktype )
        total += Node.params_count_sym(C*2, C*2, 2, restype=restype, blocktype=blocktype )
        total += Reduce.params_count_sym(C*2, C*4, restype=restype, blocktype=blocktype)
        total += Node.params_count_sym(C*2, C*2, 2, restype=restype, blocktype=blocktype )
        total += Reduce.params_count_sym(C*2, C*4, restype=restype, blocktype=blocktype)
        total += Node.params_count_sym(C*2, C*2, 1, restype=restype, blocktype=blocktype )
        total += Reduce.params_count_sym(C*2, C*4, restype=restype, blocktype=blocktype)
        total += Node.params_count_sym(C*2, C*2, 2, restype=restype, blocktype=blocktype )
        total += Reduce.params_count_sym(C*2, C*4, restype=restype, blocktype=blocktype)
        total += Node.params_count_sym(C*2, C*2, 1, restype=restype, blocktype=blocktype )
        total += Reduce.params_count_sym(C*2, C*4, restype=restype, blocktype=blocktype)
        total += Node.params_count_sym(C*2, C*2, 2, restype=restype, blocktype=blocktype )
        total += Reduce.params_count_sym(C*2, C*4, restype=restype, blocktype=blocktype)
        total += Node.params_count_sym(C*2, C*2, 3, restype=restype, blocktype=blocktype )
        total += Reduce.params_count_sym(C*2, C*4, restype=restype, blocktype=blocktype)
        total += Node.params_count_sym(C*4, C*4, 3, restype=restype, blocktype=blocktype )
        total += Node.params_count_sym(C*4, C*4, 3, restype=restype, blocktype=blocktype )
        total += Node.params_count_sym(C*4, C*4, 4, restype=restype, blocktype=blocktype )
        total += Node.params_count_sym(C*4, C*4, 1, restype=restype, blocktype=blocktype )
        total += Node.params_count_sym(C*4, C*4, 2, restype=restype, blocktype=blocktype )
        total += Node.params_count_sym(C*4, C*4, 1, restype=restype, blocktype=blocktype )
        total += Node.params_count_sym(C*4, C*4, 3, restype=restype, blocktype=blocktype )
        total += Node.params_count_sym(C*4, C*4, 2, restype=restype, blocktype=blocktype )
        total += Node.params_count_sym(C*4, C*4, 2, restype=restype, blocktype=blocktype )
        total += Node.params_count_sym(C*4, C*4, 3, restype=restype, blocktype=blocktype )
        total += Output.params_count_sym(C*4, 10)
        return total, C


    @classmethod
    def get_C(cls, params_count, restype, blocktype):
        total, C = cls.params_count_sym(restype, blocktype)
        sol = sym.solve([total-params_count,C>=0], C)
        return round(sym.solve(sol)[0].evalf())
        
        
    #------------- TESTS -------------#
        
def test():
        test_static()
        for i in [8, 16, 32, 64]:
            net = Net(i)
            test_parameters(net, i)
        
def test_parameters(net, c):
        import numpy as np
        total_params = 0

        for x in filter(lambda p: p.requires_grad, net.parameters()):
            total_params += np.prod(x.data.numpy().shape)
        print("Total number of params", total_params)
        total, C = Net.params_count_sym()
        assert total_params == total.subs(C, c), "symbolic computations do not match true parameters count"
        
def test_static():
        print(Net.meta, Net.stages, Net.num_nodes,
              Net.num_units, Net.edges, Net.pos)

