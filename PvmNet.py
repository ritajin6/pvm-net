import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
from torchsparse import SparseTensor, PointTensor
import torch_geometric.nn.norm as geo_norm

from network.modules import Intra_aggr_module, Inter_fuse_module
from network.utils import *


class BasicConvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True)
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transpose=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True)
        )

        if inc == outc and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1,
                            stride=stride),
                spnn.BatchNorm(outc)
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class Euc_branch(nn.Module):
    def __init__(self):
        super().__init__()

        # Unet-like structure
        # -------------------------------- Input ----------------------------------------
        self.input_conv = nn.Sequential(
            spnn.Conv3d(3, 32, 3, 1),
            spnn.BatchNorm(32),
            spnn.ReLU(True),
            spnn.Conv3d(32, 32, 3, 1),
            spnn.BatchNorm(32),
            spnn.ReLU(True)
        )

        # -------------------------------- Encoder ----------------------------------------
        # Level 0

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(32, 32, ks=2, stride=2, dilation=1),
            ResidualBlock(32, 64, ks=3, stride=1, dilation=1),
            ResidualBlock(64, 64, ks=3, stride=1, dilation=1)
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(64, 64, ks=2, stride=2, dilation=1),
            ResidualBlock(64, 96, ks=3, stride=1, dilation=1),
            ResidualBlock(96, 96, ks=3, stride=1, dilation=1)
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(96, 96, ks=2, stride=2, dilation=1),
            ResidualBlock(96, 128, ks=3, stride=1, dilation=1),
            ResidualBlock(128, 128, ks=3, stride=1, dilation=1)
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(128, 128, ks=2, stride=2, dilation=1),
            ResidualBlock(128, 160, ks=3, stride=1, dilation=1),
            ResidualBlock(160, 160, ks=3, stride=1, dilation=1)
        )

        self.stage5 = nn.Sequential(
            BasicConvolutionBlock(160, 160, ks=2, stride=2, dilation=1),
            ResidualBlock(160, 192, ks=3, stride=1, dilation=1),
            ResidualBlock(192, 192, ks=3, stride=1, dilation=1)
        )

        self.stage6 = nn.Sequential(
            BasicConvolutionBlock(192, 192, ks=2, stride=2, dilation=1),
            ResidualBlock(192, 224, ks=3, stride=1, dilation=1),
            ResidualBlock(224, 224, ks=3, stride=1, dilation=1)
        )

        # -------------------------------- Decoder ----------------------------------------
        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(224, 192, ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(192 + 192, 192, ks=3, stride=1, dilation=1),
                ResidualBlock(192, 192, ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(192, 160, ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(160 + 160, 160, ks=3, stride=1, dilation=1),
                ResidualBlock(160, 160, ks=3, stride=1, dilation=1),
            )
        ])

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(160, 128, ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(128 + 128, 128, ks=3, stride=1, dilation=1),
                ResidualBlock(128, 128, ks=3, stride=1, dilation=1),
            )
        ])

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(128, 96, ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(96 + 96, 96, ks=3, stride=1, dilation=1),
                ResidualBlock(96, 96, ks=3, stride=1, dilation=1),
            )
        ])

        self.up5 = nn.ModuleList([
            BasicDeconvolutionBlock(96, 64, ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(64 + 64, 64, ks=3, stride=1, dilation=1),
                ResidualBlock(64, 64, ks=3, stride=1, dilation=1),
            )
        ])

        self.up6 = nn.ModuleList([
            BasicDeconvolutionBlock(64, 32, ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(32 + 32, 32, ks=3, stride=1, dilation=1),
                ResidualBlock(32, 32, ks=3, stride=1, dilation=1),
            )
        ])

        # Linear head
        self.output_layer = nn.Sequential(nn.Linear(32, 20))

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(32, 224),
                nn.BatchNorm1d(224),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(224, 160),
                nn.BatchNorm1d(160),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(160, 96),
                nn.BatchNorm1d(96),
                nn.ReLU(True),
            ),

            nn.Sequential(
                nn.Linear(96, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(True),
            )
        ])

        self.weight_initialization()
        self.dropout = nn.Dropout(0.5, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # input
        z = PointTensor(x.F, x.C.float())

        # -------------------------------- Input ----------------------------------------
        x0 = initial_voxelize(z, 0.05, 0.05)
        x0 = self.input_conv(x0)

        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F
        # -------------------------------- Encoder ----------------------------------------
        x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x1)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        x6 = self.stage6(x5)
        z1 = voxel_to_point(x6, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)

        # -------------------------------- Decoder ----------------------------------------
        y1 = point_to_voxel(x6, z1)
        y1.F = self.dropout(y1.F)
        y1 = self.up1[0](y1)
        y1 = torchsparse.cat([y1, x5])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x4])
        y2 = self.up2[1](y2)
        z2 = voxel_to_point(y2, z1)
        z2.F = z2.F + self.point_transforms[1](z1.F)

        y3 = point_to_voxel(y2, z2)
        y3.F = self.dropout(y3.F)
        y3 = self.up3[0](y3)
        y3 = torchsparse.cat([y3, x3])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x2])
        y4 = self.up4[1](y4)
        z3 = voxel_to_point(y4, z2)
        z3.F = z3.F + self.point_transforms[2](z2.F)

        y5 = point_to_voxel(y4, z3)
        y5.F = self.dropout(y5.F)
        y5 = self.up5[0](y5)
        y5 = torchsparse.cat([y5, x1])
        y5 = self.up5[1](y5)

        y6 = self.up6[0](y5)
        y6 = torchsparse.cat([y6, x0])
        y6 = self.up6[1](y6)
        z4 = voxel_to_point(y6, z3)
        z4.F = z4.F + self.point_transforms[3](z3.F)

        # -------------------------------- output ----------------------------------------
        output = self.output_layer(z4.F)

        return output, x6, y1, y2, y3, y4, y5, y6


class Geo_branch(nn.Module):
    def __init__(self):
        super().__init__()

        # -------------------------------- Middle ----------------------------------------
        self.lin_mid_d = spnn.Conv3d(224, 32, kernel_size=1, stride=1, bias=False)

        self.lin_mid_i = nn.Linear(32, 32, bias=False)
        self.mid_geo = Intra_aggr_module(32, 32)
        self.lin_mid_o = nn.Sequential(
            nn.Linear(32, 32, bias=False),
            geo_norm.LayerNorm(32),
            nn.ReLU(True))

        # -------------------------------- Decoder ----------------------------------------
        # Level 5
        self.lin_de5_d = spnn.Conv3d(192, 32, kernel_size=1, stride=1, bias=False)

        self.cd_5 = Inter_fuse_module(32, 32)

        self.lin_de5_i = nn.Linear(96, 32, bias=False)
        self.de5_geo = Intra_aggr_module(32, 32)
        self.lin_de5_o = nn.Sequential(
            nn.Linear(32, 32, bias=False),
            geo_norm.LayerNorm(32),
            nn.ReLU(True))

        # Level 4
        self.lin_de4_d = spnn.Conv3d(160, 32, kernel_size=1, stride=1, bias=False)

        self.cd_4 = Inter_fuse_module(32, 32)

        self.lin_de4_i = nn.Linear(96, 32, bias=False)
        self.de4_geo = Intra_aggr_module(32, 32)
        self.lin_de4_o = nn.Sequential(
            nn.Linear(32, 32, bias=False),
            geo_norm.LayerNorm(32),
            nn.ReLU(True))

        # Level 3
        self.lin_de3_d = spnn.Conv3d(128, 32, kernel_size=1, stride=1, bias=False)

        self.cd_3 = Inter_fuse_module(32, 32)

        self.lin_de3_i = nn.Linear(96, 32, bias=False)
        self.de3_geo = Intra_aggr_module(32, 32)
        self.lin_de3_o = nn.Sequential(
            nn.Linear(32, 32, bias=False),
            geo_norm.LayerNorm(32),
            nn.ReLU(True))

        # Level 2
        self.lin_de2_d = spnn.Conv3d(96, 32, kernel_size=1, stride=1, bias=False)

        self.cd_2 = Inter_fuse_module(32, 32)

        self.lin_de2_i = nn.Linear(96, 32, bias=False)
        self.de2_geo = Intra_aggr_module(32, 32)
        self.lin_de2_o = nn.Sequential(
            nn.Linear(32, 32, bias=False),
            geo_norm.LayerNorm(32),
            nn.ReLU(True))

        # Level 1
        self.lin_de1_d = spnn.Conv3d(64, 32, kernel_size=1, stride=1, bias=False)

        self.cd_1 = Inter_fuse_module(32, 32)

        self.lin_de1_i = nn.Linear(96, 32, bias=False)
        self.de1_geo = Intra_aggr_module(32, 32)
        self.lin_de1_o = nn.Sequential(
            nn.Linear(32, 32, bias=False),
            geo_norm.LayerNorm(32),
            nn.ReLU(True))

        # Level 0
        self.lin_de0_d = spnn.Conv3d(32, 32, kernel_size=1, stride=1, bias=False)

        self.cd_0 = Inter_fuse_module(32, 32)

        self.lin_de0_i = nn.Linear(96, 32, bias=False)
        self.de0_geo = Intra_aggr_module(32, 32)
        self.lin_de0_o = nn.Sequential(
            nn.Linear(32, 32, bias=False),
            geo_norm.LayerNorm(32),
            nn.ReLU(True))

        # Linear head
        self.output_layer = nn.Linear(32, 20, bias=True)

    def forward(self, mesh_data: list, traces: list, x_mid, y_de5, y_de4, y_de3, y_de2, y_de1, y_de0):
        # Geodesic
        mesh_l0 = mesh_data[0]
        mesh_l1 = mesh_data[1]
        mesh_l2 = mesh_data[2]
        mesh_l3 = mesh_data[3]
        mesh_l4 = mesh_data[4]
        mesh_l5 = mesh_data[5]
        mesh_mid = mesh_data[6]

        vertices_l0 = mesh_l0.pos
        vertices_l0 = PointTensor(None, vertices_l0)
        vertices_l1 = mesh_l1.pos
        vertices_l1 = PointTensor(None, vertices_l1)
        vertices_l2 = mesh_l2.pos
        vertices_l2 = PointTensor(None, vertices_l2)
        vertices_l3 = mesh_l3.pos
        vertices_l3 = PointTensor(None, vertices_l3)
        vertices_l4 = mesh_l4.pos
        vertices_l4 = PointTensor(None, vertices_l4)
        vertices_l5 = mesh_l5.pos
        vertices_l5 = PointTensor(None, vertices_l5)
        vertices_mid = mesh_mid.pos
        vertices_mid = PointTensor(None, vertices_mid)

        trace_l1 = traces[0]
        trace_l2 = traces[1]
        trace_l3 = traces[2]
        trace_l4 = traces[3]
        trace_l5 = traces[4]
        trace_mid = traces[5]

        # -------------------------------- Transition Middle ----------------------------------------
        x_mid_d = self.lin_mid_d(x_mid)
        z_mid_d = voxel_to_point(x_mid_d, vertices_mid)
        geo_mid_i = self.lin_mid_i(z_mid_d.F)
        mesh_mid.x = geo_mid_i
        geo_mid_o = self.mid_geo(mesh_mid) + geo_mid_i
        geo_mid_o = self.lin_mid_o(geo_mid_o)
        geo_de5 = geo_mid_o[trace_mid]

        # -------------------------------- Geodesic Decoder ----------------------------------------
        # Level 5
        y_de5_d = self.lin_de5_d(y_de5)
        z_de5 = voxel_to_point(y_de5_d, vertices_l5)
        # Cross domain attention
        geo_de5_at = self.cd_5(geo_de5, z_de5.F, mesh_l5.edge_index)
        geo_de5 = torch.cat([geo_de5, geo_de5_at, z_de5.F], 1)
        geo_de5_i = self.lin_de5_i(geo_de5)
        mesh_l5.x = geo_de5_i
        geo_de5_o = self.de5_geo(mesh_l5) + geo_de5_i
        geo_de5_o = self.lin_de5_o(geo_de5_o)
        geo_de4 = geo_de5_o[trace_l5]

        # Level 4
        y_de4_d = self.lin_de4_d(y_de4)
        z_de4 = voxel_to_point(y_de4_d, vertices_l4)
        # Cross domain attention
        geo_de4_at = self.cd_4(geo_de4, z_de4.F, mesh_l4.edge_index)
        geo_de4 = torch.cat([geo_de4, geo_de4_at, z_de4.F], 1)
        geo_de4_i = self.lin_de4_i(geo_de4)
        mesh_l4.x = geo_de4_i
        geo_de4_o = self.de4_geo(mesh_l4) + geo_de4_i
        geo_de4_o = self.lin_de4_o(geo_de4_o)
        geo_de3 = geo_de4_o[trace_l4]

        # Level 3
        y_de3_d = self.lin_de3_d(y_de3)
        z_de3 = voxel_to_point(y_de3_d, vertices_l3)
        # Cross domain attention
        geo_de3_at = self.cd_3(geo_de3, z_de3.F, mesh_l3.edge_index)
        geo_de3 = torch.cat([geo_de3, geo_de3_at, z_de3.F], 1)
        geo_de3_i = self.lin_de3_i(geo_de3)
        mesh_l3.x = geo_de3_i
        geo_de3_o = self.de3_geo(mesh_l3) + geo_de3_i
        geo_de3_o = self.lin_de3_o(geo_de3_o)
        geo_de2 = geo_de3_o[trace_l3]

        # Level 2
        y_de2_d = self.lin_de2_d(y_de2)
        z_de2 = voxel_to_point(y_de2_d, vertices_l2)
        # Cross domain attention
        geo_de2_at = self.cd_2(geo_de2, z_de2.F, mesh_l2.edge_index)
        geo_de2 = torch.cat([geo_de2, geo_de2_at, z_de2.F], 1)
        geo_de2_i = self.lin_de2_i(geo_de2)
        mesh_l2.x = geo_de2_i
        geo_de2_o = self.de2_geo(mesh_l2) + geo_de2_i
        geo_de2_o = self.lin_de2_o(geo_de2_o)
        geo_de1 = geo_de2_o[trace_l2]

        # Level 1
        y_de1_d = self.lin_de1_d(y_de1)
        z_de1 = voxel_to_point(y_de1_d, vertices_l1)
        # Cross domain attention
        geo_de1_at = self.cd_1(geo_de1, z_de1.F, mesh_l1.edge_index)
        geo_de1 = torch.cat([geo_de1, geo_de1_at, z_de1.F], 1)
        geo_de1_i = self.lin_de1_i(geo_de1)
        mesh_l1.x = geo_de1_i
        geo_de1_o = self.de1_geo(mesh_l1) + geo_de1_i
        geo_de1_o = self.lin_de1_o(geo_de1_o)
        geo_de0 = geo_de1_o[trace_l1]

        # Level 0
        y_de0_d = self.lin_de0_d(y_de0)
        z_de0 = voxel_to_point(y_de0_d, vertices_l0)
        # Cross domain attention
        geo_de0_at = self.cd_0(geo_de0, z_de0.F, mesh_l0.edge_index)
        geo_de0 = torch.cat([geo_de0, geo_de0_at, z_de0.F], 1)
        geo_de0_i = self.lin_de0_i(geo_de0)
        mesh_l0.x = geo_de0_i
        geo_de0_o = self.de0_geo(mesh_l0) + geo_de0_i
        geo_de0_o = self.lin_de0_o(geo_de0_o)

        # -------------------------------- output ----------------------------------------
        output = self.output_layer(geo_de0_o)

        return output


class PvmNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.Euc_branch = Euc_branch()
        self.Geo_branch = Geo_branch()

    def forward(self, colors_v_b, coords_v_b, mesh_data: list, traces: list):
        x = SparseTensor(colors_v_b, coords_v_b)
        out_euc, x_mid, y_de5, y_de4, y_de3, y_de2, y_de1, y_de0 = self.Euc_branch(x)
        out_geo = self.Geo_branch(mesh_data, traces, x_mid, y_de5, y_de4, y_de3, y_de2, y_de1, y_de0)

        return out_euc, out_geo


def model_fn(inference=False):
    def train_fn(model, batch):

        # Load data
        mesh_data = []
        mesh_data += [batch['mesh_l0_b'].to('cuda')]
        mesh_data += [batch['mesh_l1_b'].to('cuda')]
        mesh_data += [batch['mesh_l2_b'].to('cuda')]
        mesh_data += [batch['mesh_l3_b'].to('cuda')]
        mesh_data += [batch['mesh_l4_b'].to('cuda')]
        mesh_data += [batch['mesh_l5_b'].to('cuda')]
        mesh_data += [batch['mesh_mid_b'].to('cuda')]

        traces = []
        traces += [batch['trace_l1_b'].cuda()]
        traces += [batch['trace_l2_b'].cuda()]
        traces += [batch['trace_l3_b'].cuda()]
        traces += [batch['trace_l4_b'].cuda()]
        traces += [batch['trace_l5_b'].cuda()]
        traces += [batch['trace_mid_b'].cuda()]

        coords_v_b = batch['coords_v_b'].cuda()
        colors_v_b = batch['colors_v_b'].cuda()
        labels_v_b = batch['labels_v_b'].cuda()
        labels_m_b = batch['labels_m_b'].cuda()

        # Forward
        out_euc, out_geo = model(colors_v_b, coords_v_b, mesh_data, traces)

        # Loss calculation
        loss = torch.nn.functional.cross_entropy(out_euc, labels_v_b, ignore_index=-100, reduction='mean') + \
               torch.nn.functional.cross_entropy(out_geo, labels_m_b, ignore_index=-100, reduction='mean')

        return loss

    def infer_fn(model, batch):

        # Load data
        mesh_data = []
        mesh_data += [batch['mesh_l0_b'].to('cuda')]
        mesh_data += [batch['mesh_l1_b'].to('cuda')]
        mesh_data += [batch['mesh_l2_b'].to('cuda')]
        mesh_data += [batch['mesh_l3_b'].to('cuda')]
        mesh_data += [batch['mesh_l4_b'].to('cuda')]
        mesh_data += [batch['mesh_l5_b'].to('cuda')]
        mesh_data += [batch['mesh_mid_b'].to('cuda')]

        traces = []
        traces += [batch['trace_l1_b'].cuda()]
        traces += [batch['trace_l2_b'].cuda()]
        traces += [batch['trace_l3_b'].cuda()]
        traces += [batch['trace_l4_b'].cuda()]
        traces += [batch['trace_l5_b'].cuda()]
        traces += [batch['trace_mid_b'].cuda()]

        coords_v_b = batch['coords_v_b'].cuda()
        colors_v_b = batch['colors_v_b'].cuda()

        # Forward
        _, out_geo = model(colors_v_b, coords_v_b, mesh_data, traces)  # 模型分类 out_geo(295386,20)

        # reconstruct original vertices 重建原始顶点
        predictions = out_geo.cpu()  # 复制了一份out_geo(295386,20)
        trace_l0 = batch["trace_l0_b"]
        predictions = predictions[trace_l0]  # (顶点数,20)  20分类 每个点的分类结果

        return predictions

    if inference:
        fn = infer_fn
    else:
        fn = train_fn
    return fn
