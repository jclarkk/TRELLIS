import torch
from easydict import EasyDict as edict

from .bpt.model.model import MeshTransformer
from .bpt.model.serializaiton import BPT_deserialize
from .bpt.utils import joint_filter
from ...modules.sparse import SparseTensor
from .result import MeshExtractResult


class BPTMeshExtractor:
    def __init__(self, device="cuda", res=64, use_color=True):
        super().__init__()
        self.device = device
        self.use_color = use_color
        self.res = res

        # Calculate parameters
        num_discrete_coors = 256
        block_size = 16
        offset_size = 16

        # Initialize BPT model with correct parameters
        self.model = MeshTransformer(
            dim=256,
            max_seq_len=8192,
            attn_depth=8,
            dropout=0.1,
            mode='vertices',
            num_discrete_coors=num_discrete_coors,
            block_size=block_size,
            offset_size=offset_size,
            use_special_block=False,
            conditioned_on_pc=True,
            encoder_name='miche-256-feature'
        ).to(device)

        self._calc_layout()

    def _calc_layout(self):
        # Keep the original layout calculation
        LAYOUTS = {
            'sdf': {'shape': (8, 1), 'size': 8},
            'deform': {'shape': (8, 3), 'size': 8 * 3},
            'weights': {'shape': (21,), 'size': 21}
        }
        if self.use_color:
            LAYOUTS['color'] = {'shape': (8, 6,), 'size': 8 * 6}
        self.layouts = edict(LAYOUTS)
        start = 0
        for k, v in self.layouts.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        self.feats_channels = start

    def get_layout(self, feats: torch.Tensor, name: str):
        # Keep the original layout extraction
        if name not in self.layouts:
            return None
        return feats[:, self.layouts[name]['range'][0]:self.layouts[name]['range'][1]].reshape(-1, *self.layouts[name][
            'shape'])

    def __call__(self, cubefeats: SparseTensor, training=False):
        coords = cubefeats.coords[:, 1:].to(self.device)  # Move coords to device
        feats = cubefeats.feats.to(self.device)  # Move feats to device

        # Extract features using the layout
        sdf, deform, color, weights = [self.get_layout(feats, name)
                                       for name in ['sdf', 'deform', 'color', 'weights']]

        # Combine coordinates with features for BPT input
        pc_features = torch.cat([
            coords,
            sdf.squeeze(-1),
            deform.reshape(deform.shape[0], -1),
            weights,
        ], dim=1).to(self.device)  # Ensure concatenated tensor is on device

        if self.use_color:
            pc_features = torch.cat([
                pc_features,
                color.reshape(color.shape[0], -1)
            ], dim=1).to(self.device)

        # Generate codes using BPT
        codes = self.model.generate(
            batch_size=1,
            temperature=0.5,
            pc=pc_features.cuda().half(),  # Explicitly move to CUDA and convert to half precision
            filter_logits_fn=joint_filter,
            filter_kwargs=dict(k=50, p=0.95),
            return_codes=True
        )

        # Process the codes and create mesh
        vertices = []
        attributes = []
        for code in codes:
            code = code[code != self.model.pad_id].cpu().numpy()
            verts, attrs = BPT_deserialize(
                code,
                block_size=self.model.block_size,
                offset_size=self.model.offset_size,
                use_special_block=self.model.use_special_block
            )
            vertices.append(verts)
            attributes.append(attrs)

        # Create final mesh result
        vertices_tensor = torch.tensor(vertices[0], device=self.device).float()
        faces = torch.arange(1, len(vertices[0]) + 1, device=self.device).view(-1, 3)

        if deform is not None:
            vertices_tensor = vertices_tensor + deform.mean(dim=0)

        return MeshExtractResult(
            vertices=vertices_tensor,
            faces=faces,
            vertex_attrs=torch.tensor(attributes[0], device=self.device) if attributes else None,
            res=self.res
        )
