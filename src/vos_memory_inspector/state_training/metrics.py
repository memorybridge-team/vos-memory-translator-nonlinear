"""Record-weighted MSE with source-derived strata and presence disagreement."""
import torch

GROUPS = ('all', 'present', 'absent', 'conditioning', 'nonconditioning',
          'spatial_used', 'spatial_unused')


class Metrics:
    def __init__(self):
        self.values = torch.zeros(len(GROUPS), 3, dtype=torch.float64)
        self.disagreements = 0
        self.unsupported = 0

    def add(self, spatial, pointer, batch):
        s = (spatial.detach().float() - batch['target_spatial'].float()).square().flatten(1).mean(1).cpu().double()
        p = (pointer.detach().float() - batch['target_pointer'].float()).square().mean(1).cpu().double()
        if not torch.isfinite(s).all() or not torch.isfinite(p).all():
            raise ValueError('nonfinite evaluation MSE')
        present = batch['source_present'].cpu()
        cond = batch['conditioning'].cpu()
        used = batch['spatial_used'].cpu()
        masks = (torch.ones_like(present), present, ~present, cond, ~cond, used, ~used)
        for row, mask in zip(self.values, masks):
            row += torch.tensor([float(s[mask].sum()), float(p[mask].sum()), int(mask.sum())], dtype=torch.float64)
        self.disagreements += int((present != batch['target_present'].cpu()).sum())
        self.unsupported += int((~batch['usage_supported'].cpu()).sum())

    def reduce(self, device):
        import torch.distributed as dist
        if dist.is_initialized():
            values = torch.cat((self.values.flatten(), torch.tensor([self.disagreements, self.unsupported], dtype=torch.float64))).to(device)
            dist.all_reduce(values)
            self.values = values[:-2].cpu().reshape(len(GROUPS), 3)
            self.disagreements, self.unsupported = map(int, values[-2:].cpu())

    def report(self, scales):
        result = {}
        for key, (s, p, count) in zip(GROUPS, self.values.tolist()):
            result[key] = dict(records=int(count), spatial_mse=s / count if count else None,
                               pointer_mse=p / count if count else None)
            if count:
                sn, pn = s / count / scales['spatial'] ** 2, p / count / scales['pointer'] ** 2
                result[key].update(normalized_spatial=sn, normalized_pointer=pn, state_loss=sn + pn)
        result['presence_disagreements'] = self.disagreements
        result['usage_unsupported_records'] = self.unsupported
        return result
