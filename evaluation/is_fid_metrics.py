import torch
import pandas as pd
import numpy as np
from torch.autograd import Function
import scipy
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from src.config import cfg

class FIDandIS(FrechetInceptionDistance):
    def __init__(self, normalize=True):
        super().__init__()
        self.real_features = []
        self.fake_features = []
        self.normalize = normalize
        self.inception

    def update(self, imgs, real=False):  # type: ignore
        """Update the state with extracted features."""
        imgs = (((imgs+1)/2) * 255).byte() if self.normalize else imgs
        features = self.inception(imgs)
        self.orig_dtype = features.dtype
        features = features.double()
        if real:
            self.real_features.append(features)
        else:
            self.fake_features.append(features)

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += imgs.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += imgs.shape[0]

    # def compute_FID(self):
    #     return super().compute().item()
    
    def compute_FID(self):

        x = [y for y in self.real_features]
        real_features = torch.cat(x, dim=0)
        x = [y for y in self.fake_features]
        fake_features = torch.cat(x, dim=0)

        mean_real = real_features.mean(dim=0)
        mean_fake = fake_features.mean(dim=0)

        cov_real = torch.cov(real_features.t())
        cov_fake = torch.cov(fake_features.t())

        return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).item()
    
    def compute_IS(self, real=False):
        # for y in (self.real_features if real else self.fake_features):
        #     print(y.shape)
        x = [y for y in (self.real_features if real else self.fake_features)]
        features = torch.cat(x, dim=0)
        # random permute the features
        #idx = torch.randperm(features.shape[0])
        #features = features[idx]

        # calculate probs and logits
        prob = features.softmax(dim=1)
        log_prob = features.log_softmax(dim=1)

        # # split into groups
        # prob = prob.chunk(self.splits, dim=0)
        # log_prob = log_prob.chunk(self.splits, dim=0)

        # calculate score per split
        mean_prob = prob.mean(dim=0, keepdim=True)
        kl = prob * (log_prob - mean_prob.log())
        kl = kl.sum(dim=1).mean().exp()

        # return mean and std
        return kl.item()

class ConditionalFIDandIS(FrechetInceptionDistance):
    def __init__(self, n_conditions, normalize=True):
        super().__init__()
        self.real_features = {}
        self.fake_features = {}
        self.normalize = normalize
        self.n_conditions = n_conditions

        for i in range(n_conditions):
            self.real_features[i] = []
            self.fake_features[i] = []
            
    def update(self, imgs, cond_idxs, real=False):  # type: ignore
        """Update the state with extracted features."""
        imgs = (((imgs+1)/2) * 255).byte() if self.normalize else imgs
        features = self.inception(imgs)
        for i, img_cond_idx in enumerate(cond_idxs):
            if real:
                self.real_features[img_cond_idx].append(features[i])
            else:
                self.fake_features[img_cond_idx].append(features[i])

    def compute_FID_matrix(self, conditions, index="real", column="fake"):

        matrix = np.zeros((self.n_conditions, self.n_conditions))
        for r_c in range(self.n_conditions):
            x = [y.unsqueeze(0) for y in self.real_features[r_c]]
            real_features = torch.cat(x, dim=0)
            mean_real = torch.mean(real_features, dim=0)
            cov_real = torch.cov(real_features.t())
            for r_f in range(self.n_conditions):
                x = [y.unsqueeze(0) for y in self.fake_features[r_f]]
                fake_features = torch.cat(x, dim=0)

                mean_fake = torch.mean(fake_features, dim=0)

                cov_fake = torch.cov(fake_features.t())

                matrix[r_c][r_f] = _compute_fid(mean_real, cov_real, mean_fake, cov_fake).item()

        index = [f"{index.capitalize()} [{conc}]" for conc in conditions]
        columns = [f"{column.capitalize()} [{conc}]" for conc in conditions]
        df = pd.DataFrame(matrix, columns=columns, index=index)

        return df

    def compute_BCFID(self):
        """Calculate FID score based on accumulated extracted features from the two distributions."""

        cov_real_c = torch.tensor([]).to(self.device)
        cov_fake_c = torch.tensor([]).to(self.device)
        mean_real_c = torch.tensor([]).to(self.device)
        mean_fake_c = torch.tensor([]).to(self.device)
        for k in range(self.n_conditions):
            x = [y.unsqueeze(0) for y in self.real_features[k]]
            real_features = torch.cat(x, dim=0)
            x = [y.unsqueeze(0) for y in self.fake_features[k]]
            fake_features = torch.cat(x, dim=0)

            mean_real_c = torch.concat([mean_real_c, torch.mean(real_features, dim=0, keepdim=True)])
            mean_fake_c = torch.concat([mean_fake_c, torch.mean(fake_features, dim=0, keepdim=True)])

            cov_real_c = torch.concat([cov_real_c, torch.cov(real_features.t()).unsqueeze(0)])
            cov_fake_c = torch.concat([cov_fake_c, torch.cov(fake_features.t()).unsqueeze(0)])

        mean_real = mean_real_c.mean(dim=0)
        mean_fake = mean_fake_c.mean(dim=0)

        cov_real = cov_real_c.mean(dim=0)
        cov_fake = cov_fake_c.mean(dim=0)

        BCFID_score = _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).item()

        for k in range(self.n_conditions):
            x = [y.unsqueeze(0) for y in self.real_features[k]]
            real_features = torch.cat(x, dim=0)
            for k in range(self.n_conditions):
                x = [y.unsqueeze(0) for y in self.fake_features[k]]
                fake_features = torch.cat(x, dim=0)

                mean_real = torch.mean(real_features, dim=0)
                mean_fake = torch.mean(fake_features, dim=0)

                cov_real = torch.cov(real_features.t())
                cov_fake = torch.cov(fake_features.t())

                _compute_fid(mean_real, cov_real, mean_fake, cov_fake).item()

        return BCFID_score
    
    def compute_WCFID(self):
        """Calculate FID score based on accumulated extracted features from the two distributions."""
        fid_c = torch.tensor([]).to(self.device)
        for k in range(self.n_conditions):
            x = [y.unsqueeze(0) for y in self.real_features[k]]
            real_features = torch.cat(x, dim=0)
            x = [y.unsqueeze(0) for y in self.fake_features[k]]
            fake_features = torch.cat(x, dim=0)

            mean_real = torch.mean(real_features, dim=0)
            mean_fake = torch.mean(fake_features, dim=0)
            
            cov_real = torch.cov(real_features.t())
            cov_fake = torch.cov(fake_features.t())


            fid_c = torch.concat([fid_c, _compute_fid(mean_real, cov_real, mean_fake, cov_fake).unsqueeze(0)])

        return fid_c.mean(dim=0).item()
    
    def compute_BCIS(self, real=False):
        kl_d_c = []
        x = []
        for v in self.real_features.values() if real else self.fake_features.values():
            x += [y.unsqueeze(0) for y in v]
        features = torch.cat(x, dim=0)
        prob = features.softmax(dim=1)
        p_y = prob.mean(dim=0, keepdim=True)

        for v in self.real_features.values() if real else self.fake_features.values():
            x = [y.unsqueeze(0) for y in v]
            features = torch.cat(x, dim=0)
            
            p_y_c = features.softmax(dim=1).mean(dim=0)

            kl_d = p_y_c * (p_y_c.log() - p_y.log())
            kl_d = kl_d.sum(dim=1)

            kl_d_c.append(kl_d.item())
        
        is_score = np.exp(np.mean(kl_d_c))
        
        return is_score

    def compute_WCIS(self, real=False):
        kl_d_c = []
        for v in self.real_features.values() if real else self.fake_features.values():
            x = [y.unsqueeze(0) for y in v]
            features = torch.cat(x, dim=0)
            # calculate probs and logits
            p_y_c = features.softmax(dim=1).mean(dim=0)
            p_y_x = features.softmax(dim=1)

            kl_d = p_y_x * (p_y_x.log() - p_y_c.log())
            kl_d = kl_d.sum(dim=1)

            kl_d_c.append(kl_d.mean().item())
        
        is_score = np.exp(np.mean(kl_d_c))
        
        return is_score

class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.

    All credit to `Square Root of a Positive Definite Matrix`_
    """

    @staticmethod
    def forward(ctx, input_data):
        # TODO: update whenever pytorch gets an matrix square root function
        # Issue: https://github.com/pytorch/pytorch/issues/9983
        m = input_data.detach().cpu().numpy().astype(np.float_)
        scipy_res, _ = scipy.linalg.sqrtm(m, disp=False)
        sqrtm = torch.from_numpy(scipy_res.real).to(input_data)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            (sqrtm,) = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input

sqrtm = MatrixSquareRoot.apply
def _compute_fid(mu1, sigma1, mu2, sigma2, eps: float = 1e-6):
    """Adjusted version of Fid Score

    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples
        eps: offset constant - used if sigma_1 @ sigma_2 matrix is singular

    Returns:
        Scalar value of the distance between sets.
    """
    diff = mu1 - mu2

    covmean = sqrtm(sigma1.mm(sigma2))
    # Product might be almost singular
    if not torch.isfinite(covmean).all():
        offset = torch.eye(sigma1.size(0), device=mu1.device, dtype=mu1.dtype) * eps
        covmean = sqrtm((sigma1 + offset).mm(sigma2 + offset))

    tr_covmean = torch.trace(covmean)
    return diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean

