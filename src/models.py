import torch
import pytorch_lightning as pl
from torch import nn
from torch.distributions import MultivariateNormal
from pytorch_wavelets import DWTForward, DWTInverse
import math


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        img_dim,
        input_channels,
        output_channels,
        t,
        k,
        act=nn.LeakyReLU(),
        pool=nn.MaxPool2d(kernel_size=2, stride=2),
    ):
        super().__init__()
        final_hw = img_dim // 16
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=8, kernel_size=3, padding=1),
            act,
            pool,
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2),
            act,
            pool,
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            act,
            pool,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            act,
            pool,
            nn.Flatten(),
            nn.Linear(64 * final_hw * final_hw, output_channels * k * (t * t + t + 1)),
        )
        self.k = k
        self.t = t
        self.n_channels = output_channels

    def forward(self, x):
        z = self.model(x)
        mu = torch.reshape(z[..., : self.k * self.t], (-1, self.k, self.t))
        sigma = torch.reshape(z[..., self.k * self.t : -self.k], (-1, self.k, self.t, self.t))
        sigma = torch.matmul(torch.transpose(sigma, -2, -1), sigma) / 2
        pi = torch.reshape(z[..., -self.k :], (-1, self.k))
        pi = torch.softmax(pi, dim=-1)
        return mu, sigma, pi


class WaveletFeatureExtractor(nn.Module):
    def __init__(
        self,
        input_dim,
        input_channels,
        output_channels,
        n_layers,
        k,
        t,
        act=nn.LeakyReLU(),
        pool=nn.MaxPool1d(kernel_size=4, stride=4),
    ):
        super().__init__()
        # layers = [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU()]

        # for i in range(n_layers):
        #     layers.append(nn.Linear(hidden_dim, hidden_dim))
        #     layers.append(nn.LeakyReLU())

        # layers.append(nn.Linear(hidden_dim, output_channels * k * (t * t + t + 1)))
        hidden_channels = 4
        final_hw = input_dim // (4 ** (n_layers + 1))
        layers = [
            nn.Conv1d(input_channels, hidden_channels, kernel_size=3, padding=1),
            act,
            pool,
        ]

        for i in range(n_layers):
            layers.append(nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=5, padding=2))
            layers.extend([act, pool])
            hidden_channels *= 2

        layers.append(nn.Flatten())
        layers.append(nn.Linear(hidden_channels * final_hw, output_channels * k * (t * t + t + 1)))

        self.model = nn.Sequential(*layers)
        self.k = k
        self.t = t

    def forward(self, x):
        z = self.model(x)
        mu = torch.reshape(z[..., : self.k * self.t], (-1, self.k, self.t))
        sigma = torch.reshape(z[..., self.k * self.t : -self.k], (-1, self.k, self.t, self.t))
        sigma = torch.matmul(torch.transpose(sigma, -2, -1), sigma) / 2
        pi = torch.reshape(z[..., -self.k :], (-1, self.k))
        pi = torch.softmax(pi, dim=-1)
        return mu, sigma, pi


class MixtureModel(nn.Module):
    def __init__(self, mu, sigma, pi, wavelet):
        super().__init__()
        self.register_buffer("mu", mu)  # n * k * c * t
        self.register_buffer("sigma", sigma)  # n * k * c * t * t
        self.register_buffer("pi", pi)  # n * k * c
        self.dist = MultivariateNormal(loc=mu, covariance_matrix=sigma)
        self.wavelet = wavelet

    def density(self, x, per_sample=False):
        """
        :param x: n * t
        :return:
        """
        if not self.wavelet:
            x = torch.atanh(0.99 * x)

        log_probs = self.dist.log_prob(x)  # n * k
        weighted_log_prob = torch.log(self.get_buffer("pi")) + log_probs
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=-1)  # n
        return per_sample_score if per_sample else torch.mean(per_sample_score)

    def forward(self, batch_size: int):
        selected_gaussians = torch.multinomial(self.get_buffer("pi"), batch_size, replacement=True)
        selected_mu = self.get_buffer("mu")[:, selected_gaussians]
        selected_sigma = self.get_buffer("sigma")[:, selected_gaussians]
        selected_dist = MultivariateNormal(loc=selected_mu, covariance_matrix=selected_sigma)
        samples = selected_dist.sample((batch_size,))  # type: ignore
        return samples


class ImageGFN(pl.LightningModule):
    def __init__(self, n_channels, img_dim, output_dim, num_gaussians, lr=1e-3, wavelet=False):
        super().__init__()
        self.img_dim = img_dim
        self.step_size = output_dim
        self.k = num_gaussians
        self.n_channels = n_channels
        self.wavelet = wavelet

        if wavelet:
            self.max_level = int(math.log2(self.img_dim))
            self.ffm = DWTForward(J=self.max_level, wave="haar")
            self.ifm = DWTInverse(wave="haar")

            yl, yh = self.ffm(torch.zeros((1, 1, img_dim, img_dim)))
            self.wave_sizes = [yl.shape] + [e.shape for e in yh]
            dummy_stack, self.y_sizes = self.stack_wavedec(yl, yh)
            self.y_len = dummy_stack.shape[-1]

            self.feature_model = WaveletFeatureExtractor(
                self.y_len,
                n_channels + 2,
                n_channels,
                self.max_level - 1,
                self.k,
                self.step_size,
            )
        else:
            self.feature_model = FeatureExtractor(
                img_dim, n_channels + 2, n_channels, self.step_size, self.k
            )

        self.lr = lr
        self.save_hyperparameters()

    def stack_wavedec(self, yl, yh):
        n, c = yl.shape[0], yl.shape[1]
        lis = [yl.view(n, c, -1)] + [e.view(n, c, -1) for e in yh]
        y_sizes = [e.shape[-1] for e in lis]
        stack = torch.cat(lis, dim=-1)
        return stack, y_sizes

    def unstack_wavedec(self, y):
        ys = torch.split(y, self.y_sizes, dim=-1)
        ys = [e.view(self.wave_sizes[i]) for i, e in enumerate(ys)]
        return ys[0], ys[1:]

    def forward(self):
        """
        Samples a single image.
        :return:
        """
        if self.wavelet:
            out = torch.zeros((1, self.n_channels, self.y_len), device=self.device)
            vis = torch.zeros((1, 1, self.y_len), device=self.device)
        else:
            out = torch.zeros((1, self.n_channels, self.img_dim, self.img_dim), device=self.device)
            vis = torch.zeros((1, 1, self.img_dim, self.img_dim), device=self.device)

        indices = torch.nonzero(vis == 0, as_tuple=True)
        order = torch.randperm(len(vis.flatten()), device=self.device)

        if len(order) % self.step_size != 0:
            pad_length = self.step_size - (
                len(order) % self.step_size
            )  # in case we can't evenly divide, add from the beginning again
            order = torch.cat([order, order[:pad_length]])

        order = order.view((-1, self.step_size))
        selected_indices = [[dim[order[i]] for dim in indices] for i in range(order.shape[0])]

        for i in range(order.shape[0]):
            take = torch.zeros(vis.shape, device=self.device)
            take[selected_indices[i]] = 1
            img = torch.cat([out, vis, take], dim=1)
            mu, sigma, pi = self.feature_model(img)
            sigma *= torch.eye(self.step_size, device=self.device)
            gmm = MixtureModel(mu, sigma, pi, self.wavelet)
            out[selected_indices[i]] = gmm(1).squeeze(0)
            vis[selected_indices[i]] = 1

        if self.wavelet:
            yl, yh = self.unstack_wavedec(out)
            return self.ifm((yl, yh)).squeeze(0)

        return (out.squeeze(0) + 1) * 0.5

    def training_take(self, vis, num_left):
        with torch.no_grad():
            selection = torch.tensor(list(range(num_left)), device=self.device)[: self.step_size]
            indices_left = torch.nonzero(vis == 0, as_tuple=True)
            selected_indices = [row[selection] for row in indices_left]

            if num_left < self.step_size:
                rem = self.step_size - num_left
                indices_fill = torch.nonzero(vis, as_tuple=True)
                fill_selection = torch.randperm(len(indices_fill[0]), device=self.device)[:rem]
                indices_fill = [dim[fill_selection] for dim in indices_fill]
                selected_indices = [
                    torch.cat([orig, fill]) for orig, fill in zip(selected_indices, indices_fill)
                ]

            take = torch.zeros(vis.shape, device=self.device)
            take[selected_indices] = 1

        return take, selected_indices

    def wavelet_likelihood(self, x, batch_idx):
        yl, yh = self.ffm(x)
        y, _ = self.stack_wavedec(yl, yh)

        with torch.no_grad():
            batch_size = x.shape[0]
            eps = self.step_size / self.y_len
            p = torch.rand((batch_size, 1, 1), device=self.device) * (1 - eps) + eps
            vis = (torch.rand((batch_size, 1, self.y_len), device=self.device) > p).float()

        y_hat = torch.masked_fill(y, vis == 0, -1)
        max_steps = (2 * self.y_len) // self.step_size
        ll = 0

        for i in range(max_steps):
            num_left = int(torch.sum(vis == 0))

            if num_left == 0:
                break

            take, selected_indices = self.training_take(vis, num_left)
            inp = torch.cat([y_hat, vis, take], dim=1)
            mu, sigma, pi = self.feature_model(inp)
            sigma *= torch.eye(
                self.step_size, device=self.device
            )  # Diagonal for now - think of workaround later
            gmm = MixtureModel(mu, sigma, pi, self.wavelet)

            y_true = y[selected_indices]  # ground truth
            ll += gmm.density(y_true)

            y_hat[selected_indices] = y_true
            vis[selected_indices] = 1

        return -ll

    def likelihood(self, x, batch_idx):
        x = x * 2 - 1

        with torch.no_grad():
            batch_size = x.shape[0]
            eps = self.step_size / (self.img_dim * self.img_dim)
            p = torch.rand((batch_size, 1, 1, 1), device=self.device) * (1 - eps) + eps
            vis = (
                torch.rand((batch_size, 1, self.img_dim, self.img_dim), device=self.device) > p
            ).float()

        x_hat = torch.masked_fill(x, vis == 0, -1)
        max_steps = (2 * self.img_dim * self.img_dim) // self.step_size
        ll = 0

        for i in range(max_steps):
            num_left = int(torch.sum(vis == 0))

            if num_left == 0:
                break

            take, selected_indices = self.training_take(vis, num_left)
            inp = torch.cat([x_hat, vis, take], dim=1)
            mu, sigma, pi = self.feature_model(inp)  # k * t, k * t * t, k
            sigma *= torch.eye(
                self.step_size, device=self.device
            )  # Diagonal for now - think of workaround later
            gmm = MixtureModel(mu, sigma, pi, self.wavelet)

            x_true = x[selected_indices]  # ground truth
            ll += gmm.density(x_true)

            x_hat[selected_indices] = x_true
            vis[selected_indices] = 1

        return -ll

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss = (
            self.wavelet_likelihood(x, batch_idx) if self.wavelet else self.likelihood(x, batch_idx)
        )

        if loss is None:
            return None

        self.log("train_loss", loss)
        return loss

    # def on_epoch_end(self) -> None:
    #     self.logger.log_image("sample_image", [self.forward().cpu().numpy()])

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        loss = self.likelihood(x, batch_idx)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=10)
        return {"optimizer": optim, "lr_scheduler": scheduler, "monitor": "train_loss"}
