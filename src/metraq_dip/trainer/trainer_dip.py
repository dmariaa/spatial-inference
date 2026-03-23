import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dateutil.parser import parse
from tqdm import tqdm

from metraq_dip.data.data import collect_data, collect_ensemble_data
from metraq_dip.model.base_model_v2 import Autoencoder3D
from metraq_dip.tools.tools import get_padding, get_interpolation_loss
from metraq_dip.trainer.loss import get_losses


# TODO: this was first pytorch, just take a look maybe it can work in GPU?
#  reconsider the change to numpy
def get_model_output(*, k_output: np.ndarray,
                        k_val_mask: np.ndarray,
                        k_val_data: np.ndarray,
                        k_best_n: int = 10):
    k, c, t, h, w = k_output.shape
    k_output = k_output[:, 0]
    k_val_mask = k_val_mask[:, 0]
    k_val_data = k_val_data[:, 0]

    k_val_count = k_val_mask[0].astype(bool).sum()
    k_y_hat = k_output * k_val_mask
    k_l1_losses = np.sum(np.abs(k_y_hat - k_val_data), axis=(2, 3)) / k_val_count
    k_mse_losses = np.sum((k_y_hat - k_val_data) ** 2, axis=(2, 3)) / k_val_count

    k_best_l1_idx = np.argpartition(k_l1_losses, k_best_n, axis=1)[:, :k_best_n]
    k_best_l1_losses = np.take_along_axis(k_l1_losses, k_best_l1_idx, axis=1)
    k_best_mse_losses = np.take_along_axis(k_mse_losses, k_best_l1_idx, axis=1)

    idx = k_best_l1_idx[:, :, None, None]
    k_best_output = np.mean(np.take_along_axis(k_output, idx, axis=1), axis=(1, 0))

    return k_best_output, k_best_l1_idx


class DipTrainer:
    def __init__(self, configuration: dict):
        self.config = configuration
        self.end_date = parse(configuration.get('date'))
        self.date_window = pd.to_timedelta(configuration.get('hours') - 1, unit='h')
        self.start_date = self.end_date - self.date_window
        self.split = self.config.get('data_split', None)
        self.ensemble_size = self.config.get('ensemble_size')

        self.pollutants = configuration.get('pollutants', None)
        self.test_sensors = configuration.get("test_sensors", None)
        self.normalize = configuration.get('normalize', False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_static_data(self):
        self.static_data = collect_data(
            start_date=self.start_date,
            end_date=self.end_date,
            add_meteo=self.config.get('add_meteo'),
            add_time_channels=self.config.get('add_time_channels'),
            add_coordinates=self.config.get('add_coordinates'),
            add_traffic_data=self.config.get('add_traffic_data'),
            test_sensors=self.test_sensors,
            pollutants=self.pollutants,
        )

    def _get_ensemble_data(self):
        data = collect_ensemble_data(
            data=self.static_data,
            number_of_noise_channels=8,
            number_of_val_sensors=self.config.get('validation_sensors'),
            add_distance_to_sensors=self.config.get('add_distance_to_sensors'),
            normalize=self.config.get('normalize')
        )

        self.x_data = torch.Tensor(data['input_data'])[None, ...].to(self.device)
        self.train_data = torch.Tensor(data['train_data'])[None, ...].to(self.device)
        self.val_data = torch.Tensor(data['val_data'])[None, ...].to(self.device)
        self.test_data = torch.Tensor(data['test_data'])[None, ...].to(self.device)

        self.train_mask = torch.Tensor(data['train_mask'])[None, ...].to(self.device)
        self.val_mask = torch.Tensor(data['val_mask'])[None, ...].to(self.device)

        self.test_mask = torch.Tensor(data['test_mask'])[None, ...].to(self.device)
        self.data = data


    def _get_model(self):
        levels = self.config['model']['levels']
        base_channels = self.config['model']['base_channels']
        preserve_time = self.config['model']['preserve_time']
        skip_connections = self.config['model']['skip_connections']
        neural_upscale = self.config['model']['neural_upscale']

        self.padding = get_padding(self.x_data.shape,
                                   levels=levels,
                                   preserve_time=preserve_time)

        in_channels = self.x_data.shape[1]
        out_channels = self.val_data.shape[1]
        self.model = Autoencoder3D(in_channels=in_channels,
                                   out_channels=out_channels,
                              base_channels=base_channels,
                              levels=levels,
                              preserve_time=preserve_time,
                              use_skip_connections=skip_connections,
                              neural_upscale=neural_upscale
                              ).to(self.device)

    def _get_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
        return optimizer

    def _call_model(self, x: torch.Tensor) -> torch.Tensor:
        _, c, t, h, w = self.x_data.shape
        y_hat = self.model(x)
        return y_hat[:, :c, :t, :h, :w]

    def _get_loss(self, y_hat):
        _, c, t, h, w = self.x_data.shape
        losses = get_losses(self.train_data, y_hat, self.train_mask)
        return losses

    def __calculate_optimization_loss(self, losses):
        loss = losses['L1Loss']
        return loss

    def __do_step(self):
        _, p, t, h, w = self.x_data.shape

        x = F.pad(self.x_data, self.padding)

        self.optimizer.zero_grad()
        # TODO: move padding to the model, it's less efficient as it's done every iteration
        #  but makes more sense, also decide return timestamps in model
        y_hat = self._call_model(x)
        losses = self._get_loss(y_hat)
        loss = self.__calculate_optimization_loss(losses)
        loss.backward()
        self.optimizer.step()

        data = y_hat[0, :, -1, ...].detach().cpu()

        self.dip_logger['train_output'][self.K:self.K + 1, :, self.step:self.step + 1, :, :] = data
        self.dip_logger['train_loss'][self.K:self.K + 1, :, self.step:self.step + 1, 0] = losses['L1Loss'].detach().cpu().item()
        self.dip_logger['train_loss'][self.K:self.K + 1, :, self.step:self.step + 1, 1] = losses['MSELoss'].detach().cpu().item()

        return {'train_mae': losses['L1Loss'].item(), 'train_mse': losses['MSELoss'].item()}

    def __do_validation(self):
        _, p, t, h, w = self.x_data.shape
        x = F.pad(self.x_data, self.padding)

        with torch.no_grad():
            y_hat = self._call_model(x)
            y_hat_val = y_hat

            loss = get_losses(self.val_data[:, :, -1, ...], y_hat_val[:, :, -1, ...], self.val_mask)
            self.dip_logger['val_loss'][self.K:self.K + 1, :, self.step:self.step + 1, 0] = loss['L1Loss'].item()
            self.dip_logger['val_loss'][self.K:self.K + 1, :, self.step:self.step + 1, 1] = loss['MSELoss'].item()

            output = {'val_mae': loss['L1Loss'].item(), 'val_mse': loss['MSELoss'].item()}

            if self.test_mask.any():
                test_loss = get_losses(self.test_data[:, :, -1, ...], y_hat_val[:, :, -1, ...], self.test_mask)
                self.dip_logger['test_loss'][self.K:self.K + 1, :, self.step:self.step + 1, 0] = test_loss['L1Loss'].item()
                self.dip_logger['test_loss'][self.K:self.K + 1, :, self.step:self.step + 1, 1] = test_loss['MSELoss'].item()
                output['test_mae'] = test_loss['L1Loss'].item()
                output['test_mse'] = test_loss['MSELoss'].item()

        return output

    def _do_optimization_loop(self):
        self._get_model()
        self.optimizer = self._get_optimizer()
        self.loss = torch.nn.L1Loss()

        log: dict = {}
        self.step = 0
        with tqdm(total=self.config.get('epochs', 1000), position=2, leave=False) as pbar:
            for self.step in range(self.config.get('epochs', 100)):
                self.model.train()
                log.update(self.__do_step())

                self.model.eval()
                log.update(self.__do_validation())

                pbar.update(1)
                pbar.set_postfix(**log)

    def get_best_result(self):
        k_output = self.dip_logger['train_output'].detach().cpu().numpy()
        k_val_mask = self.dip_logger['val_mask'].detach().cpu().numpy()
        k_test_mask = self.dip_logger['test_mask'].detach().cpu().numpy()
        k_val_data = self.dip_logger['val_data'].detach().cpu().numpy()
        k_test_data = self.dip_logger['test_data'].detach().cpu().numpy()

        model_output, min_idx = get_model_output(k_output=k_output,
                                        k_val_mask=k_val_mask,
                                        k_val_data=k_val_data,
                                        k_best_n=10)

        test_data = k_test_data[0, 0, 0]
        test_mask = k_test_mask[0, 0, 0].astype(bool)
        count = test_mask.sum()
        y_hat = model_output * test_mask
        test_l1_loss = np.sum(np.abs(test_data - y_hat)) / count
        test_mse_loss = np.sum((test_data - y_hat) ** 2) / count

        result = [
            {'pollutant': self.pollutants[0], 'criterion': 'L1Loss_test', 'loss': test_l1_loss},
            {'pollutant': self.pollutants[0], 'criterion': 'MSELoss_test', 'loss': test_mse_loss},
        ]

        return result, model_output, min_idx

    def get_best_result_old(self):
        # Get min validation iterations for K runs
        val_results = self.dip_logger['val_loss']
        min_values, min_idx = val_results[..., 0].min(dim=2)

        # Get model result for the K min iterations
        train_output = self.dip_logger['train_output']
        k, c, t, h, w = train_output.shape
        index = min_idx[:, :, None, None, None].expand(-1, -1, 1, h, w)
        out = train_output.gather(dim=2, index=index).squeeze(2)

        # Average them to get the final "model" result
        # y_hat_final = out.median(dim=0).values
        y_hat_final = out.mean(dim=0)

        # Calculate loss on test sensors, never used for optimization nor validation
        final_test_loss = get_losses(self.test_data[:, :, -1, ...].detach().cpu(), y_hat_final[None, ...], self.test_mask.detach().cpu())

        result = [
            {'pollutant': self.pollutants[0], 'criterion': 'L1Loss_test', 'loss': final_test_loss['L1Loss'].item()},
            {'pollutant': self.pollutants[0], 'criterion': 'MSELoss_test', 'loss': final_test_loss['MSELoss'].item()},
        ]

        return result, y_hat_final, min_idx

    def __call__(self):
        self.dip_logger = None
        self._get_static_data()

        with tqdm(total=self.ensemble_size, position=1) as ensemble_bar:
            for self.K in range(self.ensemble_size):
                self._get_ensemble_data()
                _, c, t, h, w = self.val_data.shape

                if not self.dip_logger:
                    self.dip_logger = {
                        # Model output
                        'train_output': torch.zeros((self.ensemble_size, c, self.config['epochs'], h, w)),
                        'train_loss': torch.zeros((self.ensemble_size, c, self.config['epochs'], 2)),
                        'val_loss': torch.zeros((self.ensemble_size, c, self.config['epochs'], 2)),
                        'test_loss': torch.zeros((self.ensemble_size, c, self.config['epochs'], 2)),

                        # Model data
                        'train_data': torch.zeros((self.ensemble_size, c, 1, h, w)),
                        'val_data': torch.zeros((self.ensemble_size, c, 1, h, w)),
                        'test_data': torch.zeros((self.ensemble_size, c, 1, h, w)),
                        'train_mask': torch.zeros((self.ensemble_size, c, 1, h, w), dtype=torch.bool),
                        'val_mask': torch.zeros((self.ensemble_size, c, 1, h, w), dtype=torch.bool),
                        'test_mask': torch.zeros((self.ensemble_size, c, 1, h, w), dtype=torch.bool)
                    }

                self.dip_logger['train_data'][self.K, 0] = self.train_data[0, 0, -1:]
                self.dip_logger['val_data'][self.K, 0] = self.val_data[0, 0, -1:]
                self.dip_logger['test_data'][self.K, 0] = self.test_data[0, 0, -1:]
                self.dip_logger['train_mask'][self.K, 0] = self.train_mask[0, 0]
                self.dip_logger['val_mask'][self.K, 0] = self.val_mask[0, 0]
                self.dip_logger['test_mask'][self.K, 0] = self.test_mask[0, 0]

                self._do_optimization_loop()
                ensemble_bar.update(1)

        pass


if __name__ == "__main__":
    from metraq_dip.tools.random_tools import set_seed, get_spread_test_groups

    output_folder = R"output"

    test_sensors, _ = get_spread_test_groups(n_groups=1, group_size=4, max_uses_per_sensor=1, magnitudes=[7])

    base_config = {
        'normalize': False,
        'pollutants': [7],
        'add_meteo': False,
        'add_time_channels': False,
        'add_coordinates': False,
        'add_distance_to_sensors': True,
        'date': '20240117T080000',
        'hours': 24,
        'test_sensors': test_sensors[0],
        'validation_sensors': 4,
        'epochs': 250,
        'ensemble_size': 5,
        'lr': 1e-2,
        'model': {
            'base_channels': 16,
            'levels': 3,
            'preserve_time': False,
            'neural_upscale': False,
            'skip_connections': True
        },
    }

    def generate_year_results(config: dict, output_folder: str = 'output'):
        output_file = os.path.join(output_folder, 'year_results_test.csv')
        if os.path.exists(output_file):
            df = pd.read_csv(output_file, parse_dates=['date'])
        else:
            start_date = parse('2024-01-01 00:00:00')
            end_date = parse('2024-12-31 00:00:00') - pd.to_timedelta(config['hours'])
            all_dates = pd.date_range(start=start_date, end=end_date, freq='h')
            df: pd.DataFrame = pd.DataFrame(all_dates, columns=['date'])
            df['processed'] = False
            df['iter'] = 0
            df['DIP_L1Loss'] = 0.0
            df['DIP_MSELoss'] = 0.0
            df['DIP_L1Loss_test'] = 0.0
            df['DIP_MSELoss_test'] = 0.0
            df['KRG_L1Loss'] = 0.0
            df['KRG_MSELoss'] = 0.0
            df['IDW_L1Loss'] = 0.0
            df['IDW_MSELoss'] = 0.0

        # for index, row in df[~df['processed']].sample(frac=1).iterrows():
        for index, row in df[~df['processed']].iterrows():
            date = row['date']
            config['date'] = date.strftime('%Y-%m-%d %H:%M:%S')
            model_results, interpolation_results, it, trainer = generate_date_result(base_config)

            df.loc[index, 'DIP_L1Loss']  = model_results[0]['loss']
            df.loc[index, 'DIP_MSELoss'] = model_results[1]['loss']
            df.loc[index, 'DIP_L1Loss_test']  = model_results[2]['loss']
            df.loc[index, 'DIP_MSELoss_test'] = model_results[3]['loss']
            df.loc[index, 'KRG_L1Loss']  = interpolation_results[0]['loss']
            df.loc[index, 'KRG_MSELoss'] = interpolation_results[1]['loss']
            df.loc[index, 'IDW_L1Loss']  = interpolation_results[2]['loss']
            df.loc[index, 'IDW_MSELoss'] = interpolation_results[3]['loss']
            df.loc[index, 'iter'] = it
            df.loc[index, 'processed'] = True

            df.to_csv(output_file, index=False)

    def generate_date_result(config: dict):
        trainer = DipTrainer(
            configuration=config,
            # seed_func=lambda shape, step: torch.Tensor(np.zeros(shape))
        )

        trainer()
        result, output, min_idx_s = trainer.get_best_result()

        # gather interpolation results
        x_data = (trainer.dip_logger['train_data'] + trainer.dip_logger['val_data'])[0, :, -1:, ...].detach().cpu()
        y_data = trainer.dip_logger['test_data'][0, :, -1:, ...].detach().cpu()
        test_mask = trainer.dip_logger['test_mask'][0, :, -1:, ...].detach().cpu()

        # y_data = trainer.test_data[0, :, -1:, ...].detach().cpu()
        # test_mask = trainer.test_mask[0, :, -1:, ...].detach().cpu()

        interpolation_results = get_interpolation_loss(
            x_data,
            y_data,
            test_mask, trainer.pollutants)

        return result, interpolation_results, output, trainer

    # generate_year_results(config=base_config, output_folder=output_folder)

    set_seed(1000)
    res = generate_date_result(config=base_config)
    l = res[0]
    l.extend(res[1])
    df = pd.DataFrame(l)
    df.loc[df["model"].isna(), ["model"]] = "DIP"
    print(df.head(100))

    # model_results, interpolation_results, trainer = generate_date_result(base_config)
    #
    # df_model = pd.DataFrame.from_dict(model_results)
    # df_model['model'] = 'DIP'
    # df_interpolation = pd.DataFrame.from_dict(interpolation_results)
    # df = pd.concat([df_model, df_interpolation])
    # df.to_csv(os.path.join(output_folder, 'results.csv'), index=False)
