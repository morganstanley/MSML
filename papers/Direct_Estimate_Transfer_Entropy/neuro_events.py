import numpy as np
import numpy.random as npr

import constants

if constants.venv_sites_path is not None:
    import site
    site.addsitedir(constants.venv_sites_path)

import sparse


class NeuroMultivariateTimeseries:

    def events_from_df(self, df, dtype=np.float,
        is_sparse=True,
        num_trials=None,
        trial_ids=None,
        seed=0,
    ):
        if trial_ids is not None:
            assert num_trials is not None

        assert not df.isnull().any().any()
        print(df.shape)

        if num_trials is not None:
            if trial_ids is not None:
                sampled_trial_ids = trial_ids
            else:
                sampled_trial_ids = npr.RandomState(seed=seed).choice(
                    df.trial_ids.drop_duplicates().values,
                    num_trials,
                )
            df = df.loc[df['trial_ids'].isin(sampled_trial_ids)].copy()

        trial_ids = df['trial_ids'].values
        df = df[['probeA', 'probeB', 'probeC', 'probeD', 'probeE', 'probeF']]

        brain_regions = np.array(df.columns)
        if is_sparse:
            activity = sparse.COO.from_numpy(df.values.T.astype(dtype))
        else:
            activity = df.values.T.astype(dtype)
        assert activity.dtype == dtype, activity.dtype

        return activity, brain_regions, trial_ids
